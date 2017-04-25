function [x,di,fs,summary,opts,info] = readraw_xdf(filename, varargin)
% Function to read a raw-{b,e,g}df file, slice it and return the data and meta-info
%
% [x,di,fs,summary,opts,info] = readraw_xdf(filename, varargin)
% Inputs:
%  filename -- the file to read
opts = struct('prune_equal',0,'epochFn',[],...
              'single',1,'verb',1,'chanIdx',[],'subsample',struct('fs',256),...
              'statusZero_samp',2,'markerCh','Status','isbdf',[]);
[opts,varargin]=parseOpts(opts,varargin);

x = [];
y = [];
fs = [];
di = [];
summary = '';
info=struct('filename',filename); % record file name

hdr=openxdf(filename); % get the header info
info.hdr=hdr;

if any(hdr.SampleRate~=hdr.SampleRate(1))
   error('channels with different sampling rate not supported');
end
fs  = hdr.SampleRate(1);
samp2ms = 1000/fs; 
ms2samp = fs/1000;

% downsampling?
subSampRatio=1;
if ( ~isempty(opts.subsample) )
   if ( isnumeric(opts.subsample) ) nfs=opts.subsample
   else nfs=opts.subsample.fs;
   end
   if ( nfs < fs ) subSampRatio = fs/nfs; end;
end

nchannels = hdr.NS;
nRec      = hdr.NRec;
reclen    = hdr.SPR;
nsamp     = reclen * nRec; 

%TODO: convert to use getEvents_xdf
if ( isempty(hdr.EVENT) && strmatch('EDF Annotations',hdr.Label) ) % use the event stream
  annotch = strmatch('EDF Annotations',hdr.Label,'exact');
  annot=readxdf(hdr,annotch,[],[]);
  annot=[char(bitand(annot(2:2:end),2^8-1));char(bitand(bitshift(annot(1:2:end),-8),2^8-1))];
  annot=annot(:)';
  [onset_s,dur_s,Desc,type,typeDesc,typeIdx]=parseAnnotations(annot);
  hdr.EVENT.pos = round(onset_s*hdr.SampleRate);
  hdr.EVENT.DUR = dur_s * HDR.SampleRate;
  hdr.EVENT.TYP = type;
  hdr.EVENT.Desc= Desc;
  hdr.EVENT.TYP_Desc = typeDesc;
end  

% Extract the status info so we know how to slice the data.  
statusch=[];
if ( ~isempty(hdr.EVENT) && (isfield(hdr.EVENT,'POS') && ~isempty(hdr.EVENT.POS)) )
  status = hdr.EVENT;
else % extract status info from the status channel
  statusch=opts.markerCh;
  if ( isstr(statusch) )
     statusch = strmatch(opts.markerCh,hdr.Label,'exact');
     if ( isempty(statusch) ) 
        statusch = strmatch(upper(opts.markerCh),hdr.Label,'exact');
        if( isempty(statusch) )
           error('Couldnt find a status channel');
        end
     end
  end

  status=readxdf(hdr,statusch,[],[]);
  
  % BODGE! - reprocess biosemi status channel to extract the marker info!
  if ( (~isempty(opts.isbdf) && opts.isbdf) || (isempty(opts.isbdf) && strcmpi(opts.markerCh,'Status')) ) 
     if ( isempty(opts.isbdf) )
        warning('Assuming this is a biosemi gdf-file - so modifying the marker channel info, use isbdf,0 to override.');
     end
     statusinf=int32(status);
     % find indices of negative numbers
     bit24i = find(statusinf < 0 | statusinf>0);
     % make number positive and preserve bits 0-22
     statusinf(bit24i) = bitcmp(abs(statusinf(bit24i))-1);
     % apparently 24 bits reside in 3 higher bytes, shift right 1 byte
     statusinf(bit24i) = bitshift(statusinf(bit24i),-8);
     % re-insert the sign bit on its original location, i.e. bit24
     statusinf(bit24i) = statusinf(bit24i)+(2^(24-1));

     % typecast the data to ensure that the status channel is represented in 32 bits
     statusinf = uint32(statusinf);

     if( isempty(statusinf) ) warning('No data in file: %s',filename); return; end
     % convert to unsigned 24-bit number by: adding 2^23 and masking out bits 25-32
     % N.B. this method loses the value of the 24th bit!
     statusinf=bitand(uint32(statusinf+2^(24-1)),2^24-1); % +24 bit number
     status   =bitand(statusinf,2^16-1);% actual status info in low-order 16 bits
     epoch    =int8(bitget(statusinf,16+1));
     cmrange  =int8(bitget(statusinf,20+1));
     battery  =int8(bitget(statusinf,22+1));
  end
  
  % % convert to a status channel, i.e. hold status value until the next marker comes in
  labtrn=[1 find(status)];
  for i=1:length(labtrn)-1;   
    status(labtrn(i):labtrn(i+1)-1)=status(labtrn(i));
    if( status(labtrn(i))==status(labtrn(i+1)) ) %N.B. -2 so we leave the transition to 0 in place
      status(max(labtrn(i)+1,labtrn(i+1)-1-round(opts.statusZero_samp*subSampRatio)):labtrn(i+1)-1)=0;
    end
  end
  status(labtrn(end)+1:end)=status(labtrn(end)); % last marker to eof is same phase
end

% Now we can extract the bits we want
epochFn=opts.epochFn;
if ( isempty(epochFn) ) 
  if ( isstruct(status) ) epochFn='eventEpochFn';
  else                    epochFn='compRecPhases';
  end
end  
[events,bgns,ends,offset_samp,opts.epOpts]=feval(epochFn,status,varargin{:},'fs',fs);
  
  
if ( isempty(bgns) || isempty(ends) ) 
   warning('File: %s didnt contain any valid epochs',filename);
   return;
end

trlens=ends-bgns+1;
% BODGE: check for badly finished file with over-long last trial
longTr=find(trlens>10*median(trlens));
if ( ~isempty(longTr) )
   warning(sprintf('%d **Very** long trial(s) found, may be CORRUPT. Deleted',numel(longTr)));
   epType(longTr)=[];bgns(longTr)=[];ends(longTr)=[];
end
maxtrlen=max(trlens(:));
if ( opts.prune_equal )
   maxtrlen=min(trlens(:));
   ends  =bgns+maxtrlen-1;           % set end to min length
end

%---------------------------------------------------------------------------
% Finally: Extract the bits we want
%---------------------------------------------------------------------------
ntrials = numel(bgns);
fprintf('Identified %d epochs to read.\n',ntrials);
chanIdx = opts.chanIdx; if ( isempty(chanIdx) ) chanIdx=1:nchannels; end;
if ( islogical(chanIdx) ) chanIdx=find(chanIdx); end;
if ( ~isempty(statusch) && ~any(chanIdx==statusch) ) chanIdx(end+1)=statusch; end;
if ( isfield(hdr,'Chan_Select') )
   hdr.Chan_Select(:)=0; hdr.Chan_Select(chanIdx)=1;
end
nchannels=numel(chanIdx);
y = events;

% downsampling?
omaxtrlen=maxtrlen; ooffset_samp=offset_samp;
if ( subSampRatio>1 )
   ofs=fs; 
   fs=fs/subSampRatio;
   fprintf('Downsampling: from %d to %d.\n',ofs,fs);
   maxtrlen   = floor(maxtrlen./subSampRatio);
   offset_samp= floor(offset_samp./subSampRatio);
end
if ( opts.single ) x = zeros(nchannels, maxtrlen, ntrials, 'single');
else               x = zeros(nchannels, maxtrlen, ntrials);
end
for tr=1:ntrials;
   bgntr= bgns(tr);
   endtr= min(max(bgns(tr)+omaxtrlen,ends(tr)),nsamp); % use all data available, i.e. don't 0-pad
   xtr  = readxdf(hdr,chanIdx,bgntr,endtr);
   if ( ~isstruct(status) )     
     str  = single(status(bgntr:endtr-1)); %hack use endtr-1 instead because readxdf does the same N.B. use fixed status channel
   else
     if ( ~isempty(statusch) ) str = xtr(statusch,:); end;
   end
   if ( subSampRatio>1 ) % sub-sample
      [xtr,idx] = subsample(xtr,size(xtr,2)./subSampRatio,2);
      if ( ~isempty(statusch) ) 
        ostr=str; str=zeros(1,size(xtr,2)); % resample, preserving markers
        for mi=1:numel(idx)-1; str(mi)=max(ostr(idx(mi)+1:idx(mi+1))); end; str(numel(idx))=ostr(idx(end));
      end
   end
   if( ~opts.single ) xtr=single(xtr); end;
   x(:,1:size(xtr,2),tr)  =xtr;
   if ( ~isempty(statusch) ) x(statusch,1:size(xtr,2),tr)=str; end;
   if ( opts.verb > 0 ) % progress reporting
      si=round(tr/(ntrials/100)); % nearest step
      if ( tr==round(si*ntrials/100) ) 
         fprintf(repmat('.',max(1,si-round((tr-1)./(ntrials/100))),1)); 
      end
   end;
end

% Now setup the dim info array
if ( isfield(hdr,'Label') ) % use names for the channels if available 
   chvals=cellstr(hdr.Label);
   % clean up the labels to remove extra 0 characters...
   for i=1:numel(chvals); 
      e=find(chvals{i}==0,1); if ( ~isempty(e) ) chvals{i}=chvals{i}(1:min(end,e-1)); end; 
   end
   if ( ~isempty(chanIdx) ) chvals=chvals(chanIdx); end;
else
   chvals=opts.chanIdx;
end
di = mkDimInfo(size(x),'ch','',chvals,...
               'time','ms',([1:maxtrlen]+offset_samp(1))*samp2ms*subSampRatio,...
               'epoch','',[],'','uV');
di(2).info =struct('fs',fs); if ( isfield(info,'ofs') ) di(2).info.ofs=ofs; end;
% restrict the states to the start of the trials too & record as trialinfo
for tr=1:ntrials;
   di(3).extra(tr).sampleIdx = [bgns(tr) ends(tr) -offset_samp(min(end,tr),:)];
   di(3).extra(tr).marker    = y(tr);
   % events which happened during this time
   if ( ~isempty(hdr.EVENT) && ~isempty(hdr.EVENT.POS) )
      ei = hdr.EVENT.POS>bgns(tr) & hdr.EVENT.POS+hdr.EVENT.DUR < ends(tr); 
      di(3).extra(tr).events    = ...
          struct('typ',hdr.EVENT.TYP(ei),'pos',hdr.EVENT.POS(ei),'dur',hdr.EVENT.DUR(ei));
   end
end

% for summary only rec which states we match
if ( ~(iscell(y) || isstruct(y)) )
  summary = [ 'Status= ' ivec2str(unique(y))];
end

return;

function [str]=ivec2str(chr,vec); % pretty print vector of ints
if ( nargin < 2 ) vec=chr; chr=' '; end;
str='';
if(numel(vec)>1) 
   str=sprintf(['%d' chr],vec(1:end-1));
   str=['[' str sprintf('%d',vec(end)) ']'];
else
   str=sprintf('%d',vec);
end;
return;


%--------------------------------------------------------------------------------
function testCase()
[x,di,fs,summary,opts,info] = readraw_xdf(fname,'RecPhaseLimits',[769:772])