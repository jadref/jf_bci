function [dat,hdr]=readbdf(hdr, chanindx, bgnsamp, endsamp, singlep)
% Read data from a BDF file
%
%   [dat] = readbdf(hdr, chanindx, begsamp, endsamp);
% Inputs:
%    hdr      -- header structure, as returned by bdfopen
%    chanindx -- index of channels to read (optional, default is all)
%    bgnsamp  -- index of the first sample to read (1)
%    endsamp  -- index of the last sample to read  (end)
%    singlep  -- [bool] store result in single precision? (true)
% Outputs:
%    dat      -- Nchans X Nsamples data matrix
if ( nargin < 2 || isempty(chanindx) ) chanindx=1:hdr.NS; end;
if ( nargin < 3 || isempty(bgnsamp) ) bgnsamp=1; end;
if ( nargin < 4 || isempty(endsamp) ) 
   endsamp=hdr.Dur * hdr.SampleRate(1)*hdr.NRec; 
   maxSamp=floor(intmax('int32')/numel(chanindx)/2/(32/8)); % max num samples in ram
   if (endsamp > maxSamp) 
      warning('To big to allocate! truncating to fit in ram');
      endsamp = maxSamp;
   end
end
if ( nargin < 5 || isempty(singlep) ) singlep=true; end;

% get the header/file and parse the inputs
filename    = hdr.FileName;
try fseek(hdr.FILE.FID,0,0);
   % OK, file already open.
   fid = hdr.FILE.FID;
catch
   % open the file ourselves
   fid = fopen(hdr.FileName,'r','ieee-le');
   hdr.FILE.FID = fid;
   if ( fid < 0 ) 
      error('Could not open %s',filename);
   end
end

% convert strings to channel numbers if needed.
if ( isstr(chanindx) ) chanindx={chanindx}; end;
if ( iscell(chanindx) )
   tmp=chanindx; chanindx=zeros(numel(chanindx),1);
   for i=1:numel(chanindx); t=strmatch(tmp{i},hdr.Label); if(~isempty(t)) chanindx(i)=t(1); end; end;
end
% determine the trial containing the begin and end sample
reclen = hdr.SPR(chanindx(1)); % samples per record
if ( numel(hdr.SPR) > 1  && ~all(hdr.SPR(chanindx)==reclen) ) 
   error('Different record lengths in different electrodes');
end
bgnrec = max(1,floor((bgnsamp-1)/reclen) + 1);
endrec = min(floor((endsamp-1)/reclen) + 1,hdr.NRec);
nrecs  = endrec - bgnrec + 1;
nchans = hdr.NS;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% allocate memory to hold the data
if ( singlep ) 
   dat = zeros(length(chanindx),nrecs*reclen,'single');
else
   dat = zeros(length(chanindx),nrecs*reclen,'double');
end

% read and concatenate all required data recs
for i=bgnrec:endrec
   offset = hdr.HeadLen + (i-1)*reclen*nchans*3;
   recIdx = ((i-bgnrec)*reclen+1):((i-bgnrec+1)*reclen);
   if length(chanindx)==1
      % this is more efficient if only one channel has to be read, e.g. the status channel
      offset = offset + (chanindx-1)*reclen*3;
      fseek(fid,offset,'bof');
      [buf,nrd] = fread(fid,reclen,'bit24');
      if ( nrd ~= reclen ) 
         warning('File is incomplete -- truncating'); 
         endsamp=recIdx(1);
         break;
      end;
%      buf = read_24bit(filename, offset, reclen);
      dat(:,recIdx) = buf';
   else
      % read the data from all channels and then select the desired channels
      fseek(fid,offset,'bof');
      [buf,nrd] = fread(fid,reclen*nchans,'bit24');
      if ( nrd ~= reclen*nchans ) 
         warning('File is incomplete -- truncating');
         endsamp=recIdx(1);
         break;                  
      end;
%      buf = read_24bit(filename, offset, reclen*nchans);
      buf = reshape(buf, reclen, nchans);
      dat(:,recIdx) = buf(:,chanindx)';
   end
end

% select the desired samples
bgnsamp = bgnsamp - (bgnrec-1)*reclen;  % correct for the number of bytes that were skipped
endsamp = endsamp - (bgnrec-1)*reclen;  % correct for the number of bytes that were skipped
dat = dat(:, bgnsamp:endsamp);

% Calibrate the data
if length(chanindx)>1
   % using a sparse matrix speeds up the multiplication
   calib = hdr.Cal(chanindx,:);
   for ch=1:size(dat,1); dat(ch,:) = calib(ch) * dat(ch,:); end;
else
   % in case of one channel the calibration would result in a sparse array
   calib = hdr.Cal(chanindx,:);
   dat   = calib * dat;
end

if( nargout < 2 ) fclose(fid); end;
return;