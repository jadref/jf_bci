function [X,di,fs,summary,opts,info]=readraw_bciobj(filename,varargin)
opts=struct('label','','fs',[],'single',0);
opts=parseOpts(opts,varargin);

z=load(filename);% load data-files
summary=z.desc_oneline;
z=z.z; 

% extract the dim-info
X=z.x;
if ( opts.single && isa(X,'double') ) X=single(X); end;
zdi=z.dim;
% correct the timing info to be 0 at the stim
di = mkDimInfo(size(z.x),...
               z.dim(1).name,z.dim(1).units,z.dim(1).vals,...
               z.dim(2).name,z.dim(2).units,z.dim(2).vals,...
               z.dim(3).name,z.dim(3).units,z.dim(3).vals);

% map names to the jf names.
znames={zdi.label};

% Fill out the epoch info.
epochDim=strmatch('trial',znames);
di(epochDim).name = 'epoch'; % info about each trial
if ( exist('getfeatureids')==2 ) 
   di(epochDim).extra = z.trialinfo(getfeatureids(z,'trial')); 
end
[ di(epochDim).extra.marker ] = num2csl(z.y);
[ di(epochDim).info.markerKey ] = unique(z.y);


chDim   =strmatch('electrode',znames);
di(chDim).name = 'ch';     % info about each channel
if( ~isempty(z.trodes) ) 
   Cnames={}; xy=[];
   try;
      fids=getfeatureids(z,'electrode');
      for i=fids; Cnames{end+1,1}=z.trodes(i).electrodes{1};end;
      xy  =position(z.trodes(i)); 
      xy=xy(fids,:);
   catch;
      for i=1:numel(z.trodes); Cnames{end+1,1}=z.trodes(i).electrodes{1};end;
   end
   if ( ~isempty(Cnames) ) di(chDim).vals = Cnames;  end;
   if ( ~isempty(xy) ) [di(chDim).extra.pos2d]=num2csl(xy'); end;
end

% Info about each sample
% find when the transition to 1 happens...
if ( isfield(z.trialinfo(1),'PresentationPhase') )
   timeDim = strmatch('sample',znames);
   zeroSamp = find(z.trialinfo(1).PresentationPhase==1,1,'first');
   zero_ms  = z.dim(timeDim).vals(zeroSamp);
   di(timeDim).vals=di(timeDim).vals-zero_ms;
end

% Info about the units
di(4).units='mV'; % Info about values

% Sampling rate
if ( ~isempty(opts.fs) ) 
   fs=opts.fs; 
elseif ( isfield(z.params,'samplingfreq_hz') )
   fs=z.params.samplingfreq_hz;
else
   fs=256;
end

timeDim   =strmatch('time',znames);
if ( ~isempty(timeDim) ) 
   switch di(timeDim).units; % convert unit names
    case 'msec'; di(timeDim).units='ms';
    case 'sec';  di(timeDim).units='s';
    otherwise; 
   end
end

if ( isfield(z,'y') ) % trial label info into marker set
   [di(n2d(di,'epoch')).extra.marker]=num2csl(z.y);
end


% Other trial info
info.filename=filename;
info.prep=z.prep;
info.params=z.params;

% ensure correct dimension order
X = permute(X,[2:ndims(X), 1]);
di= di([2:end-1 1 end]);
return;
%---------------------------------------------------------------------------
function testCase()


