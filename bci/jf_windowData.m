function [z]=jf_windowData(z,varargin);
% Function to cut data up into equal sized sub-windows & apply taper
%
% Options:
%  dim        -- dim(1)=dimension to slice along, and optionally      ('time')
%                dim(2:end) = use different starting locations for each element in these dims
%  windowType -- type of windowing to use  (1)
%  di         -- dimInfo for the new window dimension
%  nwindows   -- number of windows to use
%  overlap    -- fractional overlap for each window                   (.5)
%  width_ms/width_samp  -- width of the windows in samples or milliseonds
%  start_ms/start_samp  -- [nWin x 1 ] start location for the windows in samples or milliseconds
%                          OR
%                          [nWin x nElm] start location per each element of dim(2:end)
%  zeroPad    -- [bool] zero-pad for windows outside the data? (false)
%
% N.B. Window time is *start* of the sliced window, time-time is then relative to this window start, so vals(win_1)+vals(time) gives orginal epoch time
opts=struct('dim','time','fs',[],...
            'di',[],...
            'windowType',[],...
            'nwindows',[],'overlap',.5,... % spec windows as # and overlap
            'width_ms',[],'width_samp',[],... % win as start+width
            'start_ms',[],'start_samp',[],...
            'step_ms',[],'step_samp',[],...
            'offset_ms',[],'offset_samp',[],... % where the 0 goes?
            'zeroPad',false,...
            'verb',0,'summary',[],'subIdx',[],'windowY',1);
[opts]=parseOpts(opts,varargin);

szX=size(z.X); nd=ndims(z.X);

if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1; % convert neg dim specs

% Get the sample rate, if needed
fs=opts.fs;
if ( isempty(opts.width_samp) || isempty(opts.start_samp) )
   fs = getSampRate(z,'dim',dim(1));
   ms2samp = fs/1000; samp2ms = 1000/fs; 
end

% convert win-specs from time to samples if necessary
if(isempty(opts.width_samp)&&~isempty(opts.width_ms)) 
   opts.width_samp=floor(opts.width_ms*ms2samp);
end;
if(isempty(opts.start_samp)) % find closest sample number to the given start times
   switch ( z.di(dim(1)).units )
    case {'ms','msec'}; 
     for i=1:numel(opts.start_ms); 
        [ans,opts.start_samp(i)]=min(abs(opts.start_ms(i)-z.di(dim(1)).vals));
		  if ( isnan(opts.start_ms(i)) ) opts.start_samp(i)=NaN; end
     end
    case {'s','sec'};
     for i=1:numel(opts.start_ms); 
        [ans,opts.start_samp(i)]=min(abs(opts.start_ms(i)/1000-z.di(dim(1)).vals));
		  if ( isnan(opts.start_ms(i)) ) opts.start_samp(i)=NaN; end
     end
    otherwise;
     opts.start_samp=floor(opts.start_ms*ms2samp);
   end
   opts.start_samp=reshape(opts.start_samp,size(opts.start_ms));
   
end;
if ( isempty(opts.step_samp) && ~isempty(opts.step_ms) ) 
  opts.step_samp=floor(opts.step_ms*ms2samp); 
end;
[start width]=compWinLoc(szX(dim),opts.nwindows,opts.overlap,opts.start_samp,opts.width_samp,opts.step_samp);
% ensure start is column vector
if ( numel(dim)==1 && size(start,1)==1 ) start=start'; end;

% identify which window positions to keep
retainStart = all(start(:,:)>0,2) & all(start(:,:)+width(:,:)-1<=size(z.X,dim(1)),2);
if ( ~opts.zeroPad && ~all(retainStart(:)) ) 
   warning('Windows outside the size of X removed');
   start       = start(retainStart,:);
   if ( ~isempty(opts.di) && isstruct(opts.di) )  % update the di too
      opts.di(1).vals=opts.di(1).vals(retainStart); 
      if(numel(opts.di(1).extra)==numel(retainStart))
         opts.di(1).extra(~retainStart)=[]; % works even if extra is empty-struct array
      end;
   end;
end
opts.nwindows=size(start,1);opts.start_samp=start;opts.width_samp=width;

% Call sigproc functions to do the actual work
if ( ~isempty(start) ) 
   z.X   = windowData(z.X,start,width,dim);
else
   z.X   = reshape(z.X,[szX(1:dim(1)) 1 szX(dim(1)+1:end)]); %insert pseudo window dim
end
% apply the temporal window function
winFn=[];
if ( ~isempty(opts.windowType ) ) 
   winFn = mkFilter(size(z.X,dim(1)),opts.windowType);
   if( ~all(winFn(:)==1) ) z.X   = repop(z.X,'*',shiftdim(winFn,-dim(1)+1)); end;
end

% Update the dim info
z.di= z.di([1:dim(1) dim(1):end]); % repeat time axes
odi = z.di(dim(1));
% gen meta-info for the new dim
if ( isempty(opts.di) )    ndi=mkDimInfo(size(z.X,dim(1)+1),1,'window'); ndi.vals=[]; 
elseif ( ischar(opts.di) ) ndi=mkDimInfo(size(z.X,dim(1)+1),1,opts.di);  ndi.vals=[]; % dim name
else ndi=opts.di; % dim info
end
if ( isempty(ndi(1).units) ) ndi(1).units=z.di(dim(1)).units; end;
if ( isempty(ndi(1).vals) )  ndi(1).vals =z.di(dim(1)).vals(round(mean(start(:,:)+round(width/2),2))); end;
z.di(dim(1)+1)=ndi(1);
if( ~isempty(z.di(dim(1)).extra) && numel(z.di(dim(1)).extra)>=numel(z.di(dim(1)).vals) )
    z.di(dim(1)).extra = z.di(dim(1)).extra(1:min(end,width)); 
end
% set new window dim's values
offset_samp=opts.offset_samp; 
if( isempty(offset_samp) && ~isempty(fs) && ~isempty(opts.offset_ms) ) 
  offset_samp=opts.offset_ms*ms2samp; 
end;
if( ~isempty(offset_samp) && isnumeric(z.di(dim(1)).vals) )
  [ans,oidx]=min(abs(z.di(dim(1)).vals));  % find 0 of current values
  z.di(dim(1)).vals = z.di(dim(1)).vals(oidx+offset_samp+(0:width-1));
else
  [ans,oidx]=min(abs(z.di(dim(1)).vals));  % find 0 of current values
  z.di(dim(1)).vals = z.di(dim(1)).vals(oidx:min(end,oidx+width-1)); 
end

if ( isfield(z,'Y') && isfield(z,'Ydi') && n2d(z.Ydi,z.di(dim(1)).name,0,0) ) % need to WINDOW Y also
   ydim=n2d(z.Ydi,{z.di(dim(1)).name});
   if( opts.windowY ) 
      z.Y=windowData(z.Y,start,width,ydim);
      z.Ydi = z.Ydi([1:ydim ydim:end]);
      z.Ydi(ydim).vals = z.di(dim(1)).vals;
      z.Ydi(ydim+1)    = z.di(dim(1)+1);
   else
      z=subsampleY(z,start,ydim); %subsample to center of the window
      z.Ydi(ydim).name =z.di(dim(1)+1).name; % update the name to the window-dimension
   end
end

summary = sprintf('%d %ss of',opts.nwindows,z.di(dim(1)+1).name);
if ( ~isempty(fs) )  summary=sprintf('%s %g ms',summary,width*samp2ms);
else                 summary=sprintf('%s %g samp',summary,width);
end
if (~isempty(opts.windowType) ) 
   if ( ischar(opts.windowType) ) summary=sprintf('%s %s win',summary,opts.windowType); 
   else summary=[summary ' win'];
   end
end
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
info=struct('winFn',winFn,'start_samp',start,'width_samp',width,'odi',odi);
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%----------------------------------------------------------------------
function testCase()
w=jf_windowData(z,'width_ms',500)
