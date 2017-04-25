function [z]=jf_subsample(z,varargin);
% sub-sample the input using a bin-averaging technique
%
% Options:
%  dim        -- the time dimension
%  fs         -- the new sampling rate
%  method     -- [str] sub-sampling method {'ave','butter'} ('ave')
opts=struct('dim','time','fs',[],'method','ave','subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1; % convert neg dim specs

% extract the sampling rate
if ( isfield(z,'fs') ) fs=z.fs;
elseif ( isfield(z.di(dim).info,'fs') ) fs=z.di(dim).info.fs;
else
   warning('Couldnt find sample rate in z, estimating from vals');   
   fs = 1000./mean(diff(z.di(dim).vals));
end
ms2samp = fs/1000; samp2ms = 1000/fs; 

switch lower(opts.method);
 case 'ave';    [z.X idx]=subsample(z.X,size(z.X,dim)*opts.fs./fs,dim);
 case 'butter'; [z.X idx]=subsample_butter(z.X,size(z.X,dim)*opts.fs./fs,dim);
 otherwise; error('Unrecognised sub-sampling method: %s\n',opts.method);
end

% update the dim-info
odi=z.di;
z.di(dim).vals =z.di(dim).vals(ceil(idx));
if( numel(z.di(dim).extra)>0 ) z.di(dim).extra=z.di(dim).extra(ceil(idx)); end;
z.di(dim).info.fs = opts.fs;
if ( isfield(z,'fs') ) z=rmfield(z,'fs'); end;

if ( isfield(z,'Y') && isfield(z,'Ydi') && n2d(z.Ydi,z.di(dim).name,0,0) ) % need to sub-sample Y also
   z=subsampleY(z,round(idx),z.di(dim).name);
end

% recording the updating info
summary=sprintf('%s, %gHz -> %gHz',opts.method,fs,opts.fs);
info=struct('idx',idx,'ofs',fs);
z =jf_addprep(z,mfilename,summary,opts,info);
return
%----------------------------------------------------------------------------
function testCase()
