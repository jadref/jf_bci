function [z]=featstats(z,varargin)
% multi-dimensional variance computation
%
% [s]=featstats(z);
% Options:
%  dim -- dimension of X to sum along to compute the variance
%  wght-- [size(X,dim) x L] set of weightings for the points in dim
opts=struct('dim','time','wght',[],'di',[],'subIdx',[]);
opts=parseOpts(opts,varargin);

sz=size(z.X); nd=numel(sz);
if ( iscell(opts.dim) || ischar(opts.dim) ) % convert names to dims
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim);  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+nd+1; % convert neg dim specs
if ( isempty(dim) && ~isempty(opts.di) && numel(opts.di)>1 )    
   dim = strmatch(opts.di(1).name,{z.di.name}); % extract dim from the di...
end

% call out to do the actual computation
z.X = featstats(z.X,dim,opts.wght);

% Update the objects dim-info
odi=z.di; z.di=z.di([1:end-1 end-1 end]);
if ( ~isempty(opts.di) ) 
   if ( numel(opts.di)~=3 && numel(opts.di)~=1 ) 
      error('di size is unreasonable, should be 1 or 3');
   end
   z.di(dim) = opts.di(min(2,end)); % use the info in the given dimInfo
else
   z.di(dim) = mkDimInfo(size(z.X,dim),1,[z.di(dim).name '_stats'],[],[]);
end
z.di(end-1) = mkDimInfo(2,1,'stat',[],{'mu','var'});

summary=sprintf('over %s ',odi(dim).name);
if ( ~isempty(opts.wght) )
   summary=sprintf('%s %d->%d %ss ',summary,numel(odi(dim).vals),size(opts.wght,2),z.di(dim).name);
end
info=struct('odi',odi(dim));
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%--------------------------------------------------------------------------
