function [z]=jf_permute(z,varargin);
% permute order of dimensions
% Options:
%  dim -- new dimension order, first numel(dim) new dim if not all specified
opts=struct('dim',[],'subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

dim=n2d(z,opts.dim); 
if( any(dim==0) ) 
   error('Must specify new place for all dims');
 end
if ( numel(dim)<ndims(z.X) ) % move the specified dims to the front
  ndim=[dim(:)' setdiff(1:ndims(z.X),dim)];
else
  ndim=dim;
end

% do the work
z.X = permute(z.X,ndim);
z.di= [z.di(ndim); z.di(end)];

% re-order Y too if poss
if ( isfield(z,'Ydi') && ~isempty(z.Ydi) ) 
   Ydim=n2d(z,{z.Ydi(1:end-1).name},0,0); 
   [ans,nYdim]=sort(Ydim(Ydim>0),'ascend');
   perm =[nYdim; find(Ydim==0); numel(z.Ydi)];
   z.Y  =permute(z.Y,perm);
   z.Ydi=z.Ydi(perm);
end

summary = '';
if ( numel(dim)<ndims(z.X) )
  summary=sprintf('%s to front',sprintf('%s,',z.di(1:numel(dim)).name));
end
info    = [];
z = jf_addprep(z,mfilename,summary,opts,info);
return
%-------------------------------------------------------------------------
function testCase()
