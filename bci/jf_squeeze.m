function [z]=jf_squeeze(z,varargin);
% remove singlenton dimensions from z
%
% Options:
%  dim - the dimension(s) to squeeze out ([])
%        if empty, the squeeze *all* size==1 dims
opts=struct('dim',[],'subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

szX=size(z.X); szX(end+1:numel(z.di)-1)=1; 
dim=n2d(z,opts.dim,[],0);
if ( isempty(dim) ) % all non-singlentons...
  dim=find(szX==1);
else
  if ( any(dim==0) ) 
    warning('Some dims didnt match anything, ignored!'); 
    dim(dim==0)=[];
  end
end
if ( ~all(szX(dim)==1) ) 
  error('cant squeeze non-singlenton dims');
end

z.X=reshape(z.X,szX(setdiff(1:end,dim)));
odi = z.di(dim);
z.di(dim)=[];
info=[];
summary=sprintf('%s,',odi.name);
z=jf_addprep(z,mfilename,summary,opts,info);
return;
