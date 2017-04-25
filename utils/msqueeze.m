function mx=msqueeze(dim,mx);
% mx=msqueeze(dim,mx)
szmx=size(mx);
dim(dim>numel(szmx))=[];
if(~isempty(dim) && all(szmx(dim)==1)) 
  tmp=true(size(szmx));tmp(dim)=false;
  mx=reshape(mx,[szmx(tmp) 1]);
end
return;
