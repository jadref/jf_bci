function [mu]=clsAve(X,Y,dim)
if ( nargin<3 || isempty(dim) ) dim=ndims(X); end;
if ( all(Y(:)==0 | Y(:)==-1 | Y(:)==1) && size(Y,2)>1 ) 
  key=1:size(Y,2);
else
  key=unique(Y);
end
szX=size(X);
szX(dim)=numel(key);
mu=zeros([szX(1:dim-1) numel(key) szX(dim+1:end)]);
idx={};for di=1:numel(szX); idx{di}=1:szX(di); end;
for ci=1:numel(key);
  if( size(Y,2)==1 ) 
    mu(idx{1:dim-1},ci,idx{dim+1:end})=mean(X(idx{1:dim-1},Y==key(ci),idx{dim+1:end}),dim);
  else
    mu(idx{1:dim-1},ci,idx{dim+1:end})=mean(X(idx{1:dim-1},Y(:,ci)>0,idx{dim+1:end}),dim);
  end
end
