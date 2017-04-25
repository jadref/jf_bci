function [mu,X]=robustCenter(X,dim,maxIter,tol,verb)
% robustly center the data X by making have (approx) median=0
if ( nargin < 2 || isempty(dim) ) dim=find(size(X)>1,1); end; % 1st non-sing dim
if ( nargin < 3 || isempty(maxIter) ) maxIter=[]; end;
if ( nargin < 4 || isempty(tol) ) tol=1e-6; end;
if ( nargin < 5 || isempty(verb) ) verb=1; end;
if ( numel(dim)==1 )
  if ( isempty(maxIter) )% 1-d so can just sub the median directly
      mu=median(X,dim); 
  else
      [badInd,feat,threshs,stds,mus]=idOutliers(X,dim,maxIter);
      mu=tprod(X,[1:dim-1 -dim dim+1:ndims(X)],single(~badInd(:))./sum(~badInd(:)),-dim);
  end
  if (nargout>1 ) 
     X=repop(X,'-',mu); 
  end;
else % multi-dim version, use iterative approx to the median
  % compute the thing to subtract (multi-dim aware)
  mu=X;N=1; for di=1:numel(dim); N=N*size(mu,dim(di)); mu=sum(mu,dim(di)); end; mu=mu./N;
  X = repop(X,'-',mu); % 1st normal center
  if ( verb>0 ) fprintf('robustCenter:'); end;
  if(isempty(maxIter) ) maxIter=10; end;
  for iter=1:maxIter; % make outlier robust, i.e. compute median
    % multi-dim aware new-mean computation
    clear mu1 mu2;
    mu1=single(sign(X)); % single to save ram
    for di=1:numel(dim); mu1=sum(mu1,dim(di)); end; % sum immediately to save RAM
    mu2=abs(X); mu2(mu2<eps)=1; mu2=1./mu2; %  prevent div by 0
    for di=1:numel(dim); mu2=sum(mu2,dim(di)); end; % sum immediately to save RAM
    mu1=mu1./mu2; clear mu2; % save some ram
    % center
    X  = repop(X,'-',mu1);
    mu = mu+mu1; % accum total change in X
    %fprintf('%g,',max(abs(mu1(:)))./max(abs(mu(:))))
    if ( max(abs(mu1(:)))./max(abs(mu(:))) < tol ) break; end; % convergence test
    %mean(mu1(:))
    if ( verb>0 && maxIter>1 ) textprogressbar(iter,maxIter);  end
  end
  if( verb>0) fprintf('\n'); end;
end
return;
%-----------------------------------------
function testCases()
X=randn(100,1); X([25 50 75])=100;
plot([X robustCenter(X)])

X=randn(100,100); X([25 50 75],:)=100;
[X2,mu]=robustCenter(X,[],10);
mimage(X,X2,'clim',[-2 2],'diff',1)
