function [U,S]=olEig(X,U,S,lambda,tol,maxIter)
% on-line eigen-decomposition of the data covariance matrices using the power method
% 
% [U,S]=olCovEig(X,U,S,[lambda,tol,maxIter,verb])
%
% Time complexity ~ O(rd^2). N.B. if r<<d can be reduced to O(dr^2)
%  N.B. this is *the same* complexity as just computing the eig(Cov) directly!
%
% Inputs:
%  X - [d x N] N examples of d-dimensional data vector
%  U - [d x r] r eigenvectors of the decompostion of cov(X) = X*X';
%  S - [r x 1] r eigenvalues of cov(X) corrospending to U
%  lambda - [1x1] exponiential forgetting factor for the coveriance,  (1)
%              cov(X) = \sum_{0:T} X(:,t)*X(:,t)' * lambda.^(T-t)
%  tol - [3x1] convergence tolerances.                                (1e-2)
%          [relative change in U, absolute change in U, magnitude of S]
%  maxIter - [1x1] max number of iterations per component             (3*d)
%  verb - [1x1] verbosity level.                                      (0)
% 
if ( nargin<4 || isempty(lambda) ) lambda=0; end;
if ( nargin<5 || isempty(tol) ) tol=1e-2; end;
if ( nargin<6 || isempty(maxIter) ) maxIter=size(X,1)*3; end;
if ( nargin<7 || isempty(verb) ) verb=0; end;
tol(end+1:3)=tol(end);
if ( nargin<3 || (nargin>2 && isempty(S)) )
  if ( numel(U)==1 ) S=ones(U,1); U=[]; 
  elseif ( numel(U)==max(size(U)) ) S=U(:); U=[];
  else error('Cant extract seed info');
  end;
end;
if ( isempty(U) ) U=eye(size(X,1),size(S,1)); end
rank=numel(S);
if ( lambda>0 ) S=S*lambda; end;
C=repop(U,'*',S')*U';

% loop over ranks
%IsUUti=eye(size(X,1));
UUti=zeros(size(X,1),size(X,1));
for ri=1:rank;
  u=U(:,ri); % start with seed solution
  for iter=1:maxIter;
    ou=u;
    u = X*(u'*X)' + C*u; 
    % deflate the other components, i.e. ortho w.r.t. bits already found    
    if ( ri>1 ) u=u-UUti*u; end; %u=u-Ui*(u'*Ui)'; end %   alt: u=IsUUti*u;
    s=sqrt(u'*u); 
    if ( s<eps ) break; end; % prevent divide by 0, stop when no ranks left
    u=u./s;
    % convergence test
    du = sum(abs(ou-u)); % change in vector
    if ( verb>0 ) fprintf('%d.%d) nu=%g du=%g\n',ri,iter,nu,du); end;
    if ( iter<3 ) du0=du; elseif ( du<tol(1)*du0 ) break; end;
    if ( du<tol(2) || s<tol(3) ) break; end;
  end
  U(:,ri)=u;
  S(ri)=s;
  % for the deflation
  UUti = UUti + u*u'; % for the deflation
  %Ui   = U(:,1:ri); 
  %IsUUti=IsUUti-u*u';
end
return;
% 
function testCase()
% test it's covariance tracking abilities
X=cumsum(randn(10,1000));
X(:,end/2:end)=X(:,end/2:end)*5;
[U,S]=olEig(X(:,1:10),3,[],[],[],20);
C=X(:,1:9)*X(:,1:9)';
[U,S]=olEig(X(:,10)  ,3,[],C, [],20);

% test tracking ability - using a cached data covariance
[U,S]=olEig(X(:,1:10),3,[],[],[],20);
C=X(:,1:10)*X(:,1:10)';
tic,
clear Si
for i=11:size(X,2);
  [U,S]=olEig(X(:,i),U,S,C);
  Si(:,i)=S;
  C=C+X(:,i)*X(:,i)';
end
toc
clf;plot(Si')


% test tracking ability - using previous decomp approx
[U,S]=olEig(X(:,1:10),3,[],[],[],20);
tic,
clear Si
for i=11:size(X,2);
  [U,S]=olEig(X(:,i),U,S);
  Si(:,i)=S;
end
toc
clf;plot(Si')

% test tracking ability - using previous decomp approx, with forgetting factor
[U,S]=olEig(X(:,1:10),3,[],[],[],inf,.99);
tic,
clear Si
for i=11:size(X,2);
  [U,S]=olEig(X(:,i),U,S,[],[],[],.99);
  Si(:,i)=S;
end
toc
clf;plot(Si')


% rank defficient test....
X=randn(9,1000);
X(10,:)=sum(X);
[U,S]=olEig(X(:,1:10),10,[],[],[],20);
tic,
clear Si
for i=11:size(X,2);
  [U,S]=olEig(X(:,i),U,S);
  Si(:,i)=S;
end
toc
clf;plot(Si')
