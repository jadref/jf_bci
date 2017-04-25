function [T,Z]=KLFDA(K,Y,r,metric,kNN,reg)
%
% Kernel Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction
%
% Usage:
%       [T,Z]=KLFDA(K,Y,r,metric,kNN,reg)
%
% Input:
%    K:      n x n kernel matrix
%            n --- the number of samples 
%    Y:      n dimensional vector of class labels
%    r:      dimensionality of reduced space (default: d)
%    metric: type of metric in the embedding space (default: 'weighted')
%            'weighted'        --- weighted eigenvectors 
%            'orthonormalized' --- orthonormalized
%            'plain'           --- raw eigenvectors
%    kNN:    parameter used in local scaling method (default: 7)
%    reg:    regularization parameter (default: 0.001)
%
% Output:
%    T: d x r transformation matrix (Z=T'*X)
%    Z: r x n matrix of dimensionality reduced samples 
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/LFDA/

if nargin<2
  error('Not enough input arguments.')
end

if nargin<3
  r=1;
end

if nargin<4
  metric='weighted';
end

if nargin<5
  kNN=7;
end

if nargin<6
  reg=0.0001;
end

[n ndum]=size(K);
opts.disp = 0; 
tSb=zeros(n,n);
tSw=zeros(n,n);

for c=unique(Y')
  Kcc=K(Y==c,Y==c);
  Kc=K(:,Y==c);
  nc=size(Kcc,1);

  % Define classwise affinity matrix
  Kccdiag=diag(Kcc);
  distance2=repmat(Kccdiag,1,nc)+repmat(Kccdiag',nc,1)-2*Kcc;
  [sorted,index]=sort(distance2);
  kNNdist2=sorted(kNN+1,:);
  sigma=sqrt(kNNdist2);
  localscale=sigma'*sigma;
  flag=(localscale~=0);
  A=zeros(nc,nc);
  A(flag)=exp(-distance2(flag)./localscale(flag));

  Kc1=sum(Kc,2);
  Z=Kc*(repmat(sum(A,2),[1 n]).*Kc')-Kc*A*Kc';
  tSb=tSb+Z/n+Kc*Kc'*(1-nc/n)+Kc1*Kc1'/n;
  tSw=tSw+Z/nc;
end

K1=sum(K,2);
tSb=tSb-K1*K1'/n-tSw;

tSb=(tSb+tSb')/2;
tSw=(tSw+tSw')/2;

%[eigvec,eigval_matrix]=eig(tSb,tSw+reg*eye(size(tSw)));
opts.disp = 0; 
[eigvec,eigval_matrix]=eigs(tSb,tSw+reg*eye(size(tSw)),r,'la',opts);

eigval=diag(eigval_matrix);
[sort_eigval,sort_eigval_index]=sort(eigval);
T0=eigvec(:,sort_eigval_index(end:-1:1));

switch metric %determine the metric in the embedding space
  case 'weighted'
   T=T0.*repmat(sqrt(sort_eigval(end:-1:1))',[n,1]);
  case 'orthonormalized'
   [T,dummy]=qr(T0,0);
  case 'plain'
   T=T0;
end

Z=T'*K;

