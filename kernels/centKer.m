function K=centKer(K,mu)
% Center the matrix K
% 
% function K=centKer(K)
%INPUTS
% K  -- [N x N] kernel matrix to be centered
% mu -- [N x 1] point to center about (mean if not given)
% 
%OUTPUTS
% K  -- [N x N] centered kernel matrix

N = size(K,1);
if ( nargin < 2 ) mu=ones(N,1)./N; end;
Kmu = K' *mu;
mu2 = Kmu'*mu;
% mem efficient centering
K = repop(repop(K,'-',Kmu'),'-',Kmu-mu2); % J=ones(N,1)*D; K=K-J-J'+E;
%-------------------
function testCase
X=randn(100,99,98);
Xc=repop(X,'-',mmean(X,3));  
k=tprod(X,[-1 -2 1],[],[-1 -2 2]);
kc=centKer(k);
Xc=repop(X,'-',mean(X,2));
kxc=tprod(X,[-1 -2 1],[],[-1 -2 2]);
mad(kc,kxc)
clf;plot([diag(kc) diag(kxc)])