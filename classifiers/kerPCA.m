function [alpha,L,K2] = kerPCA(K,nDim)
% Kernel PCA decomposition
%
%function [alpha] = kerPCA(K,nDim)
%
%INPUTS
% K      - [N x N] kernel matrix
% nDim   - [1 x 1] number of components to take
%
%OUTPUTs
% alpha  - [N x nDim] the dual coefficients corresponding to the PCA feature
%          directions,  A mapped feature vector is given by:
%             \sum_i K(x_i,x)*alpha_i = K(x)'*alpha
% L      - [nDim x 1] eigenvalues of the PCA coefficients
% Kpca   - [N x N] kernel for the new set of features
%
% Jason Farquhar 08/02/05
% The basic algorithm is taken from:
%   "Kernel Methods for Pattern Analysis" JS-T and N. Cristianini (P187)
%   based on the dualpls.m code from: www.kernel-methods.net
% 
% Example output usage:
%  X = K(X)*alpha' ;
%

N=size(K,1);
D=sum(K,1)/N;
E=sum(D,2)/N;
% Centering
K = repop(repop(K,'-',D'),'-',D+E); % J=ones(N,1)*D; K=K-J-J'+E;
% kPCA
[V,D]=eigs(K,nDim);%eigs(K,nDim,'LM');
alpha=(diag(1./sqrt(diag(D)))*V')';

% New kernel, if wanted
if ( nargout > 2 )   K2 = V * D * V'; end
return;
%---------------------------------------------------------------------------
function []=testCase()