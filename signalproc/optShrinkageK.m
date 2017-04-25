function [lambda,C]=optShrinkage(K,d,centerp)
% kernelised optimal shrinkage estimation
%
%  [lambda]=optShrinkage(K,d,centerp)
%
% Inputs:
%  K - [N x N] kernel matrix
%  d - [int] input dimensionality
%  centerp - [bool] do we center the data (no effect)
N     =size(K,1);
X2    =diag(K);
mu    =sum(X2)./N/d; %mu=<I,Sigma>, proj to sphere
nSigma=K(:)'*K(:)./N./N; % variance of the sigma entries
alpha2=nSigma-mu*mu*d; %alpha^2=|muI-Sigma|_F^2, error to sphere
beta2 =(sum(X2.^2)/N-nSigma)/N;
lambda=beta2./(beta2+alpha2);
C     =beta2/alpha2*mu; % lambda./(1-lambda)
return;