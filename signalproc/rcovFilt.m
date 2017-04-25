function [covXeXe,wght,muXe,Ne]=rcovFilt(X,rType,thresh,wght);
% robust covariance estimation for each of the time-series in the input
% matrix X.
% Inputs:
% X      -- data matrix [samples x dim x examples] 
% rType  -- type of robust estimator to use (see rCov)
% thresh -- threshold for the estimator
% wght   -- an initial sample weighting [samples x examples]
% Outputs
% covXeXe -- matrix of example covariance matrices [dim x dim x examples]
% wght    -- final sample weightings               [samples x examples]
% muXe    -- matrix of example means               [dim x examples]
% Ne      -- number of samples in each example     [1   x examples]
%
% $Id: rcovFilt.m,v 1.7 2007-05-13 22:12:39 jdrf Exp $
[dim,samp,nEx]=size(X);
if ( nargin < 2 ) rType='none'; end;
if ( nargin < 3 ) thresh=inf; end;
if ( nargin < 4 | isempty(wght) ) wght=ones(samp,nEx); end;
covXeXe=zeros(dim,dim,nEx);
muXe   =zeros(dim,nEx);
Ne     =zeros(1,nEx);
for ex=1:nEx;
   Ne(ex)=samp;
   muXe(:,ex)=mean(X(:,:,ex),2);
   [covXeXe(:,:,ex) wght(:,ex)]=rCov(X(:,:,ex),[],rType,thresh,wght(:,ex));
end
return;
