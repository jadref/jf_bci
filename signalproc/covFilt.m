c function [covXeXe,muXe,Ne]=covFilt(X);
% Comp the covariance structure for each of the time-series in the input
% matrix X.
% Inputs:
% X - data matrix [samples x dim x examples] 
% Outputs
% covXeXe -- matrix of example covariance matrices [dim x dim x examples]
% muXe    -- matrix of example means               [dim x examples]
% Ne      -- number of samples in each example     [1   x examples]
[samp,dim,tr]=size(X);
covXeXe=zeros(dim,dim,tr);
muXe   =zeros(dim,tr);
Ne     =zeros(1,tr);
for i=1:tr;
   Ne(i)=samp;
   muXe(:,i)=mean(X(:,:,i));
   covXeXe(:,:,i)=X(:,:,i)'*X(:,:,i);
end
return;
