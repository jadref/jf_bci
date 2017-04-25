function [c,conf]=est2loss(Y,Yest,losstype)
% classificiation performance between Y and it's estimate
%
%  [c,conf]=est2loss(Y,Yest)
%
% Input
%  Y    - [nCls x N] target to fit with N examples
%  Yest - [nCls x N] estimate of the target with N examples
% Output
%  c    - [float] correlation between Y and Yest
%  conf - [nCls x nCls] confusion matrix
if ( nargin< 3 ) losstype=''; end;
if ( ndims(Y)==2 && size(Y,2)==1 && size(Y,1)>1 ) Y=Y'; end; %[1 x N]
Y     = reshape(Y,[],size(Y,ndims(Y)));
Yest  = reshape(Yest,[],size(Yest,ndims(Yest)));
if ( size(Y,1)==1 && size(Yest,1)==2 ) Y=cat(1,Y,-Y); end;
exInd = any(isnan(Y),1) | all(Y==0,1);% excluded points
Y     = Y(:,~exInd);
Yest  = Yest(:,~exInd);
pred  = dv2pred(Yest,1);
conf  = pred2conf(Y,pred,[2 1]);
c     = conf2loss(conf,2,losstype);
return;
%-----------------------------------------------------------------------------
function testCase()
Yl  = ceil(rand(1,1000)*3);
Y   = lab2ind(Yl(:))';
Yest= Y+randn(size(Y))*.3;
est2loss(Y,Yest)

