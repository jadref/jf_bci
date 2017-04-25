function [alphab,f,J]=kfda(K,Y,C,varargin)
% Kernel (Fisher) discriminant analysis (LDA/FDA) -- wrapping nlda
%
%function [wb,f,J]=lda2(X,Y,C)
%
%INPUTS
% K     - [N x N] data matrix
% Y     - [N x 1] set of class labels
% C     - [1 x 1] regularization parameter
% 
%OUTPUTS
% alphab- [N+1 x L] set of fisher-directions/classifier weight vectors 
% f     - [N x 1] The decision value for all the inputs
% J     - the final objective value
if ( nargin < 3 ) C(1)=0; end;
% compute the kernel to use in the nlda call
if ( isa(Y,'single') ) Y=double(Y); end;
if ( isa(K,'single') ) K=double(K); end;

% remove the excluded points from the training set
incIdx=Y~=0; 
if( ~all(incIdx) ) Ytrn=Y(incIdx); Ktrn=K(incIdx,incIdx); else Ytrn=Y; Ktrn=K; end;
% call nlda to do the training
[alphai,bi,xi,err,fi]=nlda(Ytrn,Ktrn,C,'kfd');
% check for sign inversion
dir=fi'*Ytrn; if( dir<0 ) alphai=-alphai; end; 
% optimise b
b = optbias(Ytrn,fi);
% extract the solution information
alphab([incIdx;true],1)=[alphai;b+bi]; 
f =K*alphab(1:end-1)+alphab(end);
J =0;
return;
%--------------------------------------------------------------------------------------------
function testCases()
%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

[Wb,f,J]=lda2(X,Y,0);
plotLinDecisFn(X,Y,Wb(1:end-1,:),Wb(end,:));

% test implicit ignored
[alphab0,f0,J0]=klr_cg(K,Y.*single(fInds(:,end)<0),1,'verb',1,'ridge',1e-7);

% for linear kernel
alpha=zeros(N,1);alpha(find(trnInd))=alphab(1:end-1); % equiv alpha
plotLinDecisFn(X,Y,X(:,trnInd)*alphab(1:end-1),alphab(end),alpha);

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
