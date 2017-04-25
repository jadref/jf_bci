function [Wb,f,J]=lda2(X,Y,C,varargin)
% Linear (Fisher) discriminant analysis (LDA/FDA) -- wrapping nlda
%
%function [wb,f,J]=lda2(X,Y,C)
%
%INPUTS
% X     - [n-d x N] data matrix
% Y     - [N x 1] set of class labels
%          OR
%         [N x L] 1vsRest set of class indicators in -1/0/+1 format
% C     - [1 x 1] regularization parameter
% 
%Options:
% dim   - [int] dimension of X which contains the trials (ndims(X))
%
%OUTPUTS
% wb    - [N+1 x L] set of fisher-directions/classifier weight vectors 
% f     - [N x 1] The decision value for all the inputs
% J     - the final objective value

if ( nargin < 3 ) C(1)=0; end;
opts=struct('dim',[]);
[opts,varargin]=parseOpts(opts,varargin{:});
dim=opts.dim; if ( isempty(opts.dim) ) dim=ndims(X); end;

if( size(Y,2)==1 ) [Y,key]=lab2ind(Y); end;
nSp = size(Y,2);
szX=size(X);

% compute the kernel to use in the nlda call
if ( isa(Y,'single') ) Y=double(Y); end;
if ( isa(X,'single') ) X=double(X); end;
K = tprod(X,[-(1:dim-1) 1 -(dim+1:ndims(X))],[],[-(1:dim-1) 2 -(dim+1:ndims(X))],'n');

for spi=1:nSp;
   % remove the excluded points from the training set
   incIdx=Y(:,spi)~=0; 
   if( ~all(incIdx) ) Ytrn=Y(incIdx,spi); Ktrn=K(incIdx,incIdx); else Ytrn=Y(:,spi); Ktrn=K; end;
   % call nlda to do the training
   [alphai,bi,xi,err,fi]=nlda(Ytrn,Ktrn,C,'kfd');
   % check for sign inversion
   dir=fi'*Ytrn; if( dir<0 ) alphai=-alphai; end; 
   % optimise b
   bi = optBias(Ytrn,fi);
   % extract the solution information
   alphab([incIdx;true],spi)=[alphai;bi]; f(incIdx,spi)=fi; % taking account of excluded points
   W=tprod(X,[1:dim-1 -dim dim+1:ndims(X)],alphab(1:end-1,spi),-dim);
   Wb(:,spi)=[W(:);bi];% linear weighting
end

% compute the predictions for all classes/directions
f = tprod(X,[-(1:dim(1)-1) 1 -(dim(1)+1:ndims(X))],...
          reshape(Wb(1:end-1,:),[szX(1:dim(1)-1),1,szX(dim(1)+1:ndims(X)),size(Wb,2)]),...
          [-(1:dim(1)-1) 0 -(dim(1)+1:ndims(X)) 2],'n');
f = repop(f,'+',Wb(end,:));
J = 0;
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
