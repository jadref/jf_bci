function [Wb,f,J]=lda(X,Y,C,varargin)
% Linear (Fisher) discriminant analysis (LDA/FDA).
%
% N.B. We solve by mapping it to a rLSC problem to avoid singularities in the LDA formulation
% See:           An Efficient Algorithm for LDA Utilizing the Relationship between LDA
%  and the generalized Minimum Squared Error Solution, Cheong Hee Park and Haesun Park, 
%  Pattern Recognition
% 
%function [wb,f,J]=lda(K,Y,C)
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
opts=struct('dim',[],'whiten',1);
[opts,varargin]=parseOpts(opts,varargin{:});
dim=opts.dim; if ( isempty(opts.dim) ) dim=ndims(X); end;

if( size(Y,2)==1 ) [Y,keyY]=lab2ind(Y,[],[],[],1); if(size(Y,2)==1) Y=[Y -Y]; keyY=[1 -1]; end; end;
L = size(Y,2);

szX=size(X);
% compute the per-class covariance matrices, means
szSx=szX; szSx(dim(1))=1;
Cxx_L= zeros([prod(szSx) prod(szSx) L],class(X)); % sum x*x'
Mux_L = zeros([prod(szSx) L],class(X));            % sum x
idx1=1:ndims(X); idx1(dim)=-dim; idx2=ndims(X)+(1:ndims(X)); idx2(dim)=-dim;
idx={}; for d=1:ndims(X); idx{d}=1:size(X,d); end;
for l=1:L;
   idx{dim(1)}  = Y(:,l)>0;
   N_L(l,1)     = sum(Y(:,l)>0);
   X_L          = X(idx{:}); % extract the entries for this class
   Cxx_L(:,:,l) = reshape(tprod(X_L,idx1,[],idx2,'n'),prod(szSx),prod(szSx))./N_L(l); % covariance
   Mux_L(:,l)   = reshape(sum(X_L,dim(1)),[prod(szSx) 1])./N_L(l);            % class mean
end
N      = sum(N_L,1);
MuxMux_L = tprod(Mux_L,[1 3],[],[2 3],'n'); % covar of the class means

% compute the within and between class covariance matrices
Sigma_w  = sum(Cxx_L-MuxMux_L,3); % mean class cov
if( C>0 ) Sigma_w(1:size(Sigma_w,1)+1:end) = Sigma_w(1:size(Sigma_w,1)+1:end)+C(1); end; % regularised
mu       = (Mux_L*N_L)./N;
Sigma_b  = sum(MuxMux_L,3) - L*mu*mu'; % class center cov

% compute the projection direction
% 1 - map to whitened space where total co-variance is *full-rank* and identity
if ( opts.whiten )
   [U,S,V]=svd(Sigma_w,0);S=diag(S); % U (and V) contain the whitening transforms
   USigma_bU = U'*Sigma_b*U;
else
   USigma_bU = Sigma_b;
end
% 2 - compute the solution
if ( size(Sigma_w,1)<600 ) % use full eig if not high dim (as its faster)
   [W,D] = eig(USigma_bU,Sigma_w); W=W(:,1:L-1); D=diag(D); D=D(1:L-1);
else
   [W,D] = eigs(double(USigma_bU),Sigma_w,L-1,'la',struct('disp',0)); D=diag(D);
end
% map back to full space
if ( opts.whiten ) W = U*W; end;
% compute the bias term
b = -W'*mu; % bias puts the data mean at 0

% for classification calls use the class means to set the directions correctly
if ( L==2 ) dir = W'*(Mux_L(:,1)-Mux_L(:,2));if ( dir<0 ) W=-W; b=-b; end; end;

% compute the predictions for all classes/directions
Wb= [W;b]; % construct the return value
f = tprod(X,[-(1:dim(1)-1) 1 -(dim(1)+1:ndims(X))],reshape(Wb(1:end-1,:),[szSx,L-1]),[-(1:dim(1)-1) 0 -(dim(1)+1:ndims(X)) 2],'n');
f = repop(f,'+',Wb(end,:));
J = 0;
return;
%--------------------------------------------------------------------------------------------
function testCases()
%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

[Wb,f,J]=lda(X,Y,0);
plotLinDecisFn(X,Y,Wb(1:end-1,:),Wb(end,:));

% test implicit ignored
[alphab0,f0,J0]=klr_cg(K,Y.*single(fInds(:,end)<0),1,'verb',1,'ridge',1e-7);

% for linear kernel
alpha=zeros(N,1);alpha(find(trnInd))=alphab(1:end-1); % equiv alpha
plotLinDecisFn(X,Y,X(:,trnInd)*alphab(1:end-1),alphab(end),alpha);

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
