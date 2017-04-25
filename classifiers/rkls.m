function [wb,f,J,obj]=rkls(K,Y,C,varargin)
% Regularised Kernel Least Squares Classifier
%
% [alphab,J]=rkls(K,Y,C,varargin)
% Simple regularised kernel least squares classifier.
%
% J = C(1)*w'*K*w + sum_i (y_i - (w'*K_i + b)).^2
%
% Inputs:
% K       - [N x N] kernel matrix
% Y       - [N x 1] matrix of +1, -1 labels, for an L-class problem,
%           N.B. points with label 0 are ignored
% C       - [1 x 1] regularisation parameter
%
% Outputs:
% alphab  - [(N+1) x 1] matrix of kernel weights and bias [alpha;b]
% f       - [N x 1] The decision value for all the inputs
% J       - the final objective value
%
% Options:
%
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
%            [1x1] total weight, divided number elements in each class
%  bias    - [bool] flag if we want bias computed (true)
%  tol     - [float] tolerance to use for the matrix inversion to solve (1e-5)
% 
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied

% Argument processing
if ( nargin < 3 ) C(1)=0; end;
opts=struct('alphab',[],'bias',1,'dim',[],'wght',0,'method','kls','uk',1,'tol',[],'state',[],'savestate',1,'verb',0);
ignoredOpts=struct('maxEval',[],'objTol',[],'objTol0',[]);
[opts,ignoredOpts]=parseOpts({opts,ignoredOpts},varargin{:});
if ( isempty(opts.state) && isstruct(opts.alphab) ) opts.state=opts.alphab; opts.alphab=[]; end;

% Identify the training points
szK=size(K);
dim=opts.dim; if ( isempty(dim) ) dim=ndims(K); end;
if ( ndims(K)>2 ) K=reshape(K,prod(szK(1:dim-1)),szK(dim)); dim=2; end % convert to 2-d
if ( size(Y,1)~=size(K,dim) ) Y=Y'; end; % ensure is col vector
trnInd = (Y~=0); 
N=sum(trnInd);

% Now train on the training points.
Ytrn = Y(trnInd,:);
if ( opts.wght ) % reweight
   if( numel(opts.wght)<=2 ) 
      Ytrn(Ytrn>0)=Ytrn(Ytrn>0)./sum(Ytrn>0)*opts.wght(min(end,1)); 
      Ytrn(Ytrn<0)=Ytrn(Ytrn<0)./sum(Ytrn<0)*opts.wght(min(end,2));
   end
end;

% Pick the subset of data/features we want
if( size(K,1)==size(K,2) && isequal(opts.method,'kls') ) % subset features
   featInd=trnInd; % kernel, so feat subset == training subset
else
   featInd=1:size(K,1); % non-kernel, use all features
end

% N.B. use pinv to find the min-norm solution, also threshold to ensure good inverse
% N.B. use SVD/EIG to get min-norm solution
if ( ~opts.savestate || isempty(opts.state) || ...
     ~(isstruct(opts.state) && isfield(opts.state,'U') && isfield(opts.state,'S') && isfield(opts.state,'V') ) )

  % don't use the saved state & compute the SVD as needed
  if( ~all(trnInd) ) % subset if needed
    Ktrn=K(featInd,trnInd); 
  else 
    Ktrn=K; 
  end

  % Solve the least-squares problem
  uk=opts.uk;
  if( uk>0 ) % norm kernel avoid numerical errors
    if ( size(K,1)==size(K,2) ) uk=median(abs(diag(Ktrn))); else uk=median(abs(Ktrn(:))); end;
    C=C./uk; 
    Ktrn=Ktrn./uk; 
  end;
  if ( opts.bias ) % agument the kernel/features to include a bias term
    % to preserve the condition number of the kernel
    if ( size(K,1)==size(K,1)) muEig=mean(abs(diag(Ktrn))); 
    else muEig=abs(mean(Ktrn(:))); end
    Ktrn=[Ktrn;ones(1,size(Ktrn,2))*muEig]; % agumented kernel
  end

  [U,S,V]=svd(Ktrn,0); S=diag(S); 
  
else % use the saved state info

  U=opts.state.U; S=opts.state.S; V=opts.state.V;
  uk=opts.state.uk;
  if (uk>0) C=C./uk; end;
  muEig=opts.state.muEig;
end
oS=S; % save copy before add regularisor

% add the regulariser - if wanted
if( C>0 ) S=S+C; end  

% find the optimal weighting using the SVD decomposition computed above
if( ~isempty(opts.tol) && opts.tol~=0 )
  si=(abs(S)>tol*max(abs(S)));
  w = repop(U(:,si),'./',S(si)')*V(:,si)'*Ytrn; % SVD based
  %w = pinv(Ktrn,tol*sum(diag(Ktrn)))'*Ytrn; % pinv based
elseif ( C>=0 ) % use default tol
  tol = (numel(S)+opts.bias) * eps(max(S)); si=(abs(S)>tol*max(abs(S)));
  w = repop(U(:,si),'./',S(si)')*V(:,si)'*Ytrn; % SVD based
  %w = pinv(Ktrn)'*Ytrn; % 
else % don't use the pinv
  w = Ktrn'\Ytrn; % 
end

% extract/compute the bias term
if ( opts.bias )
  b = w(end)./muEig; w = w(1:end-1); 
else
  f = Ktrn*w;
  b = optBias(f,Ytrn);
end

% Extract the final solution
if( uk>0 ) w=w./uk; end; % undo kernel normalisation
wb = zeros(size(K,1)+1,size(w,2)); % N.B. w.r.t. the full input set
wb(featInd,:) = w; wb(end,:)=b;

% Now compute the predictions and the objective function value
Kw = K(:,trnInd)*w; %Kw = K'*wb(1:end-1,:);
f  = Kw + wb(end); f=reshape(f,size(Y));
Ed = sum((Y(trnInd)-f(trnInd)).^2);
Ew = wb(1:end-1)'*Kw;
J  = C*Ew + Ed;
if ( opts.verb >= 0 ) 
   fprintf(['x=[%8f,%8f,.] J=%5f (%5f+%5f)\n'],wb(1),wb(2),J,Ew./muEig,Ed);
end
if ( size(K,1)==size(K,2) && isequal(opts.method,'kls') ) J= J + abs(C(1))*wb(1:end-1)'*Kw ; end;
if ( opts.savestate ) % save state as well as solution
  wb=struct('soln',wb,'U',U,'S',oS,'V',V,'uk',uk,'muEig',muEig); 
end;
obj=[J Ew Ed];

% % BODGE: test the loo performance estimate
% [f_loo]=rkls_loo(f(trnInd,:),Ytrn,C,U,oS,V);
% fprintf('loo=%4g\t',conf2loss(dv2conf(f_loo,Y(trnInd,:))));%,conf2loss(dv2conf(f_lootrue,Y(trnInd,:))));

return;

%-------------------------------------------------------------------------
function testCase()

% easy
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
% hard
[X,Y]=mkMultiClassTst([-1 0 zeros(1,80); 1 0 zeros(1,80); .2 .5 zeros(1,80)],[400 400 100],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

K=X'*X; % N.B. add 1 to give implicit bias term
fInds=-gennFold(Y,[],'perm',1,'foldSize',20); trnInd=any(fInds(:,1:9),2); tstInd=fInds(:,10);

[alphab,f,J]=rkls(K,Y.*double(trnInd),1);
dv=K*alphab(1:end-1)+alphab(end);
dv2conf(Y(tstInd),dv(tstInd))

% cvestimated
[clsfr,opts]=cvtrainLinearClassifier(K,Y,[],fInds,'objFn','rkls','outerSoln',0);

% with a optShrinkage estimated reg parameter
[lambda,Sigma]=optShrinkage(X,1);
C = lambda./(1-lambda)*mean(diag(Sigma));
clsfr=cvtrainLinearClassifier(K,Y,C,fInds,'objFn','rkls','outerSoln',0,'Cscale',1);
[alphab,f,J]=rkls(K,Y.*double(trnInd),1);

% for linear kernel
plotLinDecisFn(X,Y,X*alphab(1:end-1),alphab(end),alphab(1:end-1));

