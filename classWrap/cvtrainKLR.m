function [res,K]=cvtrainKLR(X,Y,Cs,fIdxs,kerType,varargin)
% train a KLR classifier using n-fold cross validation
%
% [res,K]=cvtrainKLR(X,Y,Cs,fIdxs,kerType,dim,varargin)
% 
% Inputs:
%  X       - n-dim inputs with trials in dimension dim
%  Y       - [size(X,dim) x 1] matrix of trial labels
%            OR
%            [size(X,dim) x L] matrix of binary-subproblems
%  Cs      - [nCs x 1] set of penalties to test 
%            (N.B. specify in increasing order for efficiency)
%  fIdxs   - [size(X,dim) x nFolds] logical matrix indicating which trials
%            to use in each fold, OR
%            [1 x 1] number of folds to use (only for Y=trial labels). (10)
%  kerType - Type of kernel to use, as in compKernel, ('linear')
% Options:
%  dim     - dimension of X which contains trials, (1st non-singlenton)
%  aucNoise - [bool] do we add noise to predictions to fix AUC score 
%               problems. (true)
%  recSoln  - [bool] do we store (false) 
%  reuseParms- [bool] flag if we use solution from previous C to seed
%               current run (true)
%  reorderC - [bool] do we reorder the processing of the penalty
%             parameters to the more efficient decreasing order? (true)
%  verb     - [int] verbosity level
%  spType   - [str] for multi-sub-problem types the sub-problem type
%  keyY     - [size(Y,2) x 1 cell] description of what's in each of Y's sub-probs
% Outputs:
% res       - results structure with fields
%   |.fold - per fold results
%   |     |.di      - dimInfo structure describing the contents of the matrices
%   |     |.f       - classifiers predicted decision value
%   |     |.rt      - run-time for this training run
%   |     |.trnauc  - training set Area-Under-Curve value
%   |     |.tstauc  - testing set AUC
%   |     |.trnconf - training set binary confusion matrix
%   |     |.tstconf - testing set binary confusion matrix
%   |     |.trnbin  - training set binary classification performance
%   |     |.tstbin  - testing set binary classification performance
%   |.di    - dimInfo structure describing contents of over folds ave matrices
%   |.trnauc   - average over folds training set Area-Under-Curve value
%   |.trnauc_se- training set AUC standard error estimate
%   |.tstauc   - average over folds testing set AUC
%   |.tstauc_se- testing set AUC standard error estimate
%   |.trnconf  - training set binary confusion matrix
%   |.tstconf  - testing set binary confusion matrix
%   |.trnbin   - average over folds training set binary classification performance
%   |.trnbin_se- training set binary classification performance std error
%   |.tstbin   - testing set binary classification performance
%   |.tstbin_se- testing set binary classification performance std error
opts = struct('dim',-1,'objFn','klr_cg','verb',0);
[opts,varargin]=parseOpts(opts,varargin);
if ( ndims(Y)>2 || ~any(size(Y,1)==size(X)) ) error('Y should be a vector with N elements'); end;
if ( nargin < 3 ) Cs=[]; end;
if ( nargin < 4 ) fIdxs=[]; end;
if ( nargin < 5 || isempty(kerType) ) kerType='linear'; end;
if ( isempty(opts.dim) ) 
   opts.dim=find(size(X)>1,1); if(isempty(opts.dim)) opts.dim=1;end; 
end;
if ( opts.dim<0 ) opts.dim=opts.dim+ndims(X)+1; end;

% Comp the kernel for this data
if ( opts.verb > -1 ) fprintf('Comp Kernel...'); end;
K     = compKernel(X,[],kerType,'dim',opts.dim);
if ( opts.verb > -1 ) fprintf('done\n'); end;

% call cvtrainFn to do the real work
[res]=cvtrainFn(opts.objFn,K,Y,Cs,fIdxs,'dim',opts.dim,'verb',opts.verb,varargin{:});
return;

%----------------------------------------------------------------------------
function testCase()

% test binary performance
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
labScatPlot(X,Y)
res=cvtrainKLR(X,Y,10.^[-3:3],10)

% test mc performance
[X,Y]=mkMultiClassTst([-1 1; 1 1; 0 0],[400 400 400],[.3 .3; .3 .3; .2 .2],[],[1 2 3]);
labScatPlot(X,Y)

% 1vR
[sp,spDesc]=mc2binSubProb(unique(Y),'1vR');
Yind=lab2ind(Y);
Ysp=lab2ind(Y,sp);
res=cvtrainKLR(X,Ysp,10.^[-3:3],10);
% now extract the multi-class per-fold/C performance from the recorded info
pc=dv2pred(res.fold.f,-1,'1vR');%convert to predicted class per example/C/fold
conf=pred2conf(Yind,pc,[1 -1]);   % get mc-confusion matrix
muconf=sum(conf,3);              % sum/average over folds
conf2loss(conf,'bal')            % get a nice single mc-performance measure

%1v1
[sp,spDesc]=mc2binSubProb(unique(Y),'1v1');
Yind=lab2ind(Y);
Ysp=lab2ind(Y,sp);
res=cvtrainKLR(X,Ysp,10.^[-3:3],10);
% now extract the multi-class per-fold/C performance from the recorded info
pc=dv2pred(res.fold.f,-1,'1v1');%convert to predicted class per example/C/fold
conf=pred2conf(Yind,pc,[1 -1]);  % get mc-confusion matrix
muconf=sum(conf,3);              % sum/average over folds
conf2loss(conf,'bal')            % get a nice single mc-performance measure


% Double nested cv classifier training
Y = floor(rand(100,1)*(3-eps))+1;
nFold=10;
outerfIdxs = gennFold(Y,nInner);
fi=1;
for fi=1:size(outerfIdxs,2);
   Ytrn = (Y.*double(outerfIdxs(:,fi)<0));
   Ytst = (Y.*double(outerfIdxs(:,fi)>0));
   innerfIdxs = gennFold(Ytrn,nOuter);
   
   % Inner cv to determine model parameters
   res.outer(fi)=cvtrainKLR(X,Y,innerfIdxs);
   % Model parameters are best on the validation set
   [ans optI] = max(res.outer(fi).tstauc); Copt = Cs(optI);
   
   % Re-train with all training data with this model parameter
   [alphab,p,J,dv]=klr_cg(K,Ytrn,Copt,'tol',1e-8,'maxEval',10000,'verb',-1);
   
   % Outer-cv performance recording
   res.trnauc(:,fi) =dv2auc(Ytrn,p);   res.tstauc(:,fi) =dv2auc(Ytst,p);
   res.trnconf(:,fi)=dv2conf(Ytrn,p);  res.tstconf(:,fi)=dv2conf(Ytst,p);
   res.trnbin(:,fi) =conf2loss(trnconf(:,fi),'cr');
   res.tstbin(:,fi) =conf2loss(tstconf(:,fi),'cr');
end

