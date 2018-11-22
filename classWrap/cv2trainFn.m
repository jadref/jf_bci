function [res,Cs,outerfIdxs]=cv2trainFn(objFn,X,Y,Cs,outerfIdxs,varargin)
% double nested n-fold cross validation classifier training
%
% [res,Cs,outerfIdxs]=cvtrainFn(objFn,X,Y,Cs,outerfIdxs,nInner,varargin)
%
% Inputs:
%  objFn   - [str] the m-file name of the function to call to train the
%            classifier for the current folds.  The function must have the 
%            prototype:
%              [w,f,J] = objFn(X,Y,C,..)
%            where: w - solution parameters, 
%                   f - [size(Y)]   classifier decision values (+/-1 based)
%                   J - [size(Y,2)] classifiers objective function value
%              'hps',val  -- seed argument for the classifier
%              'verb',int -- verbosity level argument of the classifier
%              'dim',int  -- dimension of x along which trials lie
%  X       - [n-d] inputs with trials in dimension dim
%  Y       - [size(X,dim) x L] matrix of binary-subproblems
%  Cs      - [1 x nCs] set of penalties to test     (10.^(3:-1:3))
%            (N.B. specify in decreasing order for efficiency)
%  outerfIdxs   - [size(Y) x nFold] 3-value (-1,0,1) matrix indicating which trials
%              to use in each fold, where
%              -1 = training trials, 0 = excluded trials,  1 = testing trials
%             OR
%            [1 x 1] number of folds to use (only for Y=trial labels). (10)
%  nInner - [1 x 1] number of inner folds to use (5)
opts = struct('nInner',5,'verb',0,'cvtrainFn','cvtrainFn');
cvtrainopts = struct('lossFn','bal');
[opts,cvtrainopts,varargin]=parseOpts({opts,cvtrainopts},varargin);

if ( ndims(Y)>2 || ~any(size(Y,1)==size(X)) ) 
   error('Y should be a vector with N elements'); 
end
if ( ~all(Y(:)==-1 | Y(:)==0 | Y(:)==1) )
  error('Y should be matrix of -1/0/+1 label indicators. Try using Y=lab2ind(Y) 1st.');
end
if ( nargin < 4 || isempty(Cs) ) Cs=10.^(3:-1:-3); end;
if ( nargin < 5 || isempty(outerfIdxs) ) nFolds=10; 
elseif ( isscalar(outerfIdxs) )          nFolds=outerfIdxs; outerfIdxs=[];
elseif( size(outerfIdxs,1)~=size(Y,1) )  error('fIdxs isnt compatiable with X,Y');
end
if ( isempty(outerfIdxs) ) 
  outerfIdxs = gennFold(Y,nFolds,'perm',1);
end
if( ndims(outerfIdxs)>2 && size(outerfIdxs,2)>1 ) error('not supported yet!'); end;
nFolds=size(outerfIdxs,ndims(outerfIdxs));

fi=1;
for fi=1:size(outerfIdxs,ndims(outerfIdxs));
   Ytrn = Y; Ytrn(outerfIdxs(:,fi)>=0,:)=0; % N.B. can be more than 1 subproblem!
   Ytst = Y; Ytst(outerfIdxs(:,fi)<=0,:)=0;
   innerfIdxs = gennFold(Ytrn,opts.nInner);
   
   % Inner cv to determine model parameters, and then train with optimal parameters on all training data
   res.outer(fi)=feval(opts.cvtrainFn,objFn,X,Ytrn,Cs,innerfIdxs,varargin{:},cvtrainopts,'outerSoln',-1,'verb',opts.verb-1);
   % Model parameters are best on the validation set
   opt = res.outer(fi).opt;
   
   % outer-cv performance recording
   if ( fi==1 )
     res.fold.f   =opt.f;
     res.fold.soln={opt.soln}; % N.B. soln is subProb x fold
     res.fold.C   =opt.C(:)';
     res.fold.Ci  =opt.Ci(:)';   
   else
     res.fold.f(:,:,fi)    =opt.f;
     res.fold.soln{:,fi}   =opt.soln; % N.B. soln is subProb x fold
     res.fold.C(1,:,fi)    =opt.C;
     res.fold.Ci(1,:,fi)   =opt.Ci;   
   end	
	if ( exist(cvtrainopts.lossFn,'file') ) % lossFn is function to call to compute loss directly
	  res.fold.trn(:,:,fi)=feval(cvtrainopts.lossFn,Ytrn,opt.f);
	  res.fold.tst(:,:,fi)=feval(cvtrainopts.lossFn,Ytst,opt.f);
	  % BODGE, store also in trnbin.... even if not binary
	  res.fold.trnbin=res.fold.trn;
	  res.fold.tstbin=res.fold.tst;
	else % lossFn is type of loss to compute
     for spi=1:size(Y,2);
		 res.fold.trnauc(:,spi,fi) =dv2auc(Ytrn(:,spi),opt.f(:,spi));
		 res.fold.tstauc(:,spi,fi) =dv2auc(Ytst(:,spi),opt.f(:,spi));
		 res.fold.trnconf(:,spi,fi)=dv2conf(Ytrn(:,spi),opt.f(:,spi));
		 res.fold.tstconf(:,spi,fi)=dv2conf(Ytst(:,spi),opt.f(:,spi));
		 res.fold.trnbin(:,spi,fi) =conf2loss(res.fold.trnconf(:,spi,fi),cvtrainopts.lossFn);
		 res.fold.tstbin(:,spi,fi) =conf2loss(res.fold.tstconf(:,spi,fi),cvtrainopts.lossFn);
	  end     
   end
	% log the performance
   if ( opts.verb > -1 )
	  for spi=1:size(Y,2);
		 if ( size(Y,2)>1 ) fprintf('(out%3d/%2d)\t',fi,spi); else; fprintf('(out%3d)\t',fi); end;
		 fprintf('%0.2f/%0.2f \t',res.fold.trnbin(:,spi,fi),res.fold.tstbin(:,spi,fi));       
		 fprintf('\n');
	  end   
	end
end
szRes=size(res.fold.trnbin);
res.fold.di = mkDimInfo(szRes,'perf',[],[],'subProb',[],[],'fold',[],[],'dv');
res.fold.fIdxs=outerfIdxs;
foldD=3;
res.trnbin = sum(res.fold.trnbin,foldD)./size(res.fold.trnbin,foldD);
res.tstbin = sum(res.fold.tstbin,foldD)./size(res.fold.tstbin,foldD);
if ( isfield(res.fold,'trnconf') )  res.trnconf= sum(res.fold.trnconf,foldD); end;
if ( isfield(res.fold,'tstconf') )  res.tstconf= sum(res.fold.tstconf,foldD); end;
if ( isfield(res.fold,'trnauc') )   
  res.trnauc = sum(res.fold.trnauc,foldD)./size(res.fold.trnauc,foldD); 
end;
if ( isfield(res.fold,'tstauc') ) 
  res.tstauc = sum(res.fold.tstauc,foldD)./size(res.fold.trnauc,foldD); 
end;
if ( opts.verb > -1 )
  fprintf('(out ave)\t');
  for spi=1:size(res.trnbin,2);
    fprintf('%0.2f/%0.2f\t',res.trnbin(:,spi),res.tstbin(:,spi));
  end
  fprintf('\n');
end

return;

%----------------------------------------------------------------------------
function testCase()
[X,Y]=mkMultiClassTst([-.5 0; .5 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
labScatPlot(X,Y)
K=compKernel(X,[],'linear','dim',-1);
res=cv2trainFn('klr_cg',K,Y,[],10)

% test mc performance
[X,Y]=mkMultiClassTst([-1 1; 1 1; 0 0],[400 400 400],[.3 .3; .3 .3; .2 .2],[],[1 2 3]);
K=compKernel(X,[],'linear','dim',-1);
Yl=Y; [Y,key,spMx]=lab2ind(Y);
[res,Cs,fIdxs]=cv2trainFn('klr_cg',K,Y,[],2,2)
cvmcPerf(Y,res.fold.f,[1 2 3],fIdxs)
