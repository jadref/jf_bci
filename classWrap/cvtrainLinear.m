function [res,K]=cvtrainKer(X,Y,Cs,fIdxs,kerType,dim)
if ( nargin < 3 || isempty(Cs) ) Cs=1; end;
if ( nargin < 4 || isempty(fIdxs) ) 
   nFolds=10; fIdxs = gennFold(Y,nFolds,'perm',1);
elseif ( isscalar(fIdxs) ) 
   nFolds=fIdxs; fIdxs = gennFold(Y,nFolds,'perm',1);
elseif ( any(size(fIdxs,1)==numel(Y)) )
   nFolds=size(fIdxs,2);
else
   error('fIdxs isnt compatiable with X,Y');
end
if ( nargin < 5 || isempty(kerType) ) kerType='linear'; end;
if ( nargin < 6 || isempty(dim) ) dim=-1; end;

% --------------------------------------------------------------------------
% Comp the kernel for this data
K     = compKernel(z.X,[],kerType,'dim',-1);

% clf; % plot the kernel
% axes('outerposition',[.25 .25 .75 .75]); imagesc(K);set(gca,'YDir','normal');
% axes('outerposition',[ 0  .25 .25 .75]); plot(Y,1:numel(Y));axis([-1.1 1.1 1 numel(Y)]);
% axes('outerposition',[.25  0  .75 .25]); plot(1:numel(Y),Y);axis([1 numel(Y) -1.1 1.1]);

% Generate an n-fold split of the data
for foldIdx=1:nFolds;
   trnInd = ~fIdxs(:,foldIdx);  tstInd = fIdxs(:,foldIdx);
   Ytrn(:,foldIdx)  = Y; Ytrn(tstInd, foldIdx)=0; 
   Ytst(:,foldIdx)  = Y; Ytst(trnInd, foldIdx)=0;
   alphab=[]; % use prev C's seed in next one to speed things up
   for cIdx=1:numel(Cs);
      [alphab,p,J]=klr_cg(K,Ytrn(:,foldIdx),Cs(cIdx),'alphab',alphab,...
                            'tol',1e-8,'maxEval',10000);
      res.outer.p(:,cIdx,foldIdx)=p;
      res.outer.trnauc(:,cIdx,foldIdx)  = dv2auc (Ytrn(:,foldIdx),p);
      res.outer.tstauc(:,cIdx,foldIdx)  = dv2auc (Ytst(:,foldIdx),p);
      res.outer.trnconf(:,cIdx,foldIdx) = dv2conf(Ytrn(:,foldIdx),(p-.5)*2);
      res.outer.tstconf(:,cIdx,foldIdx) = dv2conf(Ytst(:,foldIdx),(p-.5)*2);
      res.outer.trnbin(:,cIdx,foldIdx)  = conf2loss(res.outer.trnconf(:,cIdx,foldIdx),1,'bin');
      res.outer.tstbin(:,cIdx,foldIdx)  = conf2loss(res.outer.tstconf(:,cIdx,foldIdx),1,'bin');
   end
end;
res.outer.di=mkDimInfo(size(res.outer.trnauc),[],[],[],'C',[],Cs,'Fold',[],[]);
[res.outer.di(3).extra.fIdxs]=num2csl(fIdxs,1);
res.di     = mkDimInfo(size(Cs),[],[],[],'C',[],Cs);
res.trnbin = conf2loss(sum(res.outer.trnconf,3),'bin');
res.tstbin = conf2loss(sum(res.outer.tstconf,3),'bin');
res.trnauc = mean(res.outer.trnauc,3);
res.tstauc = mean(res.outer.tstauc,3);
