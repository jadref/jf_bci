function [optsoln,res]=cvSVD(X,varargin)
% find a good parafac decomposition of the input matrix
%
%  [Popt,res]=cvSVD(X,varargin)
%
% Options:
%  fIdxs   - fold guide, N.B. we use the *testing set* (fIdxs>0) to train with!!!!
%  ranks   - [nRank x 1] set of ranks to test
%  tol0    - [float] convergence tolerance (1e-3)
%  maxIter - [int] maximum iterations
%  outerSoln  - [bool] compute global solution using all the data (1)
%  reuseParms - [bool] re-use parameters from previous C's in subsequence runs for speed (1)
%  objFuzz  - [1x4] weighting for the objective metrics: [tstsse rank mean(stab) min(stab)]

% %  objThresh- [1x4] minimum threshold value for each of the objFuzz metrics

opts=struct('fIdxs',[],'ranks',[],'verb',1,'outerSoln',1,'tol0',1e-3,'tol',1e-4,'maxIter',1000,'svdpm',1,'reuseParms',1,'foldType','bicvFold','objFuzz',[1 1/50 1/8 0]);
foldOpts=struct('perm',0,'oneBlock',0);
[opts,foldOpts,varargin]=parseOpts({opts,foldOpts},varargin);
nd=ndims(X);
szX=size(X);

idx={}; for d=1:nd; idx{d}=1:szX(d); end; % index expression for folding
ranks=opts.ranks; if ( isempty(ranks) ) ranks=1:min(szX); end;

% setup the folding
fIdxs=opts.fIdxs; nFold=size(fIdxs,ndims(fIdxs));
if ( numel(fIdxs)==1 || numel(fIdxs)==2 ) nFold=fIdxs; fIdxs=[]; 
end;
if ( isempty(fIdxs) ) % generate requested folding
  if ( numel(nFold)>1 ) nRep=nFold(2); nFold=nFold(1); else nRep=1; end;
  if ( nFold==0 ) nFold=4; nRep=3; end;
  if ( strcmp(opts.foldType,'bicvFold') )
    fIdxs=bicvFold(szX,abs(nFold),'repeats',nRep,foldOpts);
  else
    fIdxs=int8(gennFold(ones([prod(szX),1]),abs(nFold),'repeats',nRep,'perm',1));
  end
  if ( nFold<0 ) fIdxs=-fIdxs; end; % swap train/test if wanted
end
if ( ndims(fIdxs)>2 ) % reshape to 2-d
  fIdxs=reshape(fIdxs,[prod(szX),size(fIdxs,numel(szX)+1)]);
end
nFold=size(fIdxs,ndims(fIdxs));
if ( opts.verb>0 ) fprintf('nFold=%d\n',size(fIdxs,ndims(fIdxs))); end;
if ( size(fIdxs,1) ~= prod(szX) ) 
  error('folding size doesnt agree with X size');
end

X2=X(:)'*X(:);
if ( opts.verb>0 ) 
  fprintf('(fold) rank trn/tst %%Err\n'); 
end

if ( opts.outerSoln )
  if ( opts.svdpm )
    [U,S,V]=svdpm(X,max(ranks));
  else
    [U,S,V]=svd(X,'econ');S=diag(S);
  end
  outsoln={S,U,V};
  cs=cumsum(S.^2);
  outsse=1-(cs./X2);
  fprintf('(out)\t');
  fprintf('%.2f/NA  \t',outsse(ranks));
  fprintf('\n');
end

% Now do the loop over inner folds
X2=X(:)'*X(:);
Pi={};
for fi=1:nFold;
  trnInd = reshape(fIdxs(:,fi)<0,szX);
  tstInd = reshape(fIdxs(:,fi)>0,szX);
  trnX2=sum(X(~tstInd).^2);
  tstX2=sum(X(tstInd).^2);
    
  Xi=X;
  Xi(tstInd)=0;
  % rank1 est of the unknown values?
  mu1=sum(Xi,2);
  mu2=sum(Xi,1);
  mu1=mu1./sqrt(mu1(:)'*mu1(:)).*sqrt(sqrt(tstX2)); mu2=mu2./sqrt(mu2(:)'*mu2(:)).*sqrt(sqrt(tstX2));

  if ( opts.verb>0 ) fprintf('(%3d)\t',fi); end
  for ri=1:numel(ranks);    
    nComp=ranks(ri);
    % model using only this data + number of ranks
    if ( ri==1 || ~opts.reuseParms ) 
      Xest=mu1*mu2;
    end
    Xi(tstInd) = Xest(tstInd);    % replace unknown values with seed's estimate
    for iter=1:opts.maxIter;
      if ( opts.svdpm )
        [U,S,V]=svdpm(Xi,nComp,[],[],[],U);
      else
        [U,S,V]=svd(Xi,'econ');S=diag(S);
      end
      Xest = U(:,1:nComp)*diag(S(1:nComp))*V(:,1:nComp)';        % predicted value
      dX  = sum((Xi(tstInd)-Xest(tstInd)).^2); % norm of the change
      if ( opts.verb>1 )
        Xerr=X-Xest;
        fprintf('%d) trnsse=%.2f\ttstsse=%.2f\n',iter,sum(Xerr(~tstInd).^2)./trnX2,sum(Xerr(tstInd).^2)./tstX2);
      end
      if ( iter==1 ) dX0=dX; end;
      if ( dX<opts.tol0*dX0 || dX<opts.tol ) break; end; % convergence test
      Xi(tstInd)=Xest(tstInd);    % replace unknown value with new value
    end
    soln{ri,fi}={S(1:nComp) U(:,1:nComp) V(:,1:nComp)}; % record solution in parafac fashion
    % estimate fit performance
    dX=X-Xest;
    trnsse(ri,fi)=sum(dX(~tstInd).^2)./trnX2;
    tstsse(ri,fi)=sum(dX(tstInd).^2)./tstX2;
    % estimate solution stability    
    if ( opts.verb>0 ) 
      fprintf('%.2f/%.2f\t',trnsse(ri,fi),tstsse(ri,fi));
    end
  end % rank
  if ( opts.verb>0 ) fprintf('\n'); end;
end % fold

% compute other solution summary statistics
% cross fold correlation -> solution stability
for ri=1:numel(ranks);
  nComp=ranks(ri);
  c=zeros(nFold,nFold); cc=zeros([nComp,nFold,nFold]);
  for i=1:nFold;
    for j=1:i-1;
      [cij,ccij]=parafacCorr(soln{ri,i},soln{ri,j});
      c(i,j)=cij; c(j,i)=cij;
      cc(:,i,j)=ccij; cc(:,j,i)=ccij;  % [nComp x nFold x nFold]
    end
    c(i,i)=1; cc(:,i,i)=1; % with itself has corr=1
  end
  compstab(1:ri,ri)=sum(sum(cc,2),3)./nFold/nFold;
  stab(:,:,ri) = shiftdim(sum(cc,1))./nComp;
end

if ( opts.verb>0 ) fprintf('---------------------\n'); end;
if ( opts.verb>0 )
  fprintf('(ave)\t');
   for nCompi=1:numel(ranks);
      nComp=ranks(nCompi);
      fprintf('%.2f/%.2f\t',mean(trnsse(nCompi,:),2),mean(tstsse(nCompi,:),2));
   end
   fprintf('\n');
   fprintf('(stb)\t');
   for nCompi=1:numel(ranks);
      nComp=ranks(nCompi);
      fprintf('%.2f     ',sum(sum(stab(:,:,nCompi)))./nFold./nFold);
      fprintf('\t');
   end
   fprintf('\n');
end
if ( opts.verb>0 ) fprintf('---------------------\n'); end;

minstab=zeros(1,size(compstab,2));for i=1:size(compstab,2); minstab(i)=min(compstab(find(compstab(:,i)>0),i)); end;
obj=cat(2,sum(tstsse,2)./size(tstsse,2),ranks(:)./max(ranks),1-shiftdim(sum(sum(stab))./nFold./nFold).^2,minstab');
[ans,optnCompi]=min(obj*opts.objFuzz(:)); % use rank for tie-breaker
optnComp=ranks(optnCompi);

% fit model with these parameters on the total data
if ( opts.verb>0 ) fprintf('(r=%2d)*\t',ranks(optnCompi)); end;
if ( opts.outerSoln ) % already done, just pick it up
   optsoln={outsoln{1}(1:optnComp) outsoln{2}(:,1:optnComp) outsoln{3}(:,1:optnComp)};
   if ( opts.verb>0 ) fprintf('%2f\n',outsse(optnCompi)); end;
else
  [U,S,V]=svd(X,'econ');S=diag(S);
  Popt={S(1:optnCompi),U(1:optnCompi),V(1:optnCompi)};
  if ( opts.verb>0 ) fprintf('%.2f\n',sum(Popt{1})./X2); end;
end
if ( opts.verb>0 ) fprintf('\n'); end;

% results structure
if ( nargout>1 )
  res.di    =mkDimInfo(size(soln),'rank',[],ranks,'fold',[],[]);
  res.opts  =opts;
  res.fIdxs =reshape(fIdxs,[szX size(fIdxs,2)]);
  res.opt.soln =optsoln;
  res.opt.optnCompi=optnCompi;
  res.opt.optRank=ranks(optnCompi);
  res.fold.soln  =soln;
  res.fold.trnsse=trnsse;
  res.fold.tstsse=tstsse;
  res.trnsse     =mean(trnsse,2);
  res.tstsse     =mean(tstsse,2);
  res.compstab   =compstab;
  res.stab       =stab;
  res.soln  =outsoln;
  res.outsse=outsse;
end
return;

%----------------------------------------------------------
function testCase()

rank=3;
t={2.^-(1:rank)' randn(100,rank) randn(50,rank)};
t={2.^-(1:rank)' randn(1000,rank) randn(250,rank)};
t{1}=t{1}./sqrt(sum(t{1}.^2))./1; t{2}=repop(t{2},'/',sqrt(sum(t{2}.^2)));t{3}=repop(t{3},'/',sqrt(sum(t{3}.^2)));%unit norm
A=t{2}*diag(t{1})*t{3}';
A=A+randn(size(A))*1./sqrt(numel(A));
[optP,res]=cvSVD(A,'ranks',1:4);
[optP,res]=cvSVD(A,'ranks',1:4,'foldType','rand');
clf;mimage(A,optP{2}*diag(optP{1})*optP{3}','diff',1,'clim','minmax');

[c,cc]=parafacCorr(t,optP) % similarity of final soln and true model
