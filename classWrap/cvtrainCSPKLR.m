function [res,Cs,fIdxs]=cvtrainCSPKLR(objFn,X,Y,Cs,fIdxs,varargin)
% cross-validated train of a CSP+nFeat and KLR+reg_parm classifier
%
% [res,Cs,fIdxs]=cvtrainCSPKLR(objFn,X,Y,Cs,fIdxs,varargin)
% Inputs:
%   X -- [ nCh x ... x N ] data matrix.  
%         N.B. we assume dim(1)==channels, dim(3)==epochs
%   Y -- [ N x nClass ] set of +/-1/0 per-label sub-problem indicators
%   Cs-- [ 1 x nCs ] set of KLR reg parameters to try
%  fIdxs  -- [ N x nFolds ] set of +/-1/0 sub-fold indicators
% Options:
%  nFeats -- [ 1 x nFeats] set of numbers of CSP features to try (picked
%            from each end of the eigeenvalue distribution)
% Outputs:
%  res  -- results information
if ( isnumeric(objFn) ) % BODGE: work with old calling format
   varargin={fIdxs varargin{:}};
   fIdxs=Cs;
   Cs   =Y;
   Y    =X;
   X    =objFn;
   objFn='klr_cg';
end
opts = struct('nFeats',6,'aucNoise',1,'centCSP',0,'log',1,...
              'recSoln',0,'dim',[],'reuseParms',1,'seed',[],'seedNm','alphab','verb',0,...
              'spDesc',[],'outerSoln',0,'calibrate',[],'binsp',1,'lossType','bal','reorderC',1);
[opts,varargin]=parseOpts(opts,varargin);

if ( ndims(Y)>2 || ~any(size(Y,1)==size(X)) ) error('Y should be a vector with N elements'); end;
if ( nargin < 4 || isempty(Cs) ) Cs=10.^(3:-1:-3); end;
if ( nargin < 5 || isempty(fIdxs) ) 
   nFolds=10; fIdxs = gennFold(Y,nFolds,'perm',1);
elseif ( isscalar(fIdxs) ) 
   nFolds=fIdxs; fIdxs = gennFold(Y,nFolds,'perm',1);
   if ( nFolds==1 ) fIdxs=-fIdxs; end;
elseif ( size(fIdxs,1)==size(Y,1) )
   nFolds=size(fIdxs,2);
else
   error('fIdxs isnt compatiable with X,Y');
end
nFeats=opts.nFeats;

if ( opts.reuseParms && opts.reorderC ) 
   % works better if we go from lots reg to little
   [ans,siCs]= sort(Cs(1,:),'descend'); Cs=Cs(:,siCs);
   if( ~isequal(siCs,1:numel(Cs)) )
      warning(['Re-ordered Cs in dec magnitude for efficiency']); 
   end;
end

if ( isempty(objFn) ) objFn='klr_cg'; end;
if ( ~strcmp(objFn,'klr_cg') ) error('Only klr_cg objective function supported currently'); end;
if ( ~isempty(opts.outerSoln) && opts.outerSoln ) warning('OuterSoln not implemented yet!'); end;
if ( ~isempty(opts.calibrate) && opts.calibrate ) warning('Calibrate not implemented yet!'); end;
if ( ndims(X)>3 || (~isempty(opts.dim) && opts.dim~=3) ) 
   error('Only 3-d, [ch x time x epoch] data is currently supported'); 
end;
if (isempty(opts.dim) ) opts.dim=3; end;

% get number of sub-Problems to solve: N.B. binary is special case
if( ~opts.binsp ) 
   nSubProbs=1;
else
   nSubProbs=size(Y,2);   
   if( nSubProbs==2 && all(Y(:,1)==-Y(:,2)) ); nSubProbs=1; end
end; 
if(ndims(fIdxs)<=2) fIdxs=reshape(fIdxs,[size(fIdxs,1),1,size(fIdxs,2)]); end; %include subProb dim

% Loop over the folds training and testing CSP+classifier
for foldi=1:nFolds;
   for spi=1:nSubProbs; % loop over sub-problems
      if ( ~opts.binsp ) spi=1:size(Y,2); end; % set spi to set sub-probs if not binary
      % get the training test split (possibly sub-prob specific)
      trnInd=fIdxs(:,min(end,spi),foldi)<0;  % training points
      tstInd=fIdxs(:,min(end,spi),foldi)>0;  % testing points
      exInd =fIdxs(:,min(end,spi),foldi)==0; % excluded points
      
      Ytrn  =Y(:,spi); Ytrn(tstInd)=0; Ytrn(exInd)=0;
      Ytst  =Y(:,spi); Ytst(trnInd)=0; Ytst(exInd)=0;
      seed=[]; % use prev C's seed in next one to speed things up

      % train the csp to compute spatial filters
      [sf,d,Sigmai,Sigmac,Sigma] = csp(X,Ytrn,[opts.dim 1],opts.centCSP); 

      % Compute the new features
      sfX    = tprod(X,[-1 2 3 4 5],sf,[-1 1]); % map to csp space [sf x time x epoch]
      sfXXsf = tprod(sfX,[1 -2 2 3 4],[],[1 -2 2])./size(X,2); % map to feature variances, [sf x epoch]
      if ( opts.log ) sfXXsf=log(abs(sfXXsf)); end;
      
      if ( foldi==1 ) % estimate the Cscale
         K=compKernel(sfXXsf,[],'linear','dim',-1);
         Cscale = .1*(mean(diag(K))-mean(K(:)));   % Norm the pen parameter      
      end
      
      % Order the features by eigenvalue
      nSF=sum(d>0); % d==0 indicates invalid
      si=[];
      si(1:2:nSF)=1:ceil(nSF/2);
      si(2:2:nSF)=nSF:-1:ceil(nSF/2)+1;   
      [ans,sd]=sort(d(1:nSF),'descend'); % sort by eigvalue
      si=sd(si);
      % Re-order the features
      d=d(si); sf=sf(:,si); sfX=sfX(si,:,:); sfXXsf=sfXXsf(si,:);
      
      for featIdx=1:numel(nFeats);
         nFeat=min(nFeats(featIdx),size(sfXXsf,1));
         
         % Compute the new kernel
         K=compKernel(sfXXsf(1:nFeat,:),[],'linear','dim',-1);
         %fprintf('Condition number of K: %g\n',rcond(K));      
         
         % N.B. we should do multi-class training here!
         %if( size(Ytrn,2)~=2 ) error(['Only implemented for binary problems']);end
         %spi=1; % only work with the first sub-problem
         if ( opts.verb > -1 ) % logging info
            if ( nSubProbs>1 ) 
               if(numel(nFeats)==1) fprintf('(%3d/%2d)\t',foldi,spi); else fprintf('(%3d,%2d/%2d)\t',foldi,nFeat,spi); end;
            else 
               if(numel(nFeats)==1) fprintf('(%3d)\t',foldi); else fprintf('(%3d,%2d)\t',foldi,nFeat); end;
            end
         end
         for cIdx=1:numel(Cs);
            if( ~opts.reuseParms ) seed=[]; end;
            [seed,f,J]=feval(objFn,K,Ytrn,Cscale*Cs(cIdx),varargin{:},...
                               opts.seedNm,seed,'verb',opts.verb-1);
            res.fold.C(cIdx,featIdx,foldi,spi)  =Cs(cIdx);%*Cscale;
            res.fold.f(:,cIdx,featIdx,foldi,spi)=f;
            res.fold.trnconf(:,cIdx,featIdx,foldi,spi)= dv2conf(Ytrn,f);
            res.fold.tstconf(:,cIdx,featIdx,foldi,spi)= dv2conf(Ytst,f);
            res.fold.trnbin(:,cIdx,featIdx,foldi,spi) = ...
                conf2loss(res.fold.trnconf(:,cIdx,featIdx,foldi,spi),1,opts.lossType);
            res.fold.tstbin(:,cIdx,featIdx,foldi,spi) = ...
                conf2loss(res.fold.tstconf(:,cIdx,featIdx,foldi,spi),1,opts.lossType);
            % add some noise to p to ensure dv2auc is real
            if ( opts.aucNoise )  f = f + randn(size(f))*mean(f)*1e-4; end;
            res.fold.trnauc(:,cIdx,featIdx,foldi,spi) = dv2auc (Ytrn,f);
            res.fold.tstauc(:,cIdx,featIdx,foldi,spi) = dv2auc (Ytst,f);
            fprintf('%0.2f/%0.2f\t',res.fold.trnbin(:,cIdx,featIdx,foldi,spi),...
                    res.fold.tstbin(:,cIdx,featIdx,foldi,spi));
         end
         fprintf('\n');
      end
   end
end;
spDesc=opts.spDesc; if ( isstr(spDesc) && numel(spDesc)>nSubProbs ) spDesc={spDesc}; end
res.fold.di=mkDimInfo(size(res.fold.trnauc),'perf',[],[],'C',[],Cscale*Cs,...
                      'nFeat',[],nFeats,'fold',[],[],'subProb',[],spDesc,'dv');
foldD=n2d(res.fold.di,'fold');
[res.fold.di(foldD).extra.fIdxs]=num2csl(fIdxs,1);
res.di     = res.fold.di(setdiff(1:end,foldD)); % same as fold info but without fold dim
res.trnconf= sum(res.fold.trnconf,foldD);
res.tstconf= sum(res.fold.tstconf,foldD);
res.trnbin = msqueeze(4,mean(res.fold.trnbin,foldD));
res.trnbin_se=sqrt((msqueeze(foldD,sum(res.fold.trnbin.^2,foldD))/nFolds-(res.trnbin.^2))/nFolds);
res.tstbin = msqueeze(foldD,mean(res.fold.tstbin,foldD));
res.tstbin_se=sqrt((msqueeze(foldD,sum(res.fold.tstbin.^2,foldD))/nFolds-(res.tstbin.^2))/nFolds);
res.trnauc = msqueeze(foldD,mean(res.fold.trnauc,foldD));
res.trnauc_se=sqrt((msqueeze(foldD,sum(res.fold.trnauc.^2,foldD))/nFolds-(res.trnauc.^2))/nFolds);
res.tstauc = msqueeze(foldD,mean(res.fold.tstauc,foldD));
res.tstauc_se=sqrt((msqueeze(foldD,sum(res.fold.tstauc.^2,foldD))/nFolds-(res.tstauc.^2))/nFolds);

fprintf('-------------------------\n',nFolds);
for featIdx=1:numel(nFeats);
   for spi=1:size(res.trnauc,4);
      if ( size(res.trnauc,4)==1) 
         if ( numel(nFeats)==1 ) fprintf('(ave)\t'); else fprintf('(ave,%2d)\t',nFeats(featIdx)); end;
      else fprintf('(ave,%2d/%2d)\t',nFeats(featIdx),spi); end;
      for cIdx=1:numel(Cs);
         fprintf('%0.2f/%0.2f\t',res.trnbin(:,cIdx,featIdx,spi),res.tstbin(:,cIdx,featIdx,spi));
      end
      fprintf('\n');
   end
end
fprintf('\n');

return;
%----------------------------------------------------------------------------
function testCase()
expt='eeg/motor/im-tapping';subj='pd';label='nips2008';
z=jf_load(expt,subj,label);

% Generate the outer folds
oY         = lab2ind(cat(1,z.di(n2d(z,'epoch')).extra.marker));
z.foldIdxs = gennFold(oY,10,'perm',0);

[res]=cvtrainCSPKLR(z.X,oY,10.^[-5:6],[2 4 8 16],z.foldIdxs);


% Compare with Borris
z=struct('expt',expt,'subj',subj,'label',label);
z.X=reshape(single(training_set.data),[size(training_set.data,1) ...
                    training_set.featsize]); % extract data
z.X=permute(z.X,[2 3 1]); % permute to my normal order [nCh x nSamp x N]
z.di = mkDimInfo(size(z.X),'ch',[],[],'time','samp',[],'epoch',[],[]);
[Y,uY]=lab2ind(training_set.nlab);
z.Y=Y;
z.foldIdxs = gennFold(Y,10,'perm',0);
z=addprepjf(z,'prtools2jf','Converted from prtools format');
dispjf(z)
tic,
[res]=cvtrainCSPKLR([],z.X,z.Y(:,1),10.^[-5:6],[2 4 8 16],z.foldIdxs);
toc
