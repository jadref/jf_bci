function [res,Cs,fIdxs]=cvtrainFn(objFn,X,Y,Cs,fIdxs,varargin)
% train a classifier using n-fold cross validation
%
% [res,Cs,fIdxs]=cvtrainFn(objFn,X,Y,Cs,fIdxs,varargin)
%
% Inputs:
%  objFn   - [str] the m-file name of the function to call to train the
%            classifier for the current folds.  The function must have the 
%            prototype:
%              [soln,f,J] = objFn(X,Y,C,..)
%            where: soln - solution parameters, 
%                   f - [size(Y)]   classifier decision values (+/-1 based)
%                   J - [size(Y,2)] classifiers objective function value
%              opts.seedNm,val  -- seed argument for the classifier
%              'verb',int -- verbosity level argument of the classifier
%              'dim',int  -- dimension of x along which trials lie
%          N.B. soln can also be used to communicate solution information between
%               calls within the same fold for different paramters
%               If soln is a struct with soln.soln available then *only* soln.soln is 
%               recorded in the solution information of the output.
%  X       - [n-d] inputs with trials in dimension dim
%  Y       - [size(X,dim) x L] matrix of binary-subproblems
%  Cs      - [1 x nCs] set of penalties to test                       ([5.^(3:-1:-3) 0])
%            (N.B. specify in decreasing order for efficiency)
%  fIdxs   - [size(Y) x nFold] 3-value (-1,0,1) matrix indicating which trials
%              to use in each fold, where
%              -1 = training trials, 0 = excluded trials,  1 = testing trials
%             OR
%            [1 x 1] number of folds to use (only for Y=trial labels). (10)
% Options:
%  dim      - [int] dimension of X which contains the trials (ndims(X))
%  binsp    - [bool] do we cut the Y up into independent binary sub-problems? (true)
%             N.B. if not then objFn should solve all the binary-subproblems at once
%  aucNoise - [bool] do we add noise to predictions to fix AUC score 
%               problems. (true)
%  recSoln  - [bool] do we store the solutions found for diff folds/Cs (false) 
%  reuseParms- [bool] flag if we use solution from previous C to seed
%               current run (about 25% faster)                         (true)
%  seedNm   - [str] parameter name for the seeding and parameter reuse ('alphab')
%  seed     - the seed value to use
%  reorderC - [int] do we reorder the processing of the penalty        (1)
%             parameters to the more efficient decreasing order?
%             1-re-order decreasing reg, 0-do nothing, -1=re-order increasing reg
%  verb     - [int] verbosity level
%  keyY     - [size(Y,2) x 1 cell] description of what's in each of Y's sub-probs
%  outerSoln- [bool] compute the outer (i.e. all data) solutions also?  (-1)
%                1 = compute solution for all regularisation parameters
%                0 = don't compute outer solution at all
%               -1 = compute outer solution only for the 'optimal' parameter setting
%  calibrate- [str] calibrate the final classifier predictions using;    ('cr')
%               ''   -- no calibration
%               'cr' -- cv-estimated classification rate
%               'bal' -- balanced classification rate, so each class has same error rate
%  aucWght  - [float] weighting on the AUC score to use to determine the  (.1)
%              optimal hyperparameter
%  dispType - one of: 'bin' ,'auc' - performance information summary to print
%  varargin - any additional parameters are passed to objFn
% Outputs:
% res       - results structure with fields
%   |.fold - per fold results
%   |     |.di      - dimInfo structure describing the contents of the matrices
%   |     |.soln    - {nSp x nCs x nFold} the solution for this fold (if recSoln if true)
%   |     |.f       - [N x nSp x nCs x nFold] classifiers predicted decision value
%   |     |.trnauc  - [1 x nSp x nCs x nFold] training set Area-Under-Curve value
%   |     |.tstauc  - [1 x nSp x nCs x nFold] testing set AUC
%   |     |.trnconf - [4 x nSp x nCs x nFold] training set binary confusion matrix
%   |     |.tstconf - [4 x nSp x nCs x nFold] testing set binary confusion matrix
%   |     |.trnbin  - [1 x nSp x nCs x nFold] training set binary classification performance
%   |     |.tstbin  - [1 x nSp x nCs x nFold] testing set binary classification performance
%   |.opt   - cv-optimised classifer parameters
%   |   |.soln - {nSp x 1}cv-optimal solution
%   |   |.f    - [N x nSp] cv-optimal predicted decision values
%   |   |.C    - [1x1] cv-optimal hyperparameter
%   |.di    - dimInfo structure describing contents of over folds ave matrices
%   |.f        - [N x nSp x nCs] set of full data solution decision values
%   |.tstf     - [N x nSp x nCs] testing test only decision values for each example
%   |.soln     - {nSp x nCs} set of add data solutions
%   |.trnauc   - [1 x nSp x nCs] average over folds training set Area-Under-Curve value
%   |.trnauc_se- [1 x nSp x nCs] training set AUC standard error estimate
%   |.tstauc   - [1 x nSp x nCs] average over folds testing set AUC
%   |.tstauc_se- [1 x nSp x nCs] testing set AUC standard error estimate
%   |.trnconf  - [4 x nSp x nCs] training set binary confusion matrix
%   |.tstconf  - [4 x nSp x nCs] testing set binary confusion matrix
%   |.trnbin   - [1 x nSp x nCs] average over folds training set binary classification performance
%   |.trnbin_se- [1 x nSp x nCs] training set binary classification performance std error
%   |.tstbin   - [1 x nSp x nCs] testing set binary classification performance
%   |.tstbin_se- [1 x nSp x nCs] testing set binary classification performance std error
% Cs - the ordering of the hyper-parameters used 
%       (may be different than input, as we prefer high->low order for efficiency!)
% fIdxs -- the fold structure used. (may be different from input if input is scalar)
opts = struct('binsp',1,'aucNoise',0,'recSoln',0,'dim',[],'ydim',[],'reuseParms',1,...
              'seed',[],'seedNm','alphab','verb',0,'reorderC',1, ...
              'spDesc',[],'outerSoln',-1,'calibrate','bal',...
				  'lossType',[],'lossFn','bal','dispType','bin', 'subIdx',[],'aucWght',.1);
[opts,varargin]=parseOpts(opts,varargin);

dim=opts.dim; if ( isempty(dim) ) dim=ndims(X); end;
szX=size(X); szX(end+1:max(dim))=1;
szY=size(Y); szY(end+1:numel(dim)+1)=1; % true size of Y, inc the sub-prob dim
ydim=opts.ydim; 
if ( isempty(ydim) ) 
   if ( all(szY(1:numel(dim))==szX(dim)) )            
      ydim=1:numel(dim);                        spD=numel(dim)+1:numel(szY);
   elseif( all(szY(end-numel(dim)+1:end)==szX(dim)) ) 
      ydim=numel(szY)-numel(dim)+1:numel(szY);  spD=1:numel(szY)-numel(dim);
   else error('sizes of Y and X dont match');
   end; 
else
   if ( max(ydim)==numel(ydim) ) spD=numel(ydim)+1;
   elseif( min(ydim)>1 )         spD=1:min(ydim);
   end
end

nSubProbs=prod(szY(spD)); if ( ~opts.binsp ) nSubProbs=1; end;
N        =prod(szX(dim));
if ( any(szY(ydim)~=szX(dim)) ) error('Y and X should have the same number of examples'); end
if ( opts.binsp && (~all(Y(:)==-1 | Y(:)==0 | Y(:)==1 | isnan(Y(:)))) )
  error('Y should be matrix of -1/0/+1 label indicators.');
end
if ( isempty(opts.lossFn) && ~isempty(opts.lossType) ) opts.lossFn=opts.lossType; end;
if ( nargin < 4 || isempty(Cs) ) Cs=[5.^(3:-1:-3) 0]; end;
if ( nargin < 5 || isempty(fIdxs) ) fIdxs=10; end;

if ( isscalar(fIdxs) ) 
   nFolds= fIdxs; 
   fIdxs = gennFold(ones([prod(szY(ydim)),1]),nFolds,'dim',2); % folding ignoring structure of Y
   fIdxs = reshape(fIdxs,[szY(ydim) size(fIdxs,2)]); % back to match size of Y
else
   szfIdxs=size(fIdxs);
   if ( all(szfIdxs(ydim)==1 | szfIdxs(ydim)==szY(ydim)) )
      nFolds=size(fIdxs,ndims(fIdxs));
   else
      error('fIdxs isnt compatiable with X,Y');   
   end
end

% convert Y to [N x nSp]
% TODO: This is stupid!! the classifiers internally back-convert to [ nSp x N ]]
oszY=szY;
if ( ~isequal(ydim,1:numel(dim)) ) 
   if( isequal(ydim,numel(szY)-numel(dim)+1:numel(szY)) ) % convert [ nSp x N ] -> [ N x nSp ]
      Y     = permute(Y,[ydim 1:numel(szY)-numel(dim)]);
      szY   = szY([ydim 1:numel(szY)-numel(dim)]); % update size info
      % Update the folding info shape also....
      fIdxs = permute(fIdxs,[ydim 1:numel(szY)-numel(dim) numel(szY)+1:ndims(fIdxs)]);
      % update the dim-info
      ydim  = 1:numel(dim); 
      spD   = numel(dim)+1:numel(szY);
	else
	  error('Only supported with trials in **first** or **last** Y dimensions.'); 
	end
end;

% replicate unity trial dimensions of the folding information so can make it 3d
szfIdxs=size(fIdxs); reps=ones(1,ndims(fIdxs)); reps(szfIdxs(ydim)==1)=szY(szfIdxs(ydim)==1);
fIdxs=repmat(fIdxs,reps);
% insert sub-prob dim if not there
if(ndims(fIdxs)<=spD) 
   szfIdxs=size(fIdxs); fIdxs=reshape(fIdxs,[szfIdxs(1:spD-1) 1 szfIdxs(spD:end)]); 
end; 
% esure fIdxs is 3-d [N x nSp x nFold]
szfIdxs=size(fIdxs); szfIdxs(end+1:max(spD,numel(szY)))=1;
fIdxs=reshape(fIdxs,[prod(szfIdxs(ydim)) prod(szfIdxs(spD)) prod(szfIdxs(numel(szY)+1:end))]);


siCs=1:size(Cs(:,:),2);
if ( opts.reuseParms && opts.reorderC ) 
   % works better if we go from lots reg to little
   if ( opts.reorderC>0 )    [ans,siCs]= sort(sum(Cs(:,:),1),'descend'); 
   else                      [ans,siCs]= sort(sum(Cs(:,:),1),'ascend'); 
   end
   Cs=Cs(:,siCs);
   if( ~isequal(siCs,1:numel(Cs)) )
      if ( opts.reorderC>0 )
         warning(['Re-ordered Cs in *DECREASING* magnitude for efficiency']); 
      else
         warning(['Re-ordered Cs in *INCREASING* magnitude for efficiency']); 
      end
   end;
end

% First compute the whole data solutions for each Cs
if ( opts.outerSoln>0 )
   for spi=1:nSubProbs; % loop over sub-problems
      if ( opts.verb > -1 ) 
         if ( nSubProbs>1 ) fprintf('(out/%2d)\t',spi); else; fprintf('(out)\t'); ...
         end;
      end
      seed=opts.seed; % reset seed for each sub-prob
      if( ~opts.binsp ) spi =1:size(Y,spD); end; % set spi to set sub-probs if not binary
      exInd = all(fIdxs(:,min(end,spi),:)==0,3); % excluded points

      if(numel(dim)>1) Ytrn=reshape(Y,[prod(szY(1:end-1)),szY(end)]); end;
      Ytrn=Ytrn(:,spi); Ytrn(exInd,:)=0;  % get the sub-prob labels, and ensure excluded points are excluded
      if(numel(dim)>1) Ytrn=reshape(Ytrn,[szY(1:end-1),size(Ytrn,2)]); end;
      for ci=1:size(Cs(:,:),2);%siCs; % proc in sorted order
         if( ~opts.reuseParms ) seed=opts.seed; end;
         if( ~isempty(seed) ) 
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',opts.verb-1, ...
                             opts.seedNm,seed,'dim',opts.dim,varargin{:});
         else
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',opts.verb-1, ...
                             'dim',opts.dim,varargin{:});
         end
         if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
         if ( opts.binsp ) res.soln{spi,ci}=sol; else res.soln{ci}=sol; end;
			if ( size(f,1)==size(Ytrn,2) && size(f,2) == size(Ytrn,1) ) f=f'; end; % ensure same size as Y
         res.f(:,spi,ci)=f;      
         if( opts.verb > -1 ) 
            if( numel(spi)>1 ) fprintf('['); end;
            % N.B. we need to loop as dv2conf etc. only work on 1 sub-prob at a time            
            for spii=1:numel(spi); 
               fprintf('%0.2f/NA  ',conf2loss(dv2conf(Ytrn(:,spii),f(:,spii)),1,opts.lossFn)); 
               if( spii<numel(spi) ) fprintf('|'); end;
            end
            if(numel(spi)>1) fprintf(']'); 
               if(numel(spi)<5) fprintf(' '); else fprintf('\n');end;
            elseif ( numel(spi)==1 ) 
               fprintf('\t'); 
            end;
         end
      end % Cs
      if ( opts.verb>-1 )
         if (size(Cs(:,:),2)>1 ) fprintf('\n'); end;
      end
   end
end

res.tstf=zeros([prod(szX(dim)) size(Y,spD) size(Cs(:,:),2)],class(X));
% Next compute the per-fold values
for foldi=1:size(fIdxs,3);
   for spi=1:nSubProbs; % loop over sub-problems get train/test split (possibly sub-prob specific)
      trnInd=fIdxs(:,min(end,spi),foldi)<0;  % training points
      tstInd=fIdxs(:,min(end,spi),foldi)>0;  % testing points
      exInd =fIdxs(:,min(end,spi),foldi)==0; % excluded points
      
		% get the target's for this fold (and sub-problem)
		Ytrn=reshape(Y,prod(szY(ydim)),[]); % start with 2d version of Y - [N x nSp]
      if ( opts.binsp ) Ytrn=Ytrn(:,spi); end % select sub-prob specific targets
      Ytst=Ytrn; % trn/tst same
      Ytrn(tstInd | exInd, :)=0;             Ytst(trnInd | exInd, :)=0;       % excluded points
      if ( opts.binsp ) % back to input size
         Ytrn=reshape(Ytrn,[szY(ydim) 1]);   Ytst=reshape(Ytst,[szY(ydim) 1]);
      else
         Ytrn=reshape(Ytrn,szY);             Ytst=reshape(Ytst,szY);
      end

      if ( opts.verb > -1 )
         if ( size(fIdxs,ndims(fIdxs))>1 ) 
            if ( nSubProbs>1 ) fprintf('(%3d/%2d)\t',foldi,spi); 
            else               fprintf('(%3d)\t',foldi); 
            end
         elseif ( nSubProbs>1 && spi>1 ) fprintf('|'); 
         end
      end
      seed=opts.seed; % seed
      for ci=1:size(Cs(:,:),2);%siCs; % proc in sorted order
         if( ~opts.reuseParms ) seed=opts.seed; end;
         if( ~isempty(seed) ) 
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',opts.verb-1, ...
                             opts.seedNm,seed,'dim',opts.dim,varargin{:});
         else
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',opts.verb-1, ...
                             'dim',opts.dim,varargin{:});
         end
         
         if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
         if ( opts.binsp ) soln{spi,ci,foldi}=sol; else soln{ci,foldi}=sol; end;
			if( size(f,1)~=N ) % ensure predictions (i.e. f) is [N x nSp]
            szf=size(f);
            if ( szf(2)==N )                                    f=permute(f,[2 1 3:ndims(f)]);% f=f';
            elseif ( all(szf(1:numel(dim))==szY(1:numel(dim)))) f=reshape(f,[N szf(numel(dim)+1:end)]);
            elseif ( prod(szf(2:end))==N )                      f=f(:,:)'; 
            else error('f with shape [nSp x d1 x d2...] not supported yet');
            end
         end
			if ( opts.binsp ) spis=spi; else spis=1:size(f,2); end;
			% Initialize the storage for the per-fold predictions
			if(~isfield(res,'fold') || ~isfield(res.fold,'f') || isempty(res.fold.f))
			  res.fold.f=zeros([size(f,1) size(Y,numel(dim)+1) size(Cs(:,:),2) size(fIdxs,ndims(fIdxs))],class(f));
			end
         res.fold.f(:,spis,ci,foldi)=f;
			% accumulate test prediction decision values
         for spii=1:numel(spis);
			  tstIndi = tstInd | exInd; 
           res.tstf(tstIndi,spis(spii),ci)=f(tstIndi,spii); 
			end;
         res.fold.trnconf(:,spis(1),ci,foldi)=dv2conf(reshape(Ytrn,size(f)),f);
         res.fold.tstconf(:,spis(1),ci,foldi)=dv2conf(reshape(Ytst,size(f)),f);
         res.fold.trn(:,spis(1),ci,foldi)=conf2loss(res.fold.trnconf(:,spis(1),ci,foldi),1,opts.lossFn);
         res.fold.tst(:,spis(1),ci,foldi)=conf2loss(res.fold.tstconf(:,spis(1),ci,foldi),1,opts.lossFn);
%          for spii=1:numel(spis);% N.B. we need to loop as dv2conf
%            % etc. only work on 1 sub-prob at a time               
%            res.fold.trnbin (:,spis(spii),ci,foldi)=...
% 			        conf2loss(dv2conf(Ytrn(:,spii),f(:,spii)),1,opts.lossFn);
%            res.fold.tstbin (:,spis(spii),ci,foldi)= ...
%                  conf2loss(dv2conf(Ytst(:,spii),f(:,spii)),1,opts.lossFn);
%            % add some noise to p to ensure dv2auc is real if ( opts.aucNoise )
%            %f(:,spii) = f(:,spii) +
%            %rand(size(f(:,spii)))*max(1,f(:,spii))*1e-4; end;
%            res.fold.trnauc(:,spis(spii),ci,foldi) =dv2auc(Ytrn(:,spii),f(:,spii));
%            res.fold.tstauc(:,spis(spii),ci,foldi) =dv2auc(Ytst(:,spii),f(:,spii));
%          end % sub-probs
         if ( opts.verb>-1 ) % log the performance
            trn=res.fold.trn; tst=res.fold.tst; 
            if ( size(trn,1)>1 ) fprintf('['); end;
            fprintf('%0.2f/%0.2f',trn(:,spis(1),ci,foldi),tst(:,spis(1),ci,foldi));
            if( size(trn,1)>1 ) fprintf(']'); 
              if ( size(trn,2)<5 ) fprintf(' '); else fprintf('\n'); end;
            elseif( size(Cs(:,:),2)>1 ) fprintf('\t'); 
            end;
         end
      end % Cs
      if ( opts.verb > -1 && size(Cs(:,:),2)>1 ) fprintf('\n'); end;
   end % loop over sub-problems
   if (opts.verb>-1 )
      if ( size(fIdxs,ndims(fIdxs))==1 ) fprintf('\t'); 
      elseif ( size(Cs(:,:),2)==1 ) fprintf('\n');
      end
   end
end
szRes=size(res.fold.trn); 
res.fold.di=mkDimInfo(szRes,'perf',[],[],'subProb',[],opts.spDesc,'C',[], ...
                      Cs,'fold',[],[],'dv');
foldD=4;
res.fold.di(foldD).info.fIdxs=fIdxs;
if ( opts.recSoln ) res.fold.soln=soln; end; % record the solution for this fold
res.di     = res.fold.di(setdiff(1:end,foldD)); % same as fold info but without fold dim
res.trnconf= sum(res.fold.trnconf,foldD);
res.tstconf= sum(res.fold.tstconf,foldD);
res.trn    = conf2loss(res.trnconf,1,opts.lossFn);%mean(res.fold.trnbin,foldD);
res.tst    = conf2loss(res.tstconf,1,opts.lossFn);%mean(res.fold.tstbin,foldD);
% if ( opts.binsp )
%   res.trnauc = sum(res.fold.trnauc,foldD)./size(res.fold.trnauc,foldD);
%   %res.trnauc_se=sqrt(abs(sum(res.fold.trnauc.^2,foldD)/nFolds-(res.trnauc.^2))/nFolds);
%   res.tstauc = sum(res.fold.tstauc,foldD)./size(res.fold.tstauc,foldD);
%   %res.tstauc_se=sqrt(abs(sum(res.fold.tstauc.^2,foldD)/nFolds-(res.tstauc.^2))/nFolds);
% end
res.fIdxs  = fIdxs;
res.Y      = Y;

% record the optimal solution and it's parameters
if ( opts.binsp && isfield(res,'tstauc') )
   % best hyper-parameter for all sub-probs
   [opttstbin,optCi]=max(mean(res.tst,2)+opts.aucWght*mean(res.tstauc,2)); 
else
   [opttstbin,optCi]=max(mean(res.tst,2)); % best hyper-parameter for all sub-probs
end
% store per-info for the opt-parameter settings
res.opt.Ci  =optCi;
res.opt.C   =Cs(:,optCi);
res.opt.trnconf=res.trnconf(:,:,optCi);
res.opt.tstconf=res.tstconf(:,:,optCi);
res.opt.tst   = res.tst(:,:,optCi); 
res.opt.trn   = res.trn(:,:,optCi); 
res.opt.tstf  = res.tstf(:,:,optCi);

% print summary of the results
if ( opts.verb > -2 && size(fIdxs,ndims(fIdxs))>1)
   lab='ave'; trn=res.trn; tst=res.tst; 
   if ( opts.verb > -1 ) fprintf('-------------------------\n'); end;
   for spi=1:size(trn,2); % loop over sub-problems
      if ( size(trn,2)>1 ) fprintf('(%3s/%2d)\t',lab,spi); else; fprintf('(%3s)\t',lab); end;
      for ci=1:size(Cs(:,:),2);
         fprintf('%0.2f/%0.2f',trn(:,spi,ci),tst(:,spi,ci));
         if( ci==optCi ) fprintf('*\t'); else fprintf(' \t'); end;
      end
      fprintf('\n');
   end
   if ( size(trn,2)>1 ) % cross problem average performance
      fprintf('(%3s/av)\t',lab); 
      for ci=1:size(Cs(:,:),2);
         fprintf('%0.2f/%0.2f\t',mean(trn(:,:,ci),2),mean(tst(:,:,ci),2));
      end
      fprintf('\n');
   end
end

% Compute the outer-solution properties
if ( opts.outerSoln>0 ) % use classifier trained on all the data
   res.opt.soln=res.soln(:,optCi);
elseif( opts.outerSoln<0 ) % re-train with the optimal parameters found
   for spi=1:nSubProbs; % loop over sub-problems
      if ( opts.verb > -1 ) 
         if ( nSubProbs>1 ) fprintf('(opt/%2d)\t',spi); else; fprintf('(opt)\t'); end;
      end
      if( ~opts.binsp ) spi =1:size(Y,spD); end; % set spi to set sub-probs if not binary
      exInd = all(fIdxs(:,min(end,spi),:)==0,3); % excluded points

      Ytrn=Y; if(numel(dim)>1) Ytrn=reshape(Ytrn,[prod(szY(1:end-1)),szY(end)]); end;
      Ytrn=Ytrn(:,spi); Ytrn(exInd,:)=0;  % get the sub-prob labels, and ensure excluded points are excluded
      if(numel(dim)>1) Ytrn=reshape(Ytrn,[szY(1:end-1),size(Ytrn,2)]); end;

      [seed,f,J]=feval(objFn,X,Ytrn,res.opt.C,'verb',opts.verb-1,...
                       'dim',opts.dim,varargin{:});
      if ( size(f,1)~=N ) f=f'; end; % ensure predictions are [N x nSp]

      if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
      if ( opts.binsp ) res.opt.soln{spi}=sol; else res.opt.soln=sol; end;
		if ( opts.binsp ) spis=spi; else spis=1:size(f,2); end;
      res.opt.f(:,spis)=f; 
      % override tstf information for non-training data examples with those trained on all the data
      % now tstf contains val fold predictions in training set, and opt predictions for the rest
      res.opt.tstf(exInd,spis)  =f(exInd,:);
      res.opt.trnconf(:,spis(1))=dv2conf(reshape(Ytrn,size(f)),f);
      res.opt.trn(:,spis(1))    =conf2loss(res.opt.trnconf(:,spis(1)),opts.lossFn);
      if( opts.verb > -1 ) 
        fprintf('%0.2f/NA  ',res.opt.trn(:,spis(1)));
      end
   end % spi
   if( opts.verb > -1 ) fprintf('\n'); end;

elseif( opts.outerSoln==0 ) % estimate from per fold solutions  
   if ( opts.binsp ) opt=soln(:,optCi,:); else opt=soln(optCi,:); end; % spi x 1 x fold
   if ( isnumeric(soln{1}) || (iscell(soln{1}) && isnumeric(soln{1}{1})) ) % use average over all folds solutions
      if ( opts.binsp ) res.opt.soln = opt(:,1,1); else res.opt.soln=opt(1,1); end;
      for fi=2:size(opt,3);
         for spi=1:size(opt,1);
            if ( isnumeric(opt{1}) )
               res.opt.soln{spi}=res.opt.soln{spi}+opt{spi,1,fi};
            else
               for bi=1:numel(opt{spi,1,fi})
                  res.opt.soln{spi}{bi}=res.opt.soln{spi}{bi}+opt{spi,1,fi}{bi};
               end
            end
         end
      end
      for spi=1:size(opt,1); % average the result
         if ( isnumeric(opt{1}) )
            res.opt.soln{spi}=res.opt.soln{spi}./size(opt,3);
         else
            for bi=1:numel(opt{spi,1,1})
               res.opt.soln{spi}{bi}=res.opt.soln{spi}{bi}./size(opt,3);
            end
         end
      end
      res.opt.f    = sum(res.fold.f(:,:,optCi,:),4)./size(res.fold.f,4);
   else % use classifier trained on 1 single fold of the data
      optFold=1;
      res.opt.soln=soln(:,optCi,optFold);
      res.opt.f   =res.fold.f(:,:,optCi,optFold);		
   end
end
% override tstf information for non-training data examples with those trained on all the data
% now tstf contains val fold predictions in training set, and opt predictions for the rest
% BODGE: doesn't allow for per-sub-problem folding
trnexInd = all(all(fIdxs<=0,3),2); % trials which are only ever excluded points or training points
res.opt.tstf(trnexInd,:,:)=res.opt.f(trnexInd,:,:);

% Calibrate the optimal classifier output probabilities
% N.B. only for linear classifiers!
trnexInd = all(all(fIdxs<=0,3),2); % tst points
if ( ~all(trnexInd) && opts.binsp && ~isempty(opts.calibrate) && ~isequal(opts.calibrate,0) ) 
   cr=res.tst(:,:,optCi); % cv-estimated probability of being correct - target for calibration
   %if ( strcmp(opts.calibrate,'bal') ) cr = cr([1 1],:,:); end; % balanced calibration
   % correct the targets to prevent overfitting
   cwght=[];
   Ycal=Y; if ( numel(dim)>1 ) Ycal=reshape(Y,[prod(szY(1:end-1)),szY(end)]); end;
   if ( strcmp(opts.calibrate,'bal') ) % balanced calibration, so equal class weights
      cwght(1,:)=sum(Ycal(~trnexInd,:)~=0,1)./sum(Ycal(~trnexInd,:)<0)./2; 
      cwght(2,:)=sum(Ycal(~trnexInd,:)~=0,1)./sum(Ycal(~trnexInd,:)>0)./2;
   end
   [Ab]=dvCalibrate(Ycal(~trnexInd,:),res.tstf(~trnexInd,:,res.opt.Ci),cr,cwght);
   for i=1:numel(res.opt.soln);
      if ( iscell(res.opt.soln{i}) ) % cell, assume W is full and b is separate!
         res.opt.soln{i}{1}  =res.opt.soln{i}{1}*Ab(1,i);
         res.opt.soln{i}{end}=res.opt.soln{i}{end}*Ab(1,i)+Ab(2,i);
      else  % non-cell, assume is wb format
         res.opt.soln{i}(1:end-1)=res.opt.soln{i}(1:end-1)*Ab(1,i);
         res.opt.soln{i}(end)    =res.opt.soln{i}(end)*Ab(1,i)+Ab(2,i);
      end
      % update the predictions also
      res.opt.f(:,i)    = res.opt.f(:,i)*Ab(1,i)+Ab(2,i);
      res.opt.tstf(:,i) = res.tstf(:,i,res.opt.Ci)*Ab(1,i)+Ab(2,i);
   end  
   res.opt.cal.scale=Ab(1,:); res.opt.cal.offset=Ab(2,:);
end

return;

%----------------------------------------------------------------------------
function []=testCase()

% test binary performance
[X,Y]=mkMultiClassTst([-.5 0; .5 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
labScatPlot(X,Y)
K=compKernel(X,[],'linear','dim',-1);
res=cvtrainFn('klr_cg',K,Y,10.^[-3:3],10)

% test mc performance
[X,Y]=mkMultiClassTst([-1 1; 1 1; 0 0],[400 400 400],[.3 .3; .3 .3; .2 .2],[],[1 2 3]);
labScatPlot(X,Y)

% 1vR
Yind=lab2ind(Y);
Ysp=lab2ind(Y,sp);
res=cvtrainFn(X,Ysp,10.^[-3:3],10);
% now extract the multi-class per-fold/C performance from the recorded info
pc=dv2pred(res.fold.f,-1,'1vR');%convert to predicted class per example/C/fold
conf=pred2conf(Yind,pc,[1 -1]);   % get mc-confusion matrix
muconf=sum(conf,3);              % sum/average over folds
conf2loss(conf,'bal')            % get a nice single mc-performance measure

%1v1
[sp,spDesc]=mc2binSubProb(unique(Y),'1v1');
Yind=lab2ind(Y);
Ysp=lab2ind(Y,sp);
res=cvtrainFn(X,Ysp,10.^[-3:3],10);
% now extract the multi-class per-fold/C performance from the recorded info
pc=dv2pred(res.fold.f,-1,'1v1');%convert to predicted class per example/C/fold
conf=pred2conf(Yind,pc,[1 -1]);  % get mc-confusion matrix
muconf=sum(conf,3);              % sum/average over folds
conf2loss(conf,'bal')            % get a nice single mc-performance measure


% Double nested cv classifier training
Y = floor(rand(100,1)*(3-eps))+1;
nOuter=10;nInner=10;
outerfIdxs = gennFold(Y,nOuter);
fi=1;
for fi=1:size(outerfIdxs,2);
   Ytrn = (Y.*double(outerfIdxs(:,fi)<0));
   Ytst = (Y.*double(outerfIdxs(:,fi)>0));
   innerfIdxs = gennFold(Ytrn,nInner);
   
   % Inner cv to determine model parameters
   res.outer(fi)=cvtrainFn(X,Ytrn,innerfIdxs);
   % Model parameters are best on the validation set
   [ans optI] = max(res.outer(fi).tstauc); Copt = Cs(optI);
   
   % Outer-cv performance recording
   if ( opts.binsp )
     res.trnauc(:,fi) =dv2auc(Ytrn,res.outer(fi).f(:,:,optI));   
     res.tstauc(:,fi) =dv2auc(Ytst,res.outer(fi).f(:,:,optI));
   end
   res.trnconf(:,fi)=dv2conf(Ytrn,res.outer(fi).f(:,:,optI));  
   res.tstconf(:,fi)=dv2conf(Ytst,res.outer(fi).f(:,:,optI));
   res.trn(:,fi)    =conf2loss(res.trn(:,fi),'cr');
   res.tst(:,fi)    =conf2loss(res.tst(:,fi),'cr');
end


% test with calibration
N=200; nD=300;  % enough dims to cause over-fitting
[X,Y]=mkMultiClassTst([-.5 0 zeros(1,nD); .5 0 zeros(1,nD); 0 1 zeros(1,nD)],[N N N],[.3 .3 ones(1,nD); .3 .3 ones(1,nD); .3 .3 ones(1,nD)],[],[1 2 3]);
labScatPlot(X,Y)
Yl=Y; Y=lab2ind(Y);
K=compKernel(X,[],'linear','dim',-1);
res=cvtrainFn('klr_cg',K,Y,10.^[-3:3],10,'calibrate',1,'outerSoln',0)
