function [res,Cs,fIdxs]=cvtrainMap(objFn,X,Y,Cs,fIdxs,varargin)
% train a mapping function from X->Y using n-fold cross validation
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
%  Y       - [n-d] features to predict with trials in dimension ydim, i.e. size(Y,ydim)==size(X,dim) 
%                  and size(Y,ydim+1)=L=number sub-problems
%  Cs      - [1 x nCs] set of penalties to test     (10.^(3:-1:3))
%            (N.B. specify in decreasing order for efficiency)
%  fIdxs   - [size(X,dim) x L x nFold] 3-value (-1,0,1) matrix indicating which trials
%              to use in each fold, where
%              -1 = training trials, 0 = excluded trials,  1 = testing trials
%             OR
%            [1 x 1] number of folds to use (only for Y=trial labels). (10)
% Options:
%  dim      - [int] dimension(s) of X which contains the trials           (ndims(X))
%  ydim     - [int] dimension(s) of Y which contains the traisl           (ndims(Y))
%  lossFn   - [string] function to use to assess the performance (higher=better)
%                  of the solution found.
%                  lossFn should have the signature: perf = lossFn(Y,Yest)
%                 see, e.g. est2corr
%  binsp    - [bool] do we cut Y into sub-problems and call to solve independently (true)
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
%  varargin - any additional parameters are passed to objFn
% Outputs:
% res       - results structure with fields
%   |.fold - per fold results
%   |     |.di      - dimInfo structure describing the contents of the matrices
%   |     |.soln    - {nSp x nCs x nFold} the solution for this fold (if recSoln if true)
%   |     |.f       - [N x nSp x nCs x nFold] classifiers predicted decision value
%   |     |.trn     - [1 x nSp x nCs x nFold] training set performance measure
%   |     |.tst     - [1 x nSp x nCs x nFold] testing set performance measure
%   |.opt   - cv-optimised classifer parameters
%   |   |.soln - {nSp x 1}cv-optimal solution
%   |   |.f    - [N x nSp] cv-optimal predicted decision values
%   |   |.C    - [1x1] cv-optimal hyperparameter
%   |.di    - dimInfo structure describing contents of over folds ave matrices
%   |.f        - [N x nSp x nCs] set of full data solution decision values
%   |.tstf     - [N x nSp x nCs] testing test only decision values for each example
%   |.soln     - {nSp x nCs} set of add data solutions
%   |.trn      - [1 x nSp x nCs] average over folds training set binary classification performance
%   |.tst      - [1 x nSp x nCs] testing set binary classification performance
% Cs - the ordering of the hyper-parameters used 
%       (may be different than input, as we prefer high->low order for efficiency!)
% fIdxs -- the fold structure used. (may be different from input if input is scalar)
opts = struct('recSoln',0,'dim',[],'ydim',[],'binsp',0,...
              'reuseParms',1,'seed',[],'seedNm','alphab','verb',0,'reorderC',1,...
              'spDesc',[],'outerSoln',-1,'calibrate','bal','lossFn','est2corr',...
              'subIdx',[],'aucWght',.1);
[opts,varargin]=parseOpts(opts,varargin);

dim=opts.dim;   if ( isempty(dim) )  dim=ndims(X);   end;
szX=size(X); szX(end+1:max(dim))=1;     
szY=size(Y); szY(end+1:numel(dim)+1)=1; 
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

% get number of sub-Problems to solve: N.B. binary is special case
nSubProbs=prod(szY(spD)); if ( ~opts.binsp ) nSubProbs=1; end;
N        =prod(szX(dim));
if ( any(szY(ydim)~=szX(dim)) ) error('Y and X should have the same number of examples'); end
if ( nargin < 4 || isempty(Cs) ) Cs=[5.^(3:-1:-3) 0]; end;
if ( nargin < 5 || isempty(fIdxs) )  fIdxs = 10; end;
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
idxY={};for d=1:numel(szY); idxY{d}=1:szY(d); end;

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

% extract the set of regularisation parameters to use
siCs=1:size(Cs,2);
if ( opts.reuseParms && opts.reorderC ) 
   % works better if we go from lots reg to little
   if ( opts.reorderC>0 )    [ans,siCs]= sort(Cs(1,:),'descend'); 
   else                      [ans,siCs]= sort(Cs(1,:),'ascend'); 
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
for spii=1:nSubProbs; % loop over sub-problems
   if ( opts.binsp ) spi=spii; else spi=1:prod(szY(spD)); end;
   if ( opts.verb > -1 ) 
      if ( nSubProbs>1 ) fprintf('(out/%2d)\t',spi); else; fprintf('(out)\t'); end;
   end
   seed=opts.seed; % reset seed for each sub-prob

   Ytrn = Y;	
   exInd = all(fIdxs(:,min(end,spi),:)==0,3); 
	%if ( nSubProbs>1 ) % get this sub-problems info
	  Ytrn=reshape(Ytrn,prod(szY(ydim)),[]);  % convert to [ N x nSp ]
	  Ytrn=Ytrn(:,spi);  % sub-prob specific
	  Ytrn(exInd,spi)=NaN; % excluded points
	  Ytrn=reshape(Ytrn,szY); % back to input size
	%else
	%  idxY{ydim}=exInd;
	%  Ytrn(idxY{:})=NaN;
	%end;
   for ci=1:size(Cs,2);%siCs; % proc in sorted order
      if( ~opts.reuseParms ) seed=opts.seed; end;
      if( ~isempty(seed) ) 
         [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',max(-1,opts.verb-1),...
                          opts.seedNm,seed,'dim',opts.dim,'ydim',opts.ydim,varargin{:});
      else
         [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',max(-1,opts.verb-1),...
                          'dim',opts.dim,'ydim',opts.ydim,varargin{:});
      end
      if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
      res.soln{spi(1),ci}=sol;
      res.f(:,spi,ci)=reshape(f,N,[]); % f is [N x nSp]
      if( opts.verb > -1 ) 
        fprintf('%0.2f/NA  \t',feval(opts.lossFn,Ytrn,f)); 
      end
   end % Cs
   if ( opts.verb>-1 )
     if (size(Cs,2)>1 ) fprintf('\n'); end;
   end
end
end

res.tstf=zeros([prod(szY(ydim)) prod(szY(spD)) size(Cs,2)],class(Y));
% Next compute the per-fold values
for foldi=1:size(fIdxs,3);
   for spii=1:nSubProbs; % loop over sub-problems
      if( opts.binsp ) spi=spii; else spi=1:prod(szY(spD)); end;
      % get the training test split (possibly sub-prob specific)
      trnInd=fIdxs(:,min(end,spi(1)),foldi)<0;  % training points
      tstInd=fIdxs(:,min(end,spi(1)),foldi)>0;  % testing points
      exInd =fIdxs(:,min(end,spi(1)),foldi)==0; % excluded points
      
		% get the target's for this fold (and sub-problem)
		Ytrn=reshape(Y,prod(szY(ydim)),[]); % start with 2d version of Y - [N x nSp]
      if ( opts.binsp ) Ytrn=Ytrn(:,spi); end % select sub-prob specific targets
      Ytst=Ytrn; % trn/tst same
      Ytrn(tstInd | exInd, :)=NaN;           Ytst(trnInd | exInd, :)=NaN;       % excluded points
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
      for ci=1:size(Cs,2);%siCs; % proc in sorted order
         if( ~opts.reuseParms ) seed=opts.seed; end;
         if( ~isempty(seed) ) 
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',max(-1,opts.verb-1),...
                             opts.seedNm,seed,'dim',opts.dim,'ydim',opts.ydim,varargin{:});
         else
            [seed,f,J]=feval(objFn,X,Ytrn,Cs(:,ci),'verb',max(-1,opts.verb-1),...
                             'dim',opts.dim,'ydim',opts.ydim,varargin{:});
         end         
         if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
         soln{spi(1),ci,foldi}=sol;

			if( size(f,1)~=N ) % ensure predictions (i.e. f) is [N x nSp]
            szf=size(f);
            if ( szf(2)==N )                                    f=permute(f,[2 1 3:ndims(f)]);% f=f';
            elseif ( all(szf(1:numel(dim))==szY(1:numel(dim)))) f=reshape(f,[N szf(numel(dim)+1:end)]);
            elseif ( prod(szf(2:end))==N )                      f=f(:,:)'; 
            else error('f with shape [nSp x d1 x d2...] not supported yet');
            end
         end
         
         % compute the loss
         res.fold.trn(:,spi(1),ci,foldi)=feval(opts.lossFn,Ytrn,f);
         res.fold.tst(:,spi(1),ci,foldi)=feval(opts.lossFn,Ytst,f);

			% accumulate test prediction decision values.
			if(~isfield(res.fold,'f'))
			  res.fold.f=zeros([N size(f,2) size(Cs,2) size(fIdxs,ndims(fIdxs))],class(f));
			end
         if ( opts.binsp ) 
            res.fold.f(:,spi,ci,foldi)        = f;
            res.tstf(tstInd | exInd,spi,ci)   = f(tstInd | exInd,:); 
         else  % multiple outputs at once
            res.fold.f(:,1:size(f,2),ci,foldi)        = f;
            res.tstf(tstInd | exInd,1:size(f,2),ci)   = f(tstInd | exInd,:); 
         end
			
         if ( opts.verb>-1 ) % log the performance
			  fprintf('%3.2f/%3.2f ',[res.fold.trn(:,spi(1),ci,foldi) res.fold.tst(:,spi(1),ci,foldi)]');
           if( size(Cs,2)>1 && size(res.fold.trn,1)>1 ) fprintf('\n      '); end;
         end
      end % Cs
      if ( opts.verb > -1 && size(Cs,2)>1 ) fprintf('\n'); end;
    end % loop over sub-problems
    if (opts.verb>-1 )
      if ( size(fIdxs,ndims(fIdxs))==1 ) fprintf('\t'); 
      elseif ( size(Cs,2)==1 ) fprintf('\n');
      end
    end
end
szRes=size(res.fold.trn); 
% BODGE: check if perf-is actaully per-subproblem
if( ~opts.binsp && szRes(1)==numel(opts.spDesc) && szRes(2)==1 ) % perf is per-sub-prob
  res.fold.trn=permute(res.fold.trn,[2 1 3:ndims(res.fold.trn)]);
  res.fold.tst=permute(res.fold.tst,[2 1 3:ndims(res.fold.tst)]);
  szRes       =szRes([2 1 3:end]);
end
% make the dimension info
res.fold.di=mkDimInfo(szRes,'perf',[],[],'subProb',[],[],'C',[],Cs,'fold',[],[],'dv');
if(szRes(2)==numel(opts.spDesc) ) res.fold.di(2).vals=opts.spDesc; end;
foldD=4;
res.fold.di(foldD).info.fIdxs=fIdxs;
if ( opts.recSoln ) res.fold.soln=soln; end; % record the solution for this fold
res.di     = res.fold.di(setdiff(1:end,foldD)); % same as fold info but without fold dim
res.trn    = sum(res.fold.trn,foldD)./size(res.fold.trn,foldD);
res.tst    = sum(res.fold.tst,foldD)./size(res.fold.trn,foldD);
res.fIdxs  = fIdxs;
res.Y      = Y;

% record the optimal solution and it's parameters
[opttstbin,optCi]=max(sum(sum(res.tst,2),1)); % best hyper-parameter for all sub-probs, and features
res.opt.Ci  =optCi;
res.opt.C   =Cs(optCi);
res.opt.tst =res.tst(:,:,optCi); 
res.opt.trn =res.trn(:,:,optCi); 
res.opt.tstf=res.tstf(:,:,optCi);

% print summary of the results
% final summary performance on all the sub-problems (if there were some)
if ( opts.verb > -2 && size(fIdxs,ndims(fIdxs))>1)
   if ( opts.verb > -1 ) fprintf('-------------------------\n'); end;
   for spi=1:size(res.trn,2); % loop over sub-problems
     if ( size(res.trn,2)>1 ) fprintf('(ave/%2d)\t',spi); else; fprintf('(ave)\t'); end;
     for ci=1:size(Cs,2);
		 fprintf('%3.2f/%3.2f ',[res.trn(:,spi,ci) res.tst(:,spi,ci)]');
		 if( ci==optCi ) fprintf('* '); else fprintf('  '); end;
		 if( size(Cs,2)>1 ) if ( size(res.fold.trn,1)>1 ) fprintf('\n      '); else fprintf('\t'); end; end;
	  end
     fprintf('\n');
   end
   if ( size(res.trn,2)>1 ) % cross problem average performance
      fprintf('(ave/av)\t'); 
      for ci=1:size(Cs,2);
		  fprintf('%3.2f/%3.2f ',[mean(res.trn(:,:,ci),2) mean(res.tst(:,:,ci),2)]');
		  if( ci==optCi ) fprintf('* '); else fprintf('  '); end;
		  if( size(Cs,2)>1 ) if ( size(res.fold.trn,1)>1 ) fprintf('\n      '); else fprintf('\t'); end; end;		  
      end
      fprintf('\n');
   end
end

% Compute the outer-solution properties
exInd = all(fIdxs<=0,3); % tst points
if ( opts.outerSoln>0 ) % use classifier trained on all the data
   res.opt.soln=res.soln(:,optCi);
   % override tstf information for non-training data examples with those trained on all the data
   % now tstf contains val fold predictions in training set, and opt predictions for the rest
   for spi=1:nSubProbs; % loop over sub-problems
     exInd = all(fIdxs(:,min(end,spi),:)==0,3); % excluded points         
     res.opt.tstf(:,exInd,spi)=res.opt.f(:,exInd,spi);
   end
elseif( opts.outerSoln<0 ) % re-train with the optimal parameters found
  for spii=1:nSubProbs; % loop over sub-problems
    if( opts.binsp ) spi=spii; else spi=1:prod(szY(spD)); end;
    if ( opts.verb > -1 ) 
      if ( nSubProbs>1 ) fprintf('(opt/%2d)\t',spi); else; fprintf('(opt)\t'); end;
    end
	 Ytrn  = Y;
    exInd = all(fIdxs(:,min(end,spi),:)==0,3); 

    % get the target's for this fold (and sub-problem)
    Ytrn=reshape(Y,prod(szY(ydim)),[]); % start with 2d version of Y - [N x nSp]
    if ( opts.binsp ) Ytrn=Ytrn(:,spi); end % select sub-prob specific targets
    Ytrn(exInd, :)=NaN;                 % excluded points
    if ( opts.binsp ) % back to input size
       Ytrn=reshape(Ytrn,[szY(ydim) 1]);
    else
       Ytrn=reshape(Ytrn,szY);          
    end    
	
    [seed,f,J]=feval(objFn,X,Ytrn,res.opt.C,'verb',opts.verb-1,...
                     'dim',opts.dim,'ydim',opts.ydim,varargin{:});
    if ( isstruct(seed) && isfield(seed,'soln') ) sol=seed.soln; else sol=seed; end;
    res.opt.soln{spi(1)}=sol;

    if( size(f,1)~=N ) % ensure predictions (i.e. f) is [N x nSp]
       szf=size(f);
       if ( szf(2)==N )                                    f=permute(f,[2 1 3:ndims(f)]);% f=f';
       elseif ( all(szf(1:numel(dim))==szY(1:numel(dim)))) f=reshape(f,[N szf(numel(dim)+1:end)]);
       elseif ( prod(szf(2:end))==N )                      f=f(:,:)'; 
       else error('f with shape [nSp x d1 x d2...] not supported yet');
       end
    end
    if ( opts.binsp ) % store f and tstf
       res.opt.f(:,spi)=f; 
       res.opt.tstf(exInd,spi)=f(exInd,spi);
    else 
       res.opt.f(:,1:size(f,2))=f; 
       res.opt.tstf(exInd,1:size(f,2))=f(exInd,:);
    end;
    % override tstf information for non-training data examples with those trained on all the data
    % now tstf contains val fold predictions in training set, and opt predictions for the rest
    if( opts.verb > -1 ) 
      fprintf('%3.2f/NA   ',feval(opts.lossFn,Ytrn,f)); 
    end
  end % spi
  if ( opts.verb>-1 ) fprintf('\n'); end;
elseif( opts.outerSoln==0 ) % don't compute outer solution at all...
  %fprintf('cvtrainMap: Not supported at the moment\n');
end
return;

%----------------------------------------------------------------------------
function testCase()

% test multi-regression performance
X = cumsum(randn(10,100,100));
A = mkSig(size(X,1),'exp',1); % strongest on 1st channel
Y = tprod(A,[-1],X,[-1 2 3]);

res=cvtrainMap('rrXY',X,Y,10.^[-3:3],10,'lossFn','est2corr')
res=cvtrainMap('rrXY',X,Y,10.^[-3:3],10,'lossFn','est2corr','groupdim',2)

% try with other mapping functions

%----------------------------------------------------------------------------
% perception prediction example
nSamp =100;
irflen=10;
nEpoch=100;
offset=0;
bias  =0;
nCh   =10;
irf   =mkSig(irflen,'gaus',irflen/3,.5)-mkSig(irflen,'gaus',irflen*2/3,.5)+.5; %DoG
irf   =irf./sum(abs(irf));
Y     =cumsum(randn([1,nSamp,nEpoch]),2); 
Y     =repop(Y,'-',mean(Y,2)); Y=repop(Y,'/',std(Y,[],2));% continuous output, 0-mean, unit-variance
xtrue=filter(irf(end:-1:1),1,Y,[],2); % convolve stimulus with irf
%clf;mcplot([Y(:,:,1); xtrue(:,:,1); Y(:,:,2); xtrue(:,:,2)]')
if ( nCh>1 ) % add a spatial dimension
  A  =mkSig(nCh,'exp',1)-.2; A=A./norm(A);
  X0 =tprod(A,[1 -1],xtrue,[-1 2 3]);
else % no space
  X0 =xtrue;
end
X  = X0 + randn(size(X0))*1e1 + offset;
taus=0:-1:-irflen;

Cs=[1 .1 .01 .001];
% pre-process by computing the time-shifted auto-correlations
XXY =cat(2,xytaucov(X,[],[],taus,'bias',bias),xytaucov(X,Y,[],taus,'bias',bias));
Cscale=covVarEst(XXY,[3 4]);
cvtrainMap('plsClsfr',XXY,Y,Cs*Cscale,10,'ydim',3,'lossFn','avedv','bias',bias,'clsfr',0)
cvtrainMap('wienerClsfr',XXY,Y,Cs*Cscale,10,'ydim',3,'lossFn','avedv','bias',bias,'clsfr',0)

% simple low-rank proto clsfr ~= CCA
XY  =xytaucov(X,Y,[],taus,'bias',bias);
cvtrainMap('svdProtoClsfr',XY,Y,[1 2 3],10,'ydim',3,'lossFn','avedv','bias',bias,'clsfr',0)


%----------------------------------------------------------------------------
% stim-sequence classification example
nSamp=100;
irflen=10;
isi   =3;
nEpoch=100;
offset=0;
bias  =0;
nCh   =10;
nCls  =3;
irf=mkSig(irflen,'gaus',irflen/2); irf=irf./sum(abs(irf));
y2s=randn(ceil(nSamp/isi),nCls)>.5; while( abs(diff(sum(y2s>0)))>5 ); y2s=randn(size(y2s,1),nCls)>.5; end;
tmp=zeros(nSamp,nCls);tmp(1:isi:end,:)=y2s;y2s=tmp;
xtrue=filter(irf(end:-1:1),1,y2s); % convolve stimulus with irf
%clf;mcplot([y2s(:,1) xtrue(:,1) y2s(:,2) xtrue(:,2)])
Yl =ceil(rand(nEpoch,1)*nCls);
[Y,key]=lab2ind(Yl);
if ( nCh>1 ) % add a spatial dimension
  A  =mkSig(nCh,'exp',1)-.2;
  X0 =tprod(A,[1 -1],shiftdim(xtrue(:,Yl),-1),[-1 2 3]);
else % no space
  X0 =shiftdim(xtrue(:,Yl),-1);
end
X  = X0+randn(size(X0))*5e-1+offset; % add noise
taus=0:-1:-irflen;

Cs=[1 .1 .01 .001]; % test re-seeding for increasing Cs

% pre-process by computing the time-shifted auto-correlations
XXY =cat(2,xytaucov(X,[],[],taus,'bias',bias),xytaucov(X,y2s',[],taus,'bias',bias));
Cscale=covVarEst(XXY,[3 4]);
cvtrainMap('plsClsfr',XXY,Y,Cscale*Cs,10,'ydim',1,'lossFn','est2loss','bias',bias,'clsfr',1)
cvtrainMap('wienerClsfr',XXY,Y,Cscale*Cs,10,'ydim',1,'lossFn','est2loss','bias',bias,'clsfr',1)

% simple low-rank proto clsfr ~= CCA
XY  =xytaucov(X,y2s',[],taus,'bias',bias);
cvtrainMap('svdProtoClsfr',XY,Y,[1 2 3],10,'ydim',1,'lossFn','est2loss','bias',bias,'labdim',2)
% now with a max-margin system
cvtrainMap('mmlr_cg',XY,Y,Cs,10,'ydim',1,'lossFn','est2loss','labdim',2);
% now with a max-margin + low-rank
cvtrainMap('mmlsigmalr_prox3',XY,Y,-[1 2 3],10,'ydim',1,'lossFn','est2loss','labdim',2,'verb',1);
