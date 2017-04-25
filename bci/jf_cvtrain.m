function [z]=jf_cvtrain(z,varargin);
% code to train and test a classifier on the input object
% Options:
%  Y   -- [size(X,dim) x nSubProb x nFold] the set of labels to use.  (z.Y)
%  Ydi -- [dimInfo] struct describing the structure of Y
%  Cs  -- [1 x nCs] or [ nSets x nCs ] set of classifier reg params to try ([5^(3:-1:-3)])
%          N.B. Cs*Cscale is used as the actual classifier parameter
%  dim -- the dimension which contains the epochs, deduced from Ydi if not given ([])
%  foldIdxs -- [size(X,dim) x nFold] set of -1/0/+1 fold indicators         (z.foldIdxs)
%              OR
%              [1x1] number for temporally ordered and class-stratified folds to do (10)
%  Cscale-- [1x1] scaling factor for the Cs, (1*var(z.X))
%              OR 'l1' for l1 loss scaling, or 'l2' for l2 scaling
%  objFn--[str] or [function_handle] inner function to call to do the actual     ([])
%         classification.  Function template:  [w,f,J]=obj_fn(X,Y,C,...)
%  cvtrainFn -- [str] or [function_handle] inner function to call do the the actual cvtraining  (cvtrainFn)
%            Function template: [res,Cs,fIdxs]=cvtrainFn(objFn,X,Y,Cs,foldIdxs,...)
%            See: cvtrainFn and cv2trainFn for options
%  binsp  -- [bool] training a set of binary sub-problems?                        (1)
%  mcPerf -- [bool] compute multi-class performance from sets of bin subproblems? (0)
%
% See Also: cvtrainFn, cv2trainFn
opts=struct('dim',[],'Y',[],'Ydi',[],'Yl',[],'Cs',[],'Cscale',[],'CscaleThresh',5,'CscaleAll',0,...
            'objFn',[],'cv2',0,'cvtrainFn','cvtrainFn','foldIdxs',[],'recSoln',1,...
				'mcdecoder',[],'binsp',1,'mcPerf',0,...
            'subIdx',[],'compressTrDimX',0);
[opts,varargin]=parseOpts(opts,varargin);
if ( ~isempty(opts.subIdx) ) 
   persistent warnedsubidx
   if (isempty(warnedsubidx) ) 
      warning('subIdx given ... foldIdxs overrides!');
      warnedsubidx=true;
   end
end

dim=n2d(z,opts.dim);

Y=opts.Y;     
if( isempty(Y) )
   if ( isfield(z,'Y') )            Y=z.Y; 
   elseif ( ~isempty(dim) && isfield(z.di(dim).extra,'marker') )  Y=[z.di(dim).extra.marker];
   else warning('Couldnt find the Y labels');
   end
end; 
if ( isempty(opts.objFn) ) 
  opts.objFn='lr_cg';
  % switch to kernel version if kernel input
  if ( isfield(z,'prep') && strcmp(z.prep(end).method,'jf_compKernel') ) opts.objFn='klr_cg';  end
end

% convert label lists to sub-problems if wanted
spDesc='';
Ydi=opts.Ydi;
if( isempty(Ydi) ) 
   if ( isfield(z,'Ydi') ) % use to infer the sub-problem description info
      Ydi   = z.Ydi; 
   else 
      Ydi=mkDimInfo(size(Y),'epoch',[],[],'subProb',[],[],'lab'); 
      for i=1:size(Y,2); spDesc{i}=sprintf('%d',i); end;
   end
end

% use Ydi to set the dim if not already set
if ( isempty(dim) && ~isempty(Ydi) ) 
   dim = n2d(z.di,{Ydi(1:end-1).name},1,0); 
	if(~any(dim)) % relax and try prefix match
	  warning('Couldnt get an exact match for the trial dimension');
	  dim = n2d(z.di,{Ydi(1:end-1).name},0,0); 
	end
	dim(dim==0)=[]; % get dim to work along
	if ( isempty(dim) ) warning('Couldnt work out the trial dimension!...'); end;
   if ( numel(dim)>1 && all(diff(dim)~=1) ) error('only for conseq dims'); end;   
end
if( isempty(spDesc) && ~isempty(Ydi) ) % get descprition from Y info
   spD=setdiff(1:numel(z.Ydi)-1,dim);   spDesc= z.Ydi(spD).vals;
else
   spDesc='1';
end


foldIdxs=opts.foldIdxs;
if ( isempty(opts.foldIdxs) ) 
   if ( isfield(z,'foldIdxs') ) foldIdxs=z.foldIdxs; 
   else
      foldIdxs=[];
   end
end

szX=size(z.X);     szY=size(Y); 
ndim=dim; % single dim version for the cvtrainFn call
if (  numel(dim)>1 ) % convert to 1d format for training
  if ( opts.compressTrDimX ) % compress trial dims in X
	 nszX=[szX([1:min(dim)-1]) prod(szX(dim)) szX(max(dim)+1:end)];
    if ( ndims(z.X)>=numel(dim)*2 && ... % BODGE: kernel/multi-kernel as a special case
			all(szX(1:numel(dim))==szX(numel(dim)+(1:numel(dim)))) && ...
			isequal(z.di(numel(dim)+1).name(max(1,end-3):end),'_ker') ) % N.B. should test all for kernelness
      nszX=[prod(szX(dim)) prod(szX(dim)) szX(2*numel(dim)+1:end) 1];
    end
    z.X=reshape(z.X,nszX);
    ndim=min(dim);
    nszY=[prod(szY(1:numel(dim))) szY(numel(dim)+1:end)]; Y=reshape(Y,[nszY 1]);
	end
   if ( ~isempty(foldIdxs) )
    szY=size(Y);
    szfoldIdxs=size(foldIdxs); % and foldIdxs if necessary
%     if ( any(szfoldIdxs(1:numel(dim))==1) && any(szfoldIdxs(1:numel(dim))~=szY(1:numel(dim))) )
%           % scale up foldIdxs to Y size if necess
%       foldIdxs=repmat(foldIdxs,[szY(1:numel(dim))./szfoldIdxs(1:numel(dim))...
%                           ones(1,ndims(foldIdxs)-numel(dim))]);
%       szfoldIdxs=size(foldIdxs); % new size
%     end
%     % reshape foldIdxs as necessary
%     foldIdxs=reshape(foldIdxs,[prod(szfoldIdxs(1:numel(dim))) szfoldIdxs(numel(dim)+1:end) 1]);
   end
end

if( ~isempty(foldIdxs) ) trnIdx=any(foldIdxs<0,ndims(foldIdxs)); else trnIdx=[];end;
Cscale=opts.Cscale;
Cs=opts.Cs; 
if ( isempty(Cscale) || isequal(Cscale,'l2') || isequal(Cscale,'l1') )      
  if ( isempty(trnIdx) || all(trnIdx(:)) || numel(ndim)>1 || opts.CscaleAll ) 
	 if ( numel(ndim)>1 )
       persistent warnedtrntst
		if ( ~isequal(warnedtrntst,true) ) warnedtrntst=true; 
		  warning('train/test split ignored when computing Cscale and using multi-trial dims!');
		end
	 end;	 
    if ( opts.CscaleAll ) fprintf('train/test split ignored when CscaleAll is set! '); end;
    Cscale=CscaleEst(z.X,ndim);
    if ( opts.CscaleAll ) fprintf('Cscale = %g\n',Cscale); end;
  else
	 if ( numel(ndim)>1 ) error('multi-trial dims not supported yet!'); end;
    idx={}; for d=1:ndims(z.X); idx{d}=1:size(z.X,d); end; idx{ndim}=trnIdx; 
    if ( strcmp(z.prep(end).method,'jf_compKernel') ) idx{ndim}=trnIdx; end; % kernel method
    Cscale=CscaleEst(z.X(idx{:}),ndim);
  end
  if ( isequal(opts.Cscale,'l1') ) 
    Cscale=sqrt(Cscale); 
    if (isempty(Cs) ) Cs=[3.^(3:-1:-3)]; end; % smaller range of Cs for l1 regs also
  end
elseif ( isequal(Cscale,'covScale') )
	Cscale=covVarEst(z.X,3:ndims(z.X));
   if (isempty(Cs) ) Cs=[10.^(1:-1:-5)]; end; % smaller range of Cs for cov regs also
end
if ( isempty(Cs) ) 
   Cs=[5.^(3:-1:-3)]; 
elseif ( ischar(Cs) )
   switch lower(Cs(:)');
      case 'optshrinkage';
        if ( ~isempty(strfind(lower(z.prep(end).method),'kernel')) ) % kernel input
           % extract info on the size of the orginal features
           odi=z.prep(end).info.odi;
           szX=[];for i=1:numel(odi)-1; szX(i)=numel(odi(i).vals); end; 
           featSz=szX(setdiff(1:end,n2d(odi,{z.Ydi(1:end-2).name})));
           [lambda,Cs]=optShrinkageK(z.X(foldIdxs(:,1)<0,foldIdxs(:,1)<0),prod(featSz));
        else % features input
           [lambda,Sigma]=optShrinkage(z.X(:,:,foldIdxs(:,1)<0),setdiff(1:ndims(z.X),n2d(z,{z.Ydi(1:end-2).name})));
           Cs = lambda./(1-lambda)*mean(diag(Sigma));
        end
        Cscale=1;
      otherwise; error('Unrecognised hyperparameter selection method: %s',Cs);
   end
end;
% BODGE: convert col of Cs input into a row
if ( size(Cs,2)==1 && size(Cs,1)>2 ) Cs=Cs(:)'; end;


% call cvtrain to do the actual work
if ( opts.cv2 ) % double nested, call the cv2train function directly
	[res,Cs,fIdxs]=cv2trainFn(opts.objFn,z.X,Y,Cscale*Cs,foldIdxs,'cvtrainFn',opts.cvtrainFn,'dim',ndim,'spDesc',spDesc,'recSoln',opts.recSoln,'binsp',opts.binsp,varargin{:});
else
  [res,Cs,fIdxs]=feval(opts.cvtrainFn,opts.objFn,z.X,Y,Cscale*Cs,foldIdxs,'dim',ndim,'spDesc',spDesc,'recSoln',opts.recSoln,'binsp',opts.binsp,varargin{:});
end
z.X = [];
ozdi= z.di;
if ( isfield(res,'opt') && isfield(res.opt,'tstf') ) 
  z.X = res.opt.tstf;
  ydim= n2d(Ydi,{ozdi(dim).name},1,0); spD=setdiff(1:ndims(z.Y),ydim);
  z.X = reshape(z.X,[szX(dim),szY(spD)]);
  z.di= [ozdi(dim);Ydi(spD);mkDimInfo(1,1,'pred')];
elseif ( isfield(res,'fold') && isfield(res.fold,'f') )
  z.X = res.fold.f;
  z.di= res.fold.di;
  z.di(end).label='pred';
else
  z.X=[];
  warning('Nothing to store for the results!');
end
if ( numel(dim)>1 ) % convert results back to n-d format
  rszX=size(z.X);
  if( rszX(1)==prod(szX(dim)) )
   z.X = reshape(z.X,[szX(dim),rszX(2:end)]);
   z.di= [ozdi(dim);z.di(2:end)]; % rec dim-info
  elseif( rszX(end)==prod(szX(dim)) )
   z.X = reshape(z.X,[rszX(1:end-1) szX(dim)]);
   z.di= [z.di(1:end-2);ozdi(dim)]; % rec dim-info
  end
   if ( isfield(res,'fIdxs') && size(res.fIdxs,1)==prod(szX(dim)) ) % folding info
     rszf = size(res.fIdxs);
     res.fIdxs = reshape(res.fIdxs,[szX(dim) rszf(2:end)]);
   end
else
   z.di(1)=ozdi(dim(1)); % over-ride epoch info with true epoch info
end
if( isempty(foldIdxs) || isscalar(foldIdxs) ) % record the folding used
   foldIdxs=fIdxs;
   if ( isfield(z,'foldIdxs') && isempty(z.foldIdxs) ) z.foldIdxs=fIdxs; end
end
   
% build mc-perf info
% multi-class results recording
if ( opts.mcPerf && opts.binsp && ~isempty(Ydi) && isfield(Ydi(n2d(Ydi,'subProb')).info,'spMx') && ...
     (~isfield(Ydi(n2d(Ydi,'subProb')).info,'mcSp') || Ydi(n2d(Ydi,'subProb')).info.mcSp) ) 
  if ( exist('szfoldIdxs') ) foldIdxs=reshape(foldIdxs,szfoldIdxs); end;
  spMx=Ydi(n2d(Ydi,'subProb')).info.spMx;
  spKey=Ydi(n2d(Ydi,'subProb')).info.spKey;
  if ( isfield(Ydi(n2d(Ydi,'subProb')).info,'spD') ) spD=Ydi(n2d(Ydi,'subProb')).info.spD; 
  else                                               spD=ndims(Y); 
  end;
  fldD=n2d(z,'fold');
  spD =n2d(z,spD);
  trD =setdiff(1:max(spD,ndims(Y)),spD); % BODGE: estimate position of the trials
  if ( ~isempty(spMx) && (numel(Ydi(n2d(Ydi,'subProb')).vals)>1 || numel(spD)>1) )
    % get back the multi-class labels by decoding the sub-prob spec
    Yl   = dv2pred(z.Y,n2d(z.Ydi,{z.di(spD).name}),spMx,'ml',1); % [spD(1) rest of Y] or [rest of Y (spD(1))]    
    if ( any(spD==1) ) % Y is nClass x N - permute into order cvmcPerf expects, i.e. spD at end
      Yl = permute(Yl,[setdiff(1:ndims(Yl),spD); spD]);
    end
    % ensure ignored points are ignored
    incIdx=any(z.Y~=0,spD(1));for d=2:numel(spD); incIdx=all(incIdx~=0,spD(d)); end; Yl(~incIdx,:,:,:)=0;
    mcres= cvmcPerf(Yl,z.X,[trD spD(:)' fldD],foldIdxs,spMx,spKey,opts.mcdecoder);
    res  = mergeStruct(res,mcres);
	 res.trn = mcres.trncr; res.tst=mcres.tstcr; % multi-class overrides binaries for output
    fprintf('-------------------------\n(ave/mc)\t');
    for ci=1:size(mcres.trncr,2);
      fprintf('%0.2f/%0.2f\t',mcres.trncr(:,ci),mcres.tstcr(:,ci));
    end
    fprintf('\n');
  end
end

compOuterPerf=false;
if ( isfield(z,'outfIdxs') )
  exInd = all(all(z.outfIdxs>=0,3),2); % trials which are only ever excluded points or testing points
  if ( any(exInd(:)) )
	 compOuterPerf=any(any(z.Y(exInd,:)~=0,2),1);
  else
	 compOuterPerf=false;
  end
end
if ( compOuterPerf ) % compute outer fold performance info, if are outerfold examples
  if ( numel(dim)==1 ) 
	 outres = cvPerf(z.Y,res.opt.tstf,[1 2 3],z.outfIdxs,[],opts.binsp);
  else % N.B. assumes Y has trials in starting dimensions
	 tY=reshape(z.Y,[prod(szX(dim)),size(z.Y,numel(dim)+1)]);
	 tf=reshape(z.outfIdxs,[prod(szX(dim)),size(z.outfIdxs,numel(dim)+1)]);
	 outres = cvPerf(tY,res.opt.tstf,[1 2 3],tf,[],opts.binsp);
  end
  % print the resulting test set performances
  for spi=1:size(outres.trn,n2d(outres.di,'subProb')); % loop over sub-problems
	 if ( any(z.outfIdxs(:))>0 ) % only report tst if there are actually test examples
		fprintf('(out/%2d*)\t%.2f/%.2f\n',spi,outres.trn(:,spi),outres.tst(:,spi));
	 else
		fprintf('(out/%2d*)\t%.2f/NA \n',spi,outres.trn(:,spi),outres.tst(:,spi));
	 end
  end
  % if given outer fold info overrides
  outres.inner=res;
  res=outres;
  res.opt=res.inner.opt;
end

% build a summary structure
summary=sprintf('over [%s]s with %s',...
                [sprintf('%s+',z.di(1:numel(dim)-1).name) z.di(numel(dim)).name],opts.objFn);
info=struct('Cscale',Cscale,'Y',Y,'res',res,'odi',ozdi);
if ( isfield(res,'opt') && isfield(res.opt,'soln') ) 
   info.testFn={'jf_cvtest' 'classifier' res.opt.soln 'objFn' opts.objFn 'dim',dim};
else
   info.testFn={'error' 'test function not defined!'};
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
z = jf_mksfToy('Y',ceil(rand(100,1)*2));
c = jf_cvtrain(z)
kz= jf_compKernel(z);
c = jf_cvtrain(kz)

% test with double nested CV
c2 = jf_cvtrain(kz,'cvtrainFn',@cv2trainFn)
c2 = jf_cvtrain(kz,'cvtrainFn',@cv2trainFn,'innerFold',5)

% test effect of implicitly ignored points
z.foldIdxs(end-10:end,:)=0;
c0=jf_cvtrain(z);
c1=jf_cvtrain(jf_retain(z,'dim','epoch','idx',any(z.foldIdxs~=0,2)));
mad(c0.prep(end).info.res.opt.soln{:},c1.prep(end).info.res.opt.soln{:})
clf;plot([c0.prep(end).info.res.opt.soln{:} c1.prep(end).info.res.opt.soln{:}]);

w = jf_welchpsd(z);
kw= jf_compKernel(w);
jf_cvtrain(kw);

