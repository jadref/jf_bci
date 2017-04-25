function [Wb,dv,J,obj]=factLR(X,Y,C,varargin)
% factored model LR classifier
%
% [Wb,dv,J,obj] = factLR(X,Y,C,...)
%
% Compute a low rank solution Logistic Regression solution to the input
% classification problem.  This solution is found using an "alternating
% directions" approach inspired by the ALS method used in PARAFAC
% for computing tensor decompositions.
%
% The learned decomposition of the weight vector has the form;
%   W_{i1,i2,...iN} = S^M U1_{i1,M} U2_{i2,M} ... UN_{iN,M}
%
% Where we use Einstein Summation Convention to indicate implicit
% summation over the repeated labels on the RHS.
%
% Inputs:
%  X  -- [n-d] data matrix
%  Y  -- [N x 1] set of +1/0/-1 class labels.  0 labels are test trials.
%  C  -- [1x1] regularisation parameter
%  U1 -- [size(X,1) x rank] starting value for X's 1st dimension
%  U2 -- [size(X,2) x rank] starting value for X's 2nd dimension
%  ...
%  UN -- [size(X,N) x rank] starting value for X's Nth dimension
% Options:
%  dim     -- [int] dimension of X which contains epochs
%  rank    -- the *maxium* rank of the approximation to compute, i.e. M, 
%             this should be >opt (rank) for this problem to guarantee convergence
%             to a global optimum.  But larger values slow down computation... Good 
%             default is .1*max(size(X)).                                        (1)
%  C       -- the regularisation strength
%  maxIter -- maximum number of iterations to perform
%  tol     -- [float] tolerance on change in U, to stop iteration
%  tol0    -- [float] tolerance on change in U relative to 1st step change (1e-3)
%  objTol  -- [float] convergence tolerance on the change in the objective value      (0)
%  objTol0 -- [float] convergence tolerance on the change in obj value, relative to   (1e-4)
%             initlal objective value. 
%  Wb      -- seed decomposed solution
%  initType -- {str} type of initialisation to use, 'clsfr' -- train classifier, otherwise -- diff of means
% Outputs:
%  Wb -- {S U1 U2 U3 ... U{ndims(X)} b} cell array of the decomposed solution and the bias term. 
%        Consiting of:
%               S-[rank x 1] set of (hyper)-singular values
%               Ui is [ size(X,i) x rank ]
%               b is the classifier bias value
%  dv -- [N x 1] vector of classifier decision values
%  J  -- the final objective function value
%  obj-- [J Ed Ew] bits of the final objective function value
opts=struct('rank',[],'verb',0,'maxIter',1e6,'maxEval',[],...
            'tol',0,'tol0',1e-5,'objTol',0,'objTol0',1e-4,'marate',.9,'alphab',[],'Wb',[],...
            'minVal',[],'minSeed',1e-6,'initNoise',0,'initType','proto','lineSearchAccel',0,...
            'dim',-1,'incThresh',.66,'fullOptInterval',5,...
            'minComp',1,'ortho',0,'scaleOpt',0,'h',1e-2,'innerObjFn','lr_cg');

[opts]=parseOpts(opts,varargin);

sizeX=size(X); nd=ndims(X);
dim = opts.dim; if ( isempty(dim) ) dim=-1; end; dim(dim<0)=dim(dim<0)+ndims(X)+1;
% default rank is 1/2 X's min size dim
if ( isempty(opts.rank) ) opts.rank = ceil(min(size(X))/2); end;

% Set the convergence thresholds for the inner L_2 regularised sub-problem
cgOpts=struct('maxEval',opts.maxEval,'tol',opts.tol,'tol0',opts.tol0,'objTol',opts.objTol,'objTol0',opts.objTol0);
innercgOpts=cgOpts; % faster inner tolerance
if( numel(opts.maxEval)>1 ) innercgOpts.maxEval=opts.maxEval(2); cgOpts.maxEval=opts.maxEval(1); end
if( numel(opts.tol)>1 )     innercgOpts.tol    =opts.tol(2);     cgOpts.tol    =opts.tol(1);     end
if( numel(opts.tol0)>1 )    innercgOpts.tol0   =opts.tol0(2);    cgOpts.tol0   =opts.tol0(1);    end
if( numel(opts.objTol)>1 )  innercgOpts.objTol =opts.objTol(2);  cgOpts.objTol =opts.objTol(1);  end
if( numel(opts.objTol0)>1 ) innercgOpts.objTol0=opts.objTol0(2); cgOpts.objTol0=opts.objTol0(1); end

% set the numerical threshold -- to detect when component is effectively 0
if ( isa(X,'single') ) eps=1e-6; else eps=1e-9; end;
if ( isempty(opts.minVal) ) opts.minVal=eps; end;

szX=size(X); szX(end+1:dim)=1; nd=numel(szX);
% get the outer-product dimensions, i.e. all but the trial dim
opDims=[1:dim-1 dim+1:ndims(X)];

% get the training set, and restrict to this sub-set of points if computationally efficient
incInd=any(Y~=0,2);
oX=[]; oY=[];
if ( sum(incInd)./size(Y,1) < opts.incThresh )
   % extract the training set
   idx={};for d=1:ndims(X); idx{d}=1:size(X,d); end; idx{dim}=incInd;
   oX=X; X = X(idx{:});
   oY=Y; Y = Y(incInd,:);
end

% Extract the provided seed values
if ( ~isempty(opts.alphab) && iscell(opts.alphab) && numel(opts.alphab)>1 )
  persistent warned;
  if ( opts.verb>0 && ~isequal(warned,1) )
	 warned=1; warning('Using alphab as seed, but Correct seed name is : Wb');
  end;
  opts.Wb=opts.alphab; opts.alphab=[];
end
U=cell(0); b=0; S=[];
if ( ~isempty(opts.Wb) ) % extract the seed from Wb
   if ( iscell(opts.Wb) ) % extract from cell array of params
      S=opts.Wb{1}; U(opDims)=opts.Wb(1+(1:ndims(X)-numel(dim))); b=opts.Wb{end};
   else
	  error('Dont know how to process seed solution');
   end
   if ( numel(S)>opts.rank ) % prune extra components
      S=S(1:opts.rank); for d=opDims; U{d}=U{d}(:,1:opts.rank);  end;
   end
end

% initialise the seed value
if ( isempty(U) ) 
   if ( strcmp(opts.initType,'clsfr') )   % seed with full-rank classifiers solution
      % N.B. spend more effort getting a good initial seed...
     wb=feval(opts.innerObjFn,X,Y,C,'dim',dim,'verb',opts.verb-1,cgOpts);
	  W=reshape(wb(1:end-1),[szX(1:dim-1) 1 szX(dim+1:end)]); b=wb(end);
   elseif ( strcmp(opts.initType,'clsfr1') )   % seed with full-rank classifiers solution
	  if ( opts.rank > 1 ) error('only valid if rank 1 solution wanted..'); end;
	  % N.B. spend more effort getting a good initial seed...
	  tX=X;
	  tX=msum(tX,opDims(2:end)); % cancel out other dims
     wb=feval(opts.innerObjFn,tX,Y,C,'dim',dim,'verb',opts.verb-1,cgOpts);
	  W=wb(1:end-1); b=wb(end);
	  if( numel(opDims)>1 )
		 % get right size
		 wsz=szX; wsz(dim)=1; wsz(opDims(2:end))=1;
		 W=reshape(wb(1:end-1),wsz);
		 % replicate fixed dims
		 repsz=ones(size(wsz));repsz(opDims(2:end))=szX(opDims(2:end));
		 W=repmat(W,repsz);
	  end
   else % seed with the prototype classifiers decomposition, i.e. diff btw means
      alpha=zeros(size(Y(:,1)));
      alpha(Y(:,1)>0)= .5./sum(Y(:,1)>0);
		alpha(Y(:,1)<0)=-.5./sum(Y(:,1)<0);
      W = tprod(X,[1:dim-1 -dim dim+1:ndims(X)],alpha,-dim,'n');
      dv= tprod(X,[-(1:dim-1) 1 -(dim+1:ndims(X))],W,[-(1:dim-1) 0 -(dim+1:ndims(X))]);
      b = -mean(dv);
		sd=max(1,sqrt(dv(:)'*dv(:)./numel(dv)-b*b));
		W = W./sd; b=b/sd;
   end
   [S,U{opDims}]=parafac_als(W,'rank',opts.rank,'maxIter',ndims(W)*10,'verb',opts.verb-2); % decompose 
   clear W;   
else % normalise the inputs to singular-value + direction form
   [S,U{:}]=parafacStd(S,U{:}); 
end


% This does 2 things:
%  a) it adds extra ranks if needed to make the seed as big as requested
%  b) it replaces components which are too small with random seed values
sml  = true(1,opts.rank); sml(abs(S)>max(S)*opts.minSeed)=false; % find el need to seed 
if ( any(sml) )
   for d=opDims; % add some randonmess to the all-0 components
      nVals= U{d}(:,sml)+randn(size(U{d},1),sum(sml)); % add some noise
      U{d}(:,sml)=repop(nVals,'./',sqrt(sum(nVals.^2,1))); % random direction
   end
   S(sml)=max(S)*opts.minSeed;
end

% Start the main alternating optimisation loop
J=inf; dU=inf;
for iter=1:opts.maxIter;
  oJ=J; oS=S; oU=U; ob=b; odU=dU;
   
   % get the set of active components
   actComp=(abs(S)>max(abs(S))*eps);
   if( sum(actComp) < opts.minComp )% force at least minComp comps to have useful magnitude
      [ans,si]=sort(abs(S),'descend'); 
      nactComp=false(size(actComp)); nactComp(si(1:opts.minComp))=true; nactComp(actComp)=false;
      S(nactComp)=S(nactComp)+eps;
      actComp=actComp | nactComp;
   end
   if ( ~any(actComp) ) warning('Ran out of active Components to fit!'); break; end; 
   
   % Loop over X's dimensions computing in individual optimal parameters
   XU = X;   % Temp store of: X^{1:N} \prod_{j < d} U_j^m
   dU=0;
   for di=1:numel(opDims);
      d =opDims(di);
      oUd=U{d};
      
      % Compute the full products, starting from the cached info
      tXU =XU;  % X^{1:N} \prod_{j neq d} U_j^m
      for di2=[di+1:numel(opDims)]; % up to epoch dim 
         d2  = opDims(di2);
         tXU = tprod(tXU,[1:d2-1 -d2 d2+1:nd nd+1:ndims(tXU)],U{d2}(:,actComp),[-d2 nd+1],'n'); 
      end
      
      % find optimal solution for this direction
		wbd  = repop(U{d}(:,actComp),'*',S(actComp)'); wbd=[wbd(:);b];
		% full-opt every so often
		if( mod(iter,opts.fullOptInterval)==1 ) innerObjOpts=cgOpts; else innerObjOpts=innercgOpts; end;
		[wb,f,J,obj]=feval(opts.innerObjFn,tXU,Y,C,'dim',dim,'wb',wbd,'verb',opts.verb-1,innerObjOpts);
      if ( any(isnan(wb(:))) ) error('NaN!!!!'); end;   

      % extract the weights again
      U{d}(:,actComp)= reshape(wb(1:end-1),szX(d),size(tXU,nd+1));
		b=wb(end);

		% compute the updated set of decision values
		dv = tprod(tXU,[1:d-1 -d d+1:nd -(nd+1)],U{d}(:,actComp),[-d -(nd+1)]);		
      clear tXU; % free up RAM - stop mem leaks
      
      % re-normalise the direction vectors
      nrms=sqrt(sum(U{d}.^2,1))'; ok=nrms>eps & ~isinf(nrms) & ~isnan(nrms);
      if( any(ok) ) 
         S(ok)=nrms(ok); 
         U{d}(:,ok)=repop(U{d}(:,ok),'./',nrms(ok)');  
      end;
      ok=(abs(S)>eps & ~isinf(S) & ~isnan(S)); 
      if( sum(ok)<opts.minComp ) 
        [ans,si]=sort(abs(S),'descend'); ok(si(1:opts.minComp))=true; 
      end;
      S(~ok)=0; % force to 0 if too small, until only minComp left

      if ( opts.verb > 1 )
        % compute the updated solution objective value
        fprintf('L* %2d.%2d)\tx=[%s]\tR=%2d\tJ=%.06f (%.06g+%.06g)\n',...
                iter,d,sprintf('%3f,',S(1:min(end,2))),sum(abs(S)>max(abs(S))*eps),J,obj(2),obj(3));
      end
                  
      dU = dU+abs(1-abs(sum(oUd.*U{d}))); % info on rate of change of bits
      % Update the cached info with this new value
      XU = tprod(XU,[1:d-1 -d d+1:nd nd+1:ndims(XU)],U{d}(:,actComp),[-d nd+1],'n'); 
      
      if ( any(isnan(U{d}(:))) || any(isnan(S)) ) error('NaN!!!!'); end;   
   end   
   dU = dU*abs(S);
   
   clear XU; % free up RAM - stop mem leaks
      
   % compute convergence measures
   if( J > oJ*(1+.1) ) 
      warning('Non-decrease!'); 
   end;
   if( iter==1 )      dJ0=J*2; maJ=J*2; dU0=dU*2; % get initial gradient est, for convergence tests
   elseif ( iter==2 ) dJ0=abs(oJ-J);  maJ=max(maJ,J); dU0=abs(dU-odU);
   elseif ( iter< 4 ) dJ0=max(dJ0,abs(oJ-J));  dU0=max(dU0,abs(dU-odU));
   end
   maJ =maJ*(1-opts.marate)+J(1)*(opts.marate); % move-ave obj est
   madJ=maJ-J;
   % status printout
   if ( opts.verb > 0 ) 
     fprintf('L* %2d)\tx=[%s]\tR=%2d\tJ=%.06f (%.06g+%.06g)\t|dJ|=%g %g\n',...
             iter,sprintf('%3f,',S(1:min(end,2))),sum(abs(S)>max(abs(S))*eps),J,obj(2),obj(3),abs(oJ-J),dU);
   end
   % convergence tests
   if( dU <= opts.tol || dU <= opts.tol0*dU0 || ...
       madJ <= opts.objTol || madJ <= opts.objTol0*dJ0 ) 
      break; 
   end;
end
% merge all parameters into a single output argument
Wb={S U{:} b};
% compute the final performance
if ( ~isempty(oX) ) % apply on all the data
  % accumulate away the factored weighting
  wX =oX;  
  for di=1:numel(opDims); 
    d  = opDims(di);
    wX = tprod(wX,[1:d-1 -d d+1:nd (nd+1):ndims(wX)],U{d},[-d nd+1]); 
  end
  wX = tprod(wX,[1:nd -(nd+1)],S,-(nd+1)); % scale and sum
  dv = wX+b;
end
if ( opts.verb >= 0 ) 
  fprintf('L* %2d)\tx=[%s]\tR=%2d\tJ=%.06f (%.06g+%.06g)\n',...
          iter,sprintf('%3f,',S(1:min(end,2))),sum(abs(S)>max(abs(S))*eps),J,obj(2),obj(3));
end
obj=[J obj];
return;
   
%------------------------------------------------------------------------------
function testCase()
%Make a Gaussian balls + outliers test case
nd=100; nClass=800;
[X,Y]=mkMultiClassTst([zeros(1,nd-1) -1 0 zeros(1,nd-1); zeros(1,nd-1) 1 0 zeros(1,nd-1); zeros(1,nd-1) .2 .5 zeros(1,nd-1)],[nClass nClass 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

% test this classifier, in 1-d version
[wb1 dv]=factLR(X,Y,1,'rank',1,'verb',2);

% in 2-d version
X2d=reshape(X,[size(X,1)/10 10 size(X,2)]);
[wb2 dv]=factLR(X2d,Y,1,'rank',2,'verb',2);

% in 3-d version
X3d=reshape(X,[size(X,1)/10 5 2 size(X,2)]);
[wb3 dv]=factLR(X3d,Y,1,'rank',2,'verb',2);
