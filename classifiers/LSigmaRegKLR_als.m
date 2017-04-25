function [Wb,dv,J,obj]=LSigmaRegKLR_als(X,Y,C,varargin)
% Compute the "Parallel Factors" or Canonical Decomposition of an n-d array
%
% [Wb,dv,J,obj] = LSigmaRegKLR_als(X,Y,C,...)
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
%  symDim  -- [2 x ...] pairs of dimensions for which solution should be symetric ([]) 
%  maxIter -- maximum number of iterations to perform
%  tol     -- [float] tolerance on change in U, to stop iteration
%  tol0    -- [float] tolerance on change in U relative to 1st step change (1e-3)
%  objTol  -- [float] convergence tolerance on the change in the objective value      (0)
%  objTol0 -- [float] convergence tolerance on the change in obj value, relative to   (1e-4)
%             initlal objective value. 
%  ortho   -- [bool] orthogonalise solution during/after iteration?           (1)
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
opts=struct('rank',[],'C',0,'Cscale',[],'symDim',[],'verb',0,'maxIter',1e6,'maxEval',[],...
            'tol',0,'tol0',1e-5,'objTol',0,'objTol0',1e-5,'marate',.7,'alphab',[],'Wb',[],...
            'minVal',[],'minSeed',1e-6,'initNoise',0,'initType','clsfr','lineSearchAccel',1,...
            'dim',-1,...
            'minComp',1,'ortho',1,'scaleOpt',0,'h',1e-2,'innerObjFn','klr_cg');

[opts]=parseOpts(opts,varargin);

sizeX=size(X); nd=ndims(X);
dim = opts.dim; if ( isempty(dim) ) dim=-1; end; dim(dim<0)=dim(dim<0)+ndims(X)+1;
% default rank is 1/2 X's min size dim
if ( isempty(opts.rank) ) opts.rank = ceil(min(size(X))/2); end;

% Set the convergence thresholds for the inner L_2 regularised sub-problem
cgOpts=struct('maxEval',size(X,dim),'tol',0,'tol0',0,'objTol',0,'objTol0',5e-2);
if( numel(opts.maxEval)>1 ) cgOpts.maxEval=opts.maxEval(2); opts.maxEval=opts.maxEval(1); end
if( numel(opts.tol)>1 )     cgOpts.tol    =opts.tol(2);     opts.tol    =opts.tol(1);     end
if( numel(opts.tol0)>1 )    cgOpts.tol0   =opts.tol0(2);    opts.tol0   =opts.tol0(1);    end
if( numel(opts.objTol)>1 )  cgOpts.objTol =opts.objTol(2);  opts.objTol =opts.objTol(1);  end
if( numel(opts.objTol0)>1 ) cgOpts.objTol0=opts.objTol0(2); opts.objTol0=opts.objTol0(1); end


% set the numerical threshold -- to detect when component is effectively 0
if ( isa(X,'single') ) eps=1e-6; else eps=1e-9; end;
if ( isempty(opts.minVal) ) opts.minVal=eps; end;

% get the outer-product dimensions, i.e. all but the trial dim
opDims=[1:dim-1 dim+1:ndims(X)];

sym=zeros(ndims(X),1); % indicator of pairs of symetric dimensions
if ( ~isempty(opts.symDim) ) 
   for i=1:size(opts.symDim,2); 
      if ( sym(opts.symDim(1,i))~=0 || sym(opts.symDim(2,i))~=0 ) error('double symDim not allowed'); end;
      sym(opts.symDim(1,i))=opts.symDim(2,i); sym(opts.symDim(2,i))=opts.symDim(1,i); 
   end
end

% get the training set, and restrict to this sub-set of points if computationally efficient
trnInd =(Y~=0); N=sum(trnInd);
if ( any(~trnInd) )
   % extract the training set
   szX=size(X);
   idx={};for d=1:ndims(X); idx{d}=1:szX(d); end;
   trnIdx=idx; trnIdx{dim}=trnInd;
   Xtrn = X(trnIdx{:});
   Ytrn = Y(trnInd);
else
   Xtrn=X; Ytrn=Y;
end

Cscale=opts.Cscale;
if ( isempty(Cscale) ) % est for the size of S
  Cscale=sqrt(X(:)'*X(:))./opts.rank; 
end
if ( ~isempty(Cscale) ) 
  Xtrn  =Xtrn./Cscale; % N.B. X isn't shrunk...
end;

% Extract the provided seed values
if ( ~isempty(opts.alphab) && iscell(opts.alphab) && numel(opts.alphab)>1 )
  if ( opts.verb>0 ) warning('Using alphab as seed, but Correct seed name is : Wb'); end;
  opts.Wb=opts.alphab; opts.alphab=[];
end
U=cell(0); b=0; S=[];
if ( ~isempty(opts.Wb) ) % extract the seed from Wb
   if ( iscell(opts.Wb) ) % extract from cell array of params
      S=opts.Wb{1}; U(opDims)=opts.Wb(1+(1:ndims(X)-numel(dim))); b=opts.Wb{end};
   elseif ( isnumeric(opts.Wb) ) % work it out by counting elements
      if ( isempty(opts.rank) ) opts.rank = (numel(opts.Wb)-1)./prod(sizeX(1:end-1)); end;
      nf=0;
      for d=1:ndims(X); 
         U{d}=reshape(opts.Wb(nf+1:(sizeX(d)*opts.rank)),[sizeX(d) opts.rank]);
         nf=nf+numel(U{d});
      end
      b = opts.Wb(end);
   end
   if ( numel(S)>opts.rank ) % prune extra components
      S=S(1:opts.rank); for d=opDims; U{d}=U{d}(:,1:opts.rank);  end;
   end
end

% initialise the seed value
if ( isempty(U) ) 
   if ( 0 && strcmp(opts.initType,'clsfr') )   % seed with full-rank classifiers solution
      K=compKernel(Xtrn,[],'linear','dim',dim); 
      % N.B. spend more effort getting a good initial seed...
      alphab=klr_cg(K,Ytrn,sqrt(C),'verb',opts.verb-1,cgOpts,'objTol0',cgOpts.objTol0*1e-2); 
      clear K
      W= tprod(Xtrn,[1:dim-1 -dim dim+1:ndims(Xtrn)],alphab(1:end-1),-dim,'n'); b=alphab(end);
   else % seed with the prototype classifiers decomposition, i.e. diff btw means
      alpha=zeros(size(Ytrn(:,1)));
      alpha(Ytrn(:,1)>0)=.5./sum(Ytrn(:,1)>0); alpha(Ytrn(:,1)<0)=-.5./sum(Ytrn(:,1)<0);% equal class weights
      W = tprod(Xtrn,[1:dim-1 -dim dim+1:ndims(Xtrn)],alpha,-dim,'n');
      dv=tprod(Xtrn,[-(1:dim-1) 1 -(dim+1:ndims(Xtrn))],W,[-(1:dim-1) 0 -(dim+1:ndims(Xtrn))]);
      b = -(dv(Ytrn<0)'*abs(alpha(Ytrn<0))*sum(Ytrn>0)  + dv(Ytrn>0)'*abs(alpha(Ytrn>0))*sum(Ytrn<0))./sum(Ytrn~=0); % weighted mean
      scale   = min(10,N./max(eps,C(1)))./(mean(abs(dv+b))); % unit norm + C scaling
      W=W*scale; alpha=alpha*scale; b=b*scale;
   end
   if ( strcmp(opts.initType,'clsfr') )   % seed with full-rank classifiers solution
     K=compKernel(Xtrn,[],'linear','dim',dim); 
     l2=alpha(:)'*K*alpha(:); % summed l2 norm
     % N.B. spend more effort getting a good initial seed...
     % use an approx to the L1 loss based on re-scaling the L2 loss
     alphab=klr_cg(K,Ytrn,C/2/sqrt(l2),'alphab',[alpha;b],'verb',opts.verb-1,cgOpts,'objTol0',cgOpts.objTol0*1e-2);
     clear K
     W= tprod(Xtrn,[1:dim-1 -dim dim+1:ndims(Xtrn)],alphab(1:end-1),-dim,'n'); b=alphab(end);
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

% add some noise the the seed values for convergence stuff -- if wanted
if ( ~isempty(opts.initNoise) && opts.initNoise~=0 )
   for d=opDims; U{d}=U{d}+randn(size(U{d}))*opts.initNoise; end;
end

% re-orthogonalise the solution
if ( opts.ortho>1 )         [S,U{:}]=parafacOrtho(S,U{:},'maxIter',3); 
elseif ( opts.ortho>0 )     S=parafacCompress(S,U{:});
end;

% ensure seed is symetric if wanted
if ( ~isempty(opts.symDim) ) [S,U{:}]=symetricParafac(opts.symDim,S,U{:}); end      

% pre-optimise the scale of the components
if ( opts.scaleOpt>0 )
   actComp=true(size(S));
   h  = opts.h*max(abs(S));
   XU = Xtrn;
   for d=1:numel(opDims); 
      XU = tprod(XU,[1:d-1 -d d+1:nd nd+1:ndims(XU)],U{d}(:,actComp),[-d nd+1],'n'); 
   end
   if ( opts.verb>0 ) fprintf('\n'); end;
   Sb = nonLinConjGrad(@(w) primalL1RegLRFn(w,squeeze(XU)',Ytrn,C,h),[S(actComp);b],...
                       'maxEval',cgOpts.maxEval,'tol',cgOpts.tol,'verb',opts.verb-1);
   S(actComp)  = Sb(1:end-1); b=Sb(end);
elseif ( opts.scaleOpt<0 ) 
  S=S*(1 + -opts.scaleOpt); % inc scale to improve conv
end

% compute the initial performance
[dv J Ed Ew]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,S/Cscale,U,b);
if ( opts.verb >= 0 ) 
   fprintf('\nL* %2d)\tEw=(%s)=%.06g\tR=%d\tEd=%.06g\tJ=%.06g\tdJ=%.06g\tdU=%.06g\n',...
           0,sprintf('%3f,',S(1:min(end,2))),Ew,sum(abs(S)>max(S)*eps),Ed,J,inf,inf);
end

% Start the main alternating optimisation loop
tJ=repmat(inf,[numel(U),1]); dU=inf; 
K=zeros(size(Xtrn,dim)+1,size(Xtrn,dim)+1,class(X));
oU=U; oS=S; ob=b; oJ=J;
for iter=1:opts.maxIter;
   ooJ=oJ; oJ=J; 
   odU=dU;  
   ooS=oS; oS=S; 
   ooU=oU; oU=U; 
   oob=ob; ob=b;
   
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
   XU = Xtrn;   % Temp store of: X^{1:N} \prod_{j < d} U_j^m
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

      % add in the scaling information
      % N.B. we use the sqrt of the S so this amounts to doing a local gradient fit for the
      %  abs scale component in the optimisation, c.f. L1RegKLR, hence we impose a regularisation
      %  on the sum of the singular values!
      lambda= sqrt(abs(S(actComp))); % scaling for all the fixed components
      if ( opts.h>0 ) lambda(abs(S(actComp))<opts.h) = sqrt(opts.h); end; % huberize, if S is too small
      %scale current solution by 1./sqrt(2*S), so S^2/lambda ~ |S|/2 = S^2./(sqrt(2*S))^2 & d(S^2)=2S/lambda~=1
      tUd   =repop(U{d}(:,actComp),'*',(S(actComp)./lambda(:))');  
      tXU   =repop(tXU,'*',shiftdim(lambda(:),-nd));               %scale data by inverse to maintain obj-value
      
      % ALS to find the new values 
      K(1:end-1,1:end-1)= compKernel(tXU,[],'linear','dim',dim); % new kernel, N.B. tXU=[size(X) x rank]
      % Xtrn*lambda * W/lambda = Xtrn * W
      Ka=compKernel(tXU,reshape(tUd,[ones(1,d-1) size(U{d},1) ones(1,nd-d) size(U{d}(:,actComp),2)]),...
                    'linear','dim',dim); % add virtual pt for the soln
      K(1:end-1,end) = Ka; K(end,1:end-1)= Ka;
      Kaa= compKernel(tUd,[],'linear','dim',3); % R_1(W) = sum(S)
      K(end,end) = Kaa;
      alphab = zeros(size(K,1)+1,1); alphab(end-1)=1; alphab(end)=b;
      %L2 objective for this set of factors          
      % N.B. C./2 to compensate for dw^2=2*dw
      %  As for L1RegKLR we do it this way rather than in the B so the reported numbers 
      %  are the same as for the L1 loss
      [alphab,tf,ttJ,tobj]=klr_cg(K,[Ytrn;0],C./2,'alphab',alphab,'verb',opts.verb-1,cgOpts);
      if ( any(isnan(alphab(:))) ) error('NaN!!!!'); end;   

      % extract the weights again
      tidx=zeros(1,ndims(X)+1); tidx(d)=1; tidx(dim)=-dim; tidx(end)=2; % build dimspec
      Ud2 = tprod(tXU,tidx,alphab(1:end-2),-dim); % the non-seed part
      Ud2 = repop(Ud2,'*',lambda(:)');            % undo feature space re-scaling
      Ud2 = Ud2 + repop(S(actComp)','*',U{d}(:,actComp)*alphab(end-1)); % include the weighted copy of the seed
      U{d}(:,actComp)= Ud2;
      
      if ( any(isnan(U{d}(:))) ) error('NaN!!!!'); end;   
      % extract the bias
      b=alphab(end);
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
        [dv2 J2 Ed2 Ew2]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,S./Cscale,U,b);
        fprintf('L* %2d.%2d)\tEw=(%s)=%.06g\tR=%d\tEd=%.06g\tJ=%.06g\t\n',...
                iter,d,sprintf('%3f,',S(1:min(end,2))),Ew2,sum(abs(S)>max(abs(S))*eps),Ed2,J2);        
      end
      
      % limit the minimal value to stop the 0 degeneracy.+ add some noise
      if ( ~isempty(opts.minVal) && opts.minVal~=0 ) 
        sml=abs(S)<opts.minVal & ok; 
        if ( any(sml) ) S(sml)=S(sml)+rand(sum(sml),1)*opts.minVal; end;
      end;         
             
      dU = dU+abs(1-abs(sum(oUd.*U{d}))); % info on rate of change of bits
      % Update the cached info with this new value
      XU = tprod(XU,[1:d-1 -d d+1:nd nd+1:ndims(XU)],U{d}(:,actComp),[-d nd+1],'n'); 
      
      if ( any(isnan(U{d}(:))) || any(isnan(S)) ) error('NaN!!!!'); end;   
   end   
   dU = dU*abs(S);
   
   % compute the updated optimal scaling parameters
   if ( opts.scaleOpt>1 )
      h  = opts.h*max(abs(S));
      Sb = nonLinConjGrad(@(w) primalL1RegLRFn(w,squeeze(XU)',Ytrn,C,h),[S(actComp);b],...
                          'maxEval',cgOpts.maxIter,'tol',cgOpts.tol,'verb',opts.verb-1);
      S(actComp)  = Sb(1:end-1); b=Sb(end);
   end
   clear XU; % free up RAM - stop mem leaks
   
   % compute the updated solution objective value
   [dv J Ed Ew]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,S./Cscale,U,b);

   % Do line search acceleration!
   if ( opts.lineSearchAccel && iter>3 )
     J_in=J;
     if( opts.verb > 1 ) 
       fprintf('%2d)\tJ=%8f\tdJ=%8f\tdU=%8f\n',iter,J,oJ-J,dU);
     end
     step=2.2;%(iter+1)^(1./acc_pow); % fixed prob step size
     dirS=S-ooS; for d=opDims; dirU{d}=U{d}-ooU{d}; end; dirb=b-oob;
     tS=ooS+dirS*step; for d=opDims; tU{d}=ooU{d}+dirU{d}*step;end; tb=oob+dirb*step;
     [tS,tU{:}]=parafacStd(tS,tU{:}); % re-normalise
     [tdv tJ tEd tEw]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,tS./Cscale,tU,tb);
     if ( opts.verb > 1 ) 
       fprintf('ACC: step=%g \toJ=%g \tJ=%g \tdJ=%g',step,J_in,tJ,J_in-tJ);     
     end
     if ( tJ<J_in ) % accept good accel       
       U=tU; S=tS; b=tb; J=tJ; Ed=tEd; Ew=tEw; dv=tdv;
       if( opts.verb>1 ) fprintf('\t success\n'); end;
     else % do nowt
       if( opts.verb>1 ) fprintf('\t failed\n'); end;
     end
     % do 2nd secant based line search
     [ss,fss]=secantFit([0 1 step],[ooJ J_in tJ]);
     if ( ss>1.2 && (ss<step-.2 || ss>step+.2) && ss<2*step ) 
       tS=ooS+dirS*ss; for d=opDims; tU{d}=ooU{d}+dirU{d}*ss; end; b=oob+dirb*ss;
       [tS,tU{:}]=parafacStd(tS,tU{:}); % re-normalise
       [tdv2 tJ2 tEd2 tEw2]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,tS./Cscale,tU,tb);
       if ( opts.verb > 1 ) 
         fprintf('ACC: s:%g=%g\t s:%g=%g\t s:%g=%g\t s:%g=%g',0,ooJ,1,J_in,step,tJ,ss,tJ2);
       end     
       if ( tJ2<J )
         if ( opts.verb>1 ) fprintf('\t ss succeeded'); end;
         U=tU; S=tS; b=tb; J=tJ2; Ed=tEd2; Ew=tEw2; dv=tdv2;
       else
         if ( opts.verb>1 ) fprintf('\t ss failed'); end;
       end
       if ( opts.verb>1 ) fprintf('\n'); end;
     end          
   end
   
   
   % compute convergence measures
   if( J > oJ*(1+.1) ) 
      warning('Non-decrease!'); 
   end;
   if( iter==1 )      dJ0=abs(oJ-J); maJ=max(oJ,J)*2; dU0=dU; % get initial gradient est, for convergence tests
   elseif ( iter==2 ) dJ0=max(dJ0,abs(oJ-J));  maJ=max(maJ,J*2); dU0=abs(dU-odU); % get initial gradient est for the change in parameters
   elseif ( iter< 5 ) dJ0=max(dJ0,abs(oJ-J));  dU0=max(dU0,abs(dU-odU)); % conv if enough smaller than best single step
   end
   maJ =maJ*(1-opts.marate)+J(1)*(opts.marate); % move-ave obj est
   madJ=maJ-J; 
   % status printout
   if ( opts.verb > 0 ) 
      fprintf('\nL* %2d)\tEw=(%s)=%.06g\tR=%d\tEd=%.06g\tJ=%.06g\tdJ=%.06g\tdU=%.06g\n',...
              iter,sprintf('%3f,',S(1:min(end,2))),Ew,sum(abs(S)>max(abs(S))*eps),Ed,J,madJ,dU);
   end
   % convergence tests
   if( dU <= opts.tol || dU <= opts.tol0*dU0 || ...
       madJ <= opts.objTol || madJ <= opts.objTol0*dJ0 ) 
      break; 
   end;
   
   % re-orthogonalise the solution
   if ( opts.ortho>1 ) [S,U{:}]=parafacOrtho(S,U{:},'maxIter',3); end;

   % re-symetrize the solution
   if ( ~isempty(opts.symDim) ) [S,U{:}]=symetricParafac(opts.symDim,S,U{:}); end      

end
% pick the best
if ( oJ < J ) J=oJ; S=oS; U=oU; b=ob; end
if ( ooJ< J ) J=ooJ;S=ooS;U=ooU;b=oob;end
% re-orthogonalise the solution
if ( opts.ortho>1 )         [S,U{:}]=parafacOrtho(S,U{:},'maxIter',3); 
elseif ( opts.ortho>0 )     S=parafacCompress(S,U{:});
end;
% reduce the rank to opts.rank if too big
if ( numel(S)>opts.rank && sum(abs(S)>max(abs(S))*eps)<=opts.rank ) % prune extra components
   S=S(1:opts.rank); for d=opDims; U{d}=U{d}(:,1:opts.rank);  end;
end
% ensure dec importance order
[ans,si]=sort(abs(S),'descend'); S=S(si); for d=opDims; U{d}=U{d}(:,si); end;
% undo the re-scaling
if ( ~isempty(Cscale) ) S=S./Cscale; end;

% merge all parameters into a single output argument
Wb={S U{:} b};
% compute the final performance
[dv J Ed Ew]=LSigmaObj(X,Y,C*Cscale,trnInd,dim,opDims,opts.h,S,U,b);
if ( opts.verb >= 0 ) 
   fprintf('\nL* %2d)\tEw=(%s)=%6f\tR=%d\tEd=%6f\tJ=%6f\tdJ=%6f\tdU=%6f\n',...
           iter,sprintf('%3f,',S(1:min(end,2))),Ew,sum(abs(S)>eps),Ed,J,madJ,dU);
end
obj=[J Ed Ew(:)'];
return;
   
%------------------------------------------------------------------------------
function [A]=parafac(S,varargin);
U=varargin;
if ( numel(U)==1 && iscell(U{1}) ) U=U{1}; end;
if ( numel(S)==1 ) S=S*ones(size(U{1},2),1); end;
% Compute the full tensor specified by the input parallel-factors decomposition
nd=numel(U); A=shiftdim(S,-nd);  % [1 x 1 x ... x 1 x M]
for d=1:nd; A=tprod(A,[1:d-1 0 d+1:nd nd+1],U{d},[d nd+1],'n'); end
A=sum(A,nd+1); % Sum over the sub-factor tensors to get the final result
return;

%------------------------------------------------------------------------
function [S,varargout]=parafacStd(S,varargin);
% balance the norms in the parfac, i.e. make all component norms equal=1
U=varargin;
if ( numel(U)==1 && iscell(U{1}) ) U=U{1}; end;
r=size(U{1},2);% S=ones(r,1);
for id=1:numel(U); 
   nrm  = sqrt(sum(U{id}.^2,1)); 
   ok   = nrm>eps;
   if ( any(ok) )
      U{id}(:,ok) = repop(U{id}(:,ok),'./',nrm(ok));
      S(ok)      = S(ok).*nrm(ok)';
   end
   U{id}(:,~ok)= 0; % zero out ignored
   S(~ok)     = 0; 
end;
varargout=U;
return;

%------------------------------------------------------------------------------
function [dv J Ed Ew]=LSigmaObj(X,Y,C,trnInd,dim,opDims,h,S,U,b)
% compute the L_\Sigma regularised LR objective function
% efficient way to compute the decision values
if ( isempty(h) ) h=0; end;
nd=ndims(X);
dv=X;
for d=setdiff(1:nd,dim);
  dv=tprod(dv,[1:d-1 -d d+1:nd nd+1:ndims(dv)],U{d},[-d nd+1]);
end
dv  =shiftdim(tprod(dv,[1:nd -(nd+1)],S,-(nd+1)),dim-1) + b;
g    = 1./(1+exp(-(Y(trnInd).*dv(trnInd)))); g=max(g,eps);
Ed   = -sum(log(g));  % -ln(P(D|w,b,fp))
Ew   = sum(abs(S(abs(S)>h))) + sum(S(abs(S)<h).^2./h + h/2); % huberized L1 approx
J    = Ed + C(1)*Ew; 
return;

%------------------------------------------------------------------------------
function [ss,fss,a,b,c]=secantFit(s,f)
% do a quadratic fit to the input and return the postulated minimum point
a=(f(3)-f(1) - (f(2)-f(1))*s(3))/(s(3)*s(3)-s(3));
b=f(2)-f(1)-a;
c=f(1);
if ( abs(a)<eps ) 
  ss=0;
else
  ss=-b/abs(a)/2; 
end
fss=a*ss.^2+b*ss+c;
return;

%------------------------------------------------------------------------------
function testCase()
clear z
N=400; nAmp=.25; nSpect=min(32./[1:64],8); freq=16; freqStd=1;
z=jf_mks2nsfToy('N',N,'nAmp',nAmp,'nSpect',nSpect,'freq',freq,'freqStd',freqStd);   
z.label=sprintf('%s_nAmp=%.03g',z.label,nAmp);
oz=z;
% cov decomp
z   = jf_cov(z,'dim',{'ch' 'time'});
% L2 soln
r=jf_cvtrain(jf_compKernel(z))

% test this classifier
[soln dv]=LSigmaRegKLR_als(z.X,z.Y(:,1),1,'rank',20,'dim',n2d(z,'epoch'),'verb',2);

% and again taking account of the symetric inputs
[soln dv]=LSigmaRegKLR_als(z.X,z.Y(:,1),1,'rank',20,'dim',n2d(z,'epoch'),'symDim',[1 2]','verb',2);

%z = jf_cvtrain(z,'objFn','LSigmaRegKLR_als','rank',20,'Cs',5.^4,'reorderC',0,'outerSoln',0,'seedNm','Wb');

conf2loss(dv2conf(z.Y,dv))
dv2auc(z.Y,dv)

si=soln{1}>0;
clf;fact3DPlot(soln{2}(:,si),[],soln{3}(:,si)*diag(soln{1}(si)),[z.di(1).extra.pos2d],[],z.di(2).vals)

% cov decomp
oX=X; X = tprod(X,[1 -2 3],[],[2 -2 3]);

% 3d decomp
oX=X;
[start width]=compWinLoc(size(X,2),[],.5,[],50);
X = windowData(X,start,width,2);
X = fft_posfreq(X,[],2); % fourier trans
X = tprod(real(X),[1 3 4 5],[],[2 3 4 5]) + tprod(imag(X),[1 3 4 5],[],[2 3 4 5]);
X = msqueeze(4,sum(X,4));

% L2 reg
K = compKernel(X,[],'linear','dim',-1);
tic,[Wbls,J,f]=klr_cg(K,Y(:,1),C);toc
Wl2=tprod(X,[1:ndims(X)-1 -ndims(X)],Wbls(1:end-1),-ndims(X));bl2=Wbls(end);
fl2=tprod(X,[-(1:ndims(X)-1) 1],Wl2,[-(1:ndims(X)-1)])+bl2;
[Sl2,Ul2{1:ndims(X)-1}]=parfac_als(Wl2,'rank',20,'verb',1);

[f,df,ddf,obj]=LSigmaRegLRFn([Wl2(:);bl2],X,Y(:,1),C);
fprintf('L2) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));


% L_sigma Reg
Uls=Ul2; Uls{end}=repop(Ul2{end},'*',Sl2'); bls=bl2;% push norm into last component
tic,[Wb,f,J,Sls,Uls{1:ndims(X)-1}]=LSigmaRegKLR_als(X,Y(:,1),C,'objTol',1e-3,'verb',2,'rank',20,'Wb',{Uls{:} bls});toc
szX=size(X);
Uls = Wb{1:end-1}; bls=Wb{end};
Wlsigma = parafac(Sls,Uls{:});
blsigma = bls;
flsigma = tprod(X,[-(1:ndims(X)-1) 1],Wlsigma,-(1:ndims(Wlsigma)))+blsigma;

[f,df,ddf,obj]=LSigmaRegLRFn([Wlsigma(:);blsigma],X,Y(:,1),C/2);
fprintf('L*1) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));

clf;image3d(Wlsigma,3)

% CV training
foldIdxs=gennFold(Y(:,1),10);
fprintf('Condition number of K: %g\n',rcond(K));
Cscale = .1*sqrt((mean(diag(K))-mean(K(:))));
res=cvtrainFn('LSigmaRegKLR_als',X,Y(:,1),Cscale*10.^[-3:3],foldIdxs,...
              'verb',1,'tol',1e-2,'rank',20,'reuseParms','Wb','hps',{Uls{:} bls});

