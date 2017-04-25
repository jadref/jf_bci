function [alphabB,dv,J,alphab]=mkKLR(K,Y,C,varargin)
% L1 regularised multiple Kernel, Kernel Logistic Regression
%  
% [alphabB,dv,J,alphab]=mkKLR(K,Y,C,varargin)
% 
% Inputs:
%  K       -- [N x N x M] set of kernel matrices
%  Y       -- [N x 1] set of trial data labels in -1/0/+1 format
%  C       -- [L1,L2] regularisation parameter(s)
% Options:
%  dim     -- epoch dimension of K   (1)
%  featDim -- feature dimension of K (ndims(K))
%  maxIter -- maximum number of iterations to perform
%  wtol    -- tolerance on change in W
%  objTol  -- [2x1] tolerance on change in objective value, [outer,inner]
%  objTol0 -- [2x1] tolerance on change 
%              in objective value w.r.t. initial soln value [outer,inner]
%  marate  -- moving average decay const, used to smooth J estimates for tol
%  alphabB -- seed solution, {[alphab] [B]}
%  hyperObjFn -- [str] objective function for the \betas, 
%                 'L1RegLR' - impose the sum(B)=1 constraint
%                 'L1'      - minimise the sum(abs(B))
%                 otherwise - function to call to update B
%                             B = hyperObjFn(alphaK,Y,C(1),alphaKalpha,[B(:);b])
%  structMx   -- [nFeat x nReg] matrix which shows how the different kernels combine to 
%                   make a groupwise L1 structured regularisor           ([])
%                OR
%                 'ascending','descending','ascend+descend'
%               for ideas on how to use this structure matrix to impose structure on the solution see:
%                   Bach, Francis, Rodolphe Jenatton, Julien Mairal, and Guillaume Obozinski. 2011. 
%                   “Structured sparsity through convex optimization.” 1109.2397 (September 12). 
%                   http://arxiv.org/abs/1109.2397.
% Outputs:
%  Wb     -- {W b}
%  dv     -- [size(X,dim) x M] set of classifier predictions
%  J      -- objective function value
%

% TODO: [] starting from too high a regularisation causes convergence issues with later iterations...
opts=struct('verb',0,'maxIter',[],'maxEval',inf,'dim',[],'featDim',[],'h',0,...
            'objTol',1e-4,'objTol0',[1e-5 1e-4],'marate',.8,'alphabB',[],'alphab',[],...
            'incThresh',.7,'innerObjFn','klr_cg','hyperObjFn','L1','structMx',[],'normKernels',0,'trustRegionSize',.005);
[opts,varargin]=parseOpts(opts,varargin);

dim =opts.dim; if ( isempty(dim) ) dim=1; end;
fDim=opts.featDim; if ( isempty(fDim) ) fDim=ndims(K); end;
if ( numel(dim)>1 ) error('Only 1 feature dim is supported'); end;
if ( numel(C)<2 ) C(end+1:2)=0; end

szK=size(K); nd=ndims(K); N=size(K,1); nFeat=prod(szK(fDim));

maxIter=opts.maxIter; if ( isempty(maxIter) ) maxIter=nFeat*50; end;

% force only 1st sub-problem
% setup- the sub-problems
spi=1; if ( size(Y,2)>1 ) warning('Additional subproblems ignored!'); Y=Y(:,1); end;

% restrict to the training sub-set of the data if that would help
incInd=any(Y~=0,2);
oK=K; oY=Y;
if ( sum(incInd)./size(Y,1) < opts.incThresh )
  K=K(incInd,incInd,:,:); Y=Y(incInd,:); 
  szK=size(K); N=szK(1);
else
  incInd=false;
end

% re-scale the data if wanted to ensure better performance
Cscale=opts.normKernels;
if ( ~isempty(Cscale) && isequal(Cscale,1) ) % per-kernel normalisation
  for ki=1:size(K(:,:,:),3); Cscale(ki)=CscaleEst(K(:,:,ki),1); end;
elseif ( isempty(Cscale) )
  Cscale=CscaleEst(K,1)./nFeat; % single normaliser
end
if ( ~isequal(Cscale,0) ) % re-scale the kernels as wanted
  for ki=1:size(K(:,:,:),3); K(:,:,ki)=K(:,:,ki)./Cscale(min(end,ki)); end;
  C=C./sqrt(mean(Cscale)); % (approx) Correct the C for this re-scaling, N.B. sqrt(C) because is L1 loss
  %B=B(:).*Cscale(:); % modify seed to keep solution the same
end

% Extract the provided seed values
if ( ~isempty(opts.alphab) && iscell(opts.alphab) && numel(opts.alphab)==2 )
  persistent warned;
  if ( opts.verb>0 && isempty(warned) ) warning('Using alphab as seed, but Correct seed name is : alphabB'); warned=true; end;
  opts.alphabB=opts.alphab; opts.alphab=[];
end
if ( ~isempty(opts.alphabB) )
   alphab=opts.alphabB{1}; B=opts.alphabB{2};
   if ( any(incInd) ) alphab=alphab([incInd; true],:); end
   % update the solution to take account of the scaling
   if ( ~isequal(Cscale,0) ) B=B(:).*Cscale(:); end
else % compute a decent seed solution
   alphab=zeros(N+1,1); B=ones(nFeat,1);
   % 2) Update the kernel to the new solution   
   Kp = tprod(K,[1:2 -3],B,[-3],'n'); % compute the updated K with this feature weighting
   % 3) Compute the L2 reg solution with this kernel
   [alphab,dv,J,obj]=feval(opts.innerObjFn,Kp,Y,C(1),'verb',opts.verb-1,...
                           'maxEval',opts.maxEval,'objTol',opts.objTol(min(end,2)),...
                           'objTol0',opts.objTol0(min(end,2)),varargin{:});
 end

Cc=C(1);
if( strcmp(opts.hyperObjFn,'L1RegLR') ) B = B./sum(B); Cc=C(1); end

structMx=opts.structMx; 
if ( ~isempty(structMx) && isstr(structMx) )
  switch (structMx);
   case {'ascending','ascend','last2first'}; 
    structMx=zeros(size(K,3),size(K,3)); for i=1:size(structMx,1); structMx(1:i,i)=1;   end;
   case {'descending','descend','first2last'};
    structMx=zeros(size(K,3),size(K,3)); for i=1:size(structMx,1); structMx(i:end,i)=1; end;
   case {'ascend+descend','convex'}; 
    structMx=zeros(size(K,3),size(K,3)*2); 
    for i=1:size(structMx,1); structMx(1:i,i)=1; structMx(i:end,i+size(structMx,1))=1; end;
   otherwise; error(sprintf('Unrecoginised structure matrix name: %s',structMx));
  end
  % normalise the structure matrix so each reg has equal importance, and total reg strength is 1
  structMx=repop(structMx,'./',sum(structMx,1))*size(K,3)/size(structMx,2);
end
    
% 3) Extract the current solution's info!!!
alphaK      = tprod(alphab(1:end-1),-1,K,[-1 1 2]); % [N x nFeat], per-kernel dv - not rescaled
alphaKB     = alphaK*B; % [ N x 1 ], per-kernel dv
alphaKalpha = alphab(1:end-1)'*alphaK;  % [1 x nFeat]
alphaBKBalpha= alphaKalpha.*(B(:).^2)'; % L2 % R_m  % [ 1 x nFeat ], per-kernel l2 norm
dv  = alphaKB + alphab(end); 
if ( isequal(opts.innerObjFn(1:min(end,3)),'klr') )
  g   = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count 0
  Ed  = -sum(log(g));  % -ln(P(D|w,b,fp))
elseif ( isequal(opts.innerObjFn(1:min(end,5)),'l2svm') )
  err=1-Y.*dv; svs=err>0 & Y~=0; 
  Ed  = sum(err(svs).^2);
elseif ( isequal(opts.innerObjFn(1:min(end,4)),'rkls') )
  Ed  = sum((Y-dv).^2);
else error('unsupported objective function');
end
if ( ~isempty(strmatch(opts.hyperObjFn,{'L1RegLR','L1'},'exact')) ) % the new reg loss
  % N.B. need extra B* to undo the rescaled alpha=alpha/B
  nrms   = alphaBKBalpha';
  if ( ~isempty(structMx) )           nrms=(nrms'*structMx)'; end
  Ew     = sqrt(nrms);                                 % regularisation cost is L1 loss
  idx=abs(Ew)<2*opts.h; % huberize
  if ( any(idx) ) Ew(idx)= sqrt(nrms(idx))/2/opts.h + opts.h/2; end; % huberize
end
J   = sum(Ed) + C(1)*sum(Ew);
if ( opts.verb > 0 ) 
  fprintf('MKL %2d)\tEw=(%s)=%.06g\tEd=%.06g\tJ=%.06g\tdJ=%.06g\tdalphab=%.06g\n',...
          0,sprintf('%3f,',Ew(1:min(end,2))),sum(Ew),Ed,J,inf,inf);
end

deltaalphab=inf;
for iter=1:maxIter;
   oJ=J; oalphab=alphab; oB=B; odeltaalphab=deltaalphab;

   % 4) Update the feature weightings
   % optimise w.r.t. the hyper-parameter variables
   if ( strcmp(opts.hyperObjFn,'L1RegLR') )
     [Wb,dv,J,obj]=L1RegLR(alphaK,Y,C(1),alphaKalpha,[B(:);alphab(end)],...
                           opts.h,maxIter(min(end,1)),opts.objTol(min(end,2)),opts.verb-1);
     B=abs(Wb(1:end-1)); 
     alphab(end)=Wb(end); % extract the solution
   elseif ( ~isempty(strmatch(opts.hyperObjFn,'L1')) ) % L1 objective
     % kernel re-scaling to make L2 equiv to L1 regularisation cost, 
     % N.B. w_k = B(k)*X*alpha  -> w_k^2 = alpha*X(k)*B(k)*B(k)*X(k)*alpha
     nrms   = alphaBKBalpha';   % [nFeat x 1] norm of w in the non-rescaled space
     if ( ~isempty(structMx) )
       R    = sqrt(nrms'*structMx)';
       grad = structMx*(1./R);
       B    = 2./grad;  
     else % normal L1 per kernel
       B    = 2*sqrt(nrms); % B = 2./(1./sqrt(nrms)) = 2./(d|x|/dx)
     end
     B      = max(B,2*opts.h); % huberize
   else % assume hyperObjFn is a correct function to use
     [B,Ew]=feval(opts.hyperObjFn,alphaKB,Y,C(1),alphaBKBalpha,[B(:);alphab(end)]);
   end

   % 4 ) compute how big a step along B to take and still trust the old solution
   oalphaKB = alphaKB;      oalphaKB2 = oalphaKB'*oalphaKB;
   r  = inf;
   for i=1:10;
     % how much can we re-use of the old solution in the new kernel space
     nalphaKB  = alphaK*B;           nalphaKB2 = nalphaKB'*nalphaKB;
     onalphaKB=  oalphaKB'*nalphaKB;
     a       = onalphaKB/nalphaKB2; % least squares regression of new onto old values
     r       = oalphaKB2 - 2*a*onalphaKB + a*a*nalphaKB2;
     if ( r./oalphaKB2 < opts.trustRegionSize ) 
       break; 
     else % decrease step size by factor of 2
       B=oB+(B-oB)/2;
     end
   end
   % 5) update the alphab to compensate for the change in the kernel weighting      
   alphab(1:end-1)= a*alphab(1:end-1); % re-scale alphab as appropriate
   
   % 2) Update the kernel to the new solution / seed new training with the old solution
   Kp = tprod(K,[1:2 -3],B,[-3],'n'); % compute the updated K with this feature weighting
   
   % 3) Compute the L2 reg solution with this kernel
   [alphab,dv,J,obj]=feval(opts.innerObjFn,Kp,Y,Cc,'alphab',alphab,'verb',opts.verb-1,...
                            'maxEval',opts.maxEval,'objTol',opts.objTol(min(end,2)),...
                            'objTol0',opts.objTol0(min(end,2)),varargin{:});
   % 3) Extract the current solution's info!!!
   % N.B. need extra B* to undo the rescaled alpha=alpha/B
   alphaK      = tprod(alphab(1:end-1),-1,K,[-1 1 2]); % [N x nFeat], per-kernel dv - not rescaled
   alphaKB     = alphaK*B; % [ N x nFeat ], per-kernel dv
   alphaKalpha = alphab(1:end-1)'*alphaK;
   % N.B. need extra B* to undo the rescaled alpha=alpha/B
   alphaBKBalpha = alphaKalpha.*(B(:).^2)'; % L2 % R_m  % [ 1 x nFeat ], per-kernel l2 norm

   % 6) Compute the updated solution objective value
   dv  = alphaKB + alphab(end); 
   if ( isequal(opts.innerObjFn(1:min(end,3)),'klr') )
     g   = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count
     Ed  = -sum(log(g));  % -ln(P(D|w,b,fp))
   elseif ( isequal(opts.innerObjFn(1:min(end,5)),'l2svm') )
     err=1-Y.*dv; svs=err>0 & Y~=0; 
     Ed  = sum(err(svs).^2);
   elseif ( isequal(opts.innerObjFn(1:min(end,4)),'rkls') )
     Ed  = sum((Y-dv).^2);
   else error('unsupported objective function');
   end
   if ( strcmp(opts.hyperObjFn,'L1RegLR') || strcmp(opts.hyperObjFn,'L1') ) % the new reg loss
     % N.B. need extra B* to undo the rescaled alpha=alpha/B
     nrms   = alphaBKBalpha';
     if ( ~isempty(structMx) )           nrms=(nrms'*structMx)'; end;
     Ew     = sqrt(nrms);                     % regularisation cost is L1 loss
     idx    = Ew<opts.h;
     if ( any(idx) ) Ew(idx)= sqrt(nrms(idx))/2/opts.h + opts.h/2; end; % huberize
   end
   J   = sum(Ed) + C(1)*sum(Ew);
   obj = [Ed Ew(:)'];
   
   % 7) convergence test stuff
   % compute convergence measures
   deltaalphab = norm(alphab(1:numel(oalphab))-oalphab(:));
   if( J > oJ*(1+.01) && opts.verb>1) 
      warning('Non-decrease!'); 
   end;
   if( iter<=1 )      dJ0=abs(oJ-J); madJ=dJ0; maJ=inf; % get initial gradient est, for convergence tests
   elseif ( iter==2 ) dJ0=max(dJ0,abs(oJ-J));  maJ=max(oJ,J);
   elseif ( iter< 5 ) dJ0=max(dJ0,abs(oJ-J));  maJ=max(maJ,J); % conv if enough smaller than best single step
   end
   maJ =maJ*(1-opts.marate)+J*(opts.marate); % move-ave obj est
   madJ=maJ-J; %madJ*(1-opts.marate)+max(0,oJ-J)*(opts.marate); % move-ave grad est
   % status printout
   if ( opts.verb > 0 ) 
      fprintf('MKL %2d)\tEw=(%s)=%.06g\tEd=%.06g\tJ=%.06g\tdJ=%.06g\tdalphab=%.06g\n',...
              iter,sprintf('%3f,',Ew(1:min(end,2))),sum(Ew),Ed,J,madJ,deltaalphab);
   end
   % convergence tests
   if( iter>4 && (madJ <= opts.objTol(1) || madJ <= opts.objTol0(1)*dJ0) ) 
      break; 
   end;

end

% undo the kernel scaling if necessary
if ( ~isempty(Cscale) && ~isequal(Cscale,0) ) % re-scale the kernels as wanted
  B=B(:)./Cscale(:);
  C=C.*sqrt(mean(Cscale)); % undo hyper-param re-scaling
end
% if ( strcmp(opts.hyperObjFn,'L1') )
%   B=B/2;
%   alphab(1:end-1)=2*alphab(1:end-1);  
% end
% compute the full set of predictions
if ( any(incInd) ) % map back to the full kernel space, if needed
   nalphab=zeros(size(oK,1)+1,1); nalphab(incInd)=alphab(1:end-1); nalphab(end)=alphab(end); 
   alphab=nalphab;
end
K=oK; Y=oY;
if ( any(incInd) ) % sub-setted version for speed
  alphaK      = tprod(alphab(incInd),-1,K(incInd,:,:),[-1 1 2]); % [N x nFeat], per-kernel dv - not rescaled
  % N.B. need extra B* to undo the rescaled alpha=alpha/B
  alphaBKBalpha = (alphab(incInd)'*alphaK(incInd,:)).*(B(:).^2)'; % L2 % R_m  % [ 1 x nFeat ], per-kernel l2 norm
else
  alphaK      = tprod(alphab(1:end-1),-1,K,[-1 1 2]); % [N x nFeat], per-kernel dv - not rescaled
  alphaBKBalpha = (alphab(1:end-1)'*alphaK).*(B(:).^2)'; % L2 % R_m  % [ 1 x nFeat ], per-kernel l2 norm
end
dv  = alphaK*B + alphab(end); 
g   = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count
if ( isequal(opts.innerObjFn(1:min(end,3)),'klr') )
  g   = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count
  Ed  = -sum(log(g));  % -ln(P(D|w,b,fp))
elseif ( isequal(opts.innerObjFn(1:min(end,5)),'l2svm') )
  err=1-Y.*dv; svs=err>0 & Y~=0; 
  Ed  = sum(err(svs).^2);
elseif ( isequal(opts.innerObjFn(1:min(end,4)),'rkls') )
  Ed  = sum((Y-dv).^2);
else error('unsupported objective function');
end
if ( strcmp(opts.hyperObjFn,'L1RegLR') || strcmp(opts.hyperObjFn,'L1') ) % the new reg loss
  nrms   = alphaBKBalpha';
  if ( ~isempty(structMx) )           nrms=(nrms'*structMx)'; end
  Ew     = sqrt(nrms);                                 % regularisation cost is L1 loss
  idx=abs(Ew)<2*opts.h; 
  if ( any(idx) ) Ew(idx)= sqrt(nrms(idx))/2/opts.h + opts.h/2; end; % huberize
end
J   = sum(Ed) + C(1)*sum(Ew);
if ( opts.verb > 0 ) 
      fprintf('MKL %2d)\tEw=(%s)=%.06g\tEd=%.06g\tJ=%.06g\tdJ=%.06g\tdalphab=%.06g\n',...
              iter,sprintf('%3f,',Ew(1:min(end,2))),sum(Ew),Ed,J,madJ,deltaalphab);
end

% return the solution
alphabB={alphab B};
return;

%------------------------------------------------------------------------------
function [Wb,dv,f,obj]=L1RegLR(X,Y,C,Rm,Wb,h,maxEval,tol,verb)
% L1 regularised primal LR objective function, subject to the sum(B)=1 constraint
Rm=Rm(:); Wb=Wb(:); % ensure col vector
L1nrm=sum(abs(Wb(1:end-1)));
h = h * L1nrm;
% change variables to make a regular unit regularised optimisation
iRm = 1./(max(Rm,eps)); iRm(abs(Rm)<eps)=0;
Xp  = repop(X,'*',iRm')';  % [ nFeat x N ]
Wbp = [Wb(1:end-1).*Rm;Wb(end)]; % map the space so reg is constant over all dimensions
% compute the perpendicular to the constraint surface to impose the equality constraint
eqCons=[ones(numel(Wbp)-1,1).*iRm;0]; % constrain with 1./Rm * W = 1
eqCons=eqCons./sqrt(eqCons'*eqCons); % normalise the equality constraint
% do the constrained optimisation
Wbp=nonLinConjGrad(@(w) primalL1RegLRFn(w,Xp,Y,C,h),Wbp,'maxEval',maxEval,'tol',tol,'verb',verb,...
                  'eqConstraint',eqCons);
[f,df,ddf,obj]=primalL1RegLRFn(Wb,Xp,Y,C,h); % compute the solution quality
% extract and untransform the solution
Wb=[Wbp(1:end-1).*iRm;Wbp(end)]; 
% enforce the L1 norm constraint
Wb(1:end-1)=Wb(1:end-1)./sum(abs(Wb(1:end-1)))*L1nrm; 
dv=X*Wb(1:end-1)+Wb(end); 
dv=dv(:); % ensure is col vector
return;


%------------------------------------------------------------------------------
function testCase()
clear z
N=400; nAmp=2; nSpect=min(32./[1:64],8); freq=16; freqStd=1;
z=jf_mks2nsfToy('N',N,'nAmp',nAmp,'nSpect',nSpect,'freq',freq,'freqStd',freqStd);   
z.foldIdxs=gennFold(z.Y,10);
z.label=sprintf('%s_nAmp=%.03g',z.label,nAmp);
oz=z;

K = tprod(z.X,[3 -2 1],[],[3 -2 2],'n'); % kernel per channel
[Wb]=mkKLR(K,z.Y,1,'verb',1,'Cscale',1)

[Wb]=mkKLR(K,z.Y,1,'verb',1,'hyperObjFn','L1','Cscale',1); % groupwise L1 reg learning
% compare with L1RegKLR
[Wb,f,J]=L1RegKLR(z.X,z.Y(:,1),1,'objTol0',1e-3,'verb',1,'grpDim',2,'dim',3)

% test seed propogation when re-scale a feature
[X,Y]=mkMultiClassTst([-1 0 0 0 0; 1 0 0 0 0; .2 0 .5 0 0],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
K=X'*X; 
[alphab,f,J]=klr_cg(K,Y,0,'verb',1);
W=X*alphab(1:end-1); owoX=oW'*X;
alphaK=alphab(1:end-1)'*K; 

% plot the result
alpha=zeros(N,1);alpha=alphab(1:end-1); % equiv alpha
clf;plotLinDecisFn(X,Y,X*alphab(1:end-1),alphab(end),[],alpha);

% try mkKLR
K=tprod(X,[3 1],[],[3 2]);
[Wb,dv,J]=mkKLR(K,Y,1,'verb',1);

% extract an input space weighting, and check it is correct
W = Wb{2}.*(X*Wb{1}(1:end-1));
clf;plotLinDecisFn(X,Y,W,Wb{1}(end),[],Wb{1}(1:end-1));

% try with alternative loss functions
[Wb,dv,J]=mkKLR(K,Y,1,'verb',1,'innerObjFn','klr_cg');
[Wb,dv,J]=mkKLR(K,Y,1,'verb',1,'innerObjFn','l2svm_cg');
[Wb,dv,J]=mkKLR(K,Y,1,'verb',1,'innerObjFn','rkls_cg');

nX=X; nX(1,:)=nX(1,:)*10;
nK=nX'*nX; 
nW=oW;nW(1,:)=nW(1,:)/10;
nwnX=nW'*nX; nw2=nW'*nW;
K2=[nK alphaK';alphaK nw2];
% seeded version
[alphab2,f,J]=klr_cg(K2,[Y;0],0,'alphab',[zeros(size(K2,1)-1,1);1;alphab(end)],'verb',1);
% unseeded
[alphab2,f,J]=klr_cg(K2,[Y;0],0,'verb',1);


% try with a structured regularisor
K=tprod(X,[3 1],[],[3 2]);
%[Wb,dv,J]=mkKLR(K,Y,1,'verb',1);
[Wb,dv,J]=mkKLR(K,Y,5,'verb',2,'structMx','last2first'); % last2first preference for features
% variables ordered from 1st to last
[Wb,dv,J]=mkKLR(K,Y,5,'verb',2,'structMx','first2last'); % first2last feature preference
% check computed weighting is correct
dv2 = W'*X+Wb{1}(end);
mad(dv,dv2),clf;plot([dv dv2(:)]);

% with an arbitary structure matrix
[Wb,dv,J]=mkKLR(K,Y,1,'verb',2,'structMx','ascending');

alphab=Wb{1}; B=Wb{2};
alphaK = tprod(alphab(1:end-1),-1,Ks,[-1 1 2]);
alphaKB= alphaK*B;
dv2    = alphaKB+alphab(end);
alphaKalpha = alphab(1:end-1)'*alphaK;
alphaBKBalpha = alphaKalpha.*(B(:)'.^2);

