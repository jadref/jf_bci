function [wb,f,J,obj,tstep]=lr_cg(X,Y,R,varargin);
% l1 Regularised linear Logistic Regression Classifier
%
% [wb,f,J,obj]=lr_cg(X,Y,C,varargin)
% Regularised Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J = \sum_g sqrt(w.^2' R(g)+c_g) + sum_i log( (1 + exp( - y_i ( w'*X_i + b ) ) )^-1 ) 
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
%  R       - quadratic regularisation matrix                                   (0)
%     [1x1]        -- simple regularisation constant             R(w)=sum(abs(w))
%     [d1xd2x...G] -- simple weighting of each component matrix, R(w)=\sum_g sqrt(w'*diag(R(:,g))*w)
%            (Note: can also use the structMx option to specify the regularisor structure)
%            N.B. Note that the R is square-rooted, the reg-strength C is given by: C=sqrt(R) or R=C.^2
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
%
% Options:
%  wb      - [(N+1)x1] initial guess at the weights parameters, [W;b]     ([])
%  maxEval - [int] max number for function evaluations                    (N*5)
%  maxIter - [int] max number of CG steps to do                           (inf)
%  maxLineSrch - [int] max number of line search iterations to perform    (50)
%  objTol0 - [float] relative objective gradient tolerance                (1e-5)
%  objTol  - [float] absolute objective gradient tolerance                (0)
%  tol0    - [float] relative gradient tolerance, w.r.t. initial value    (0)
%  lstol0  - [float] line-search relative gradient tolerance, w.r.t. initial value   (1e-2)
%  tol     - [float] absolute gradient tolerance                          (0)
%  verb    - [int] verbosity                                              (0)
%  step    - initial step size guess                                      (1)
%  wght    - point weights [Nx1] vector of label accuracy probabilities   ([])
%            [2x1] for per class weightings
%            [1x1] relative weight of the positive class
%  eta     - [float] smoothing constant for quadratic approx to l1 loss   (1e-6)
%  structMx   -- [nFeat x nReg] matrix which shows how the different features combine to 
%                   make a groupwise L1 structured regularisor           ([])
%                OR
%                 'ascending','descending','ascend+descend'
%               for ideas on how to use this structure matrix to impose structure on the solution see:
%                   Bach, Francis, Rodolphe Jenatton, Julien Mairal, and Guillaume Obozinski. 2011. 
%                   “Structured sparsity through convex optimization.” 1109.2397 (September 12). 
%                   http://arxiv.org/abs/1109.2397.
%  Rconst - [nReg x 1] vector of constants to add to the group-wise cost ([])
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied
if ( nargin < 3 ) R(1)=0; end;
if( numel(varargin)==1 && isstruct(varargin{1}) ) % shortcut eval option procesing
  opts=varargin{1};
else
  opts=struct('wb',[],'alphab',[],'dim',[],'Jconst',0,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-6,'objTol',0,'objTol0',1e-6,...            
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'bPC',[],'wPC',[],'incThresh',.75,'optBias',0,'maxTr',inf,...
              'getOpts',0,'eta',1e-5,'zeroStart',0,'structMx',[],'PCcondtol',10,'Rconst',[],'nFeat',[]);
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end

szX=size(X); nd=numel(szX); N=szX(end); nf=prod(szX(1:end-1)); mineta=1e-5;
Y=Y(:); % ensure Y is col vector

% Ensure all inputs have a consistent precision
if(isa(X,'double') & isa(Y,'single') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence
opts.tol(end+1:2)=opts.tol(1);
if ( isempty(opts.maxEval) ) opts.maxEval=5*max(prod(szX(1:end-1)),szX(end)); end
Rconst=opts.Rconst; if( isempty(Rconst) ) Rconst=0; end;

C=1; newC=false; % scaling for search for the right number of features.

% allow spec of reg structure with structMx option
structMx=opts.structMx; 
if ( ~isempty(structMx) && (isstr(structMx) || numel(structMx)==1) )
  structMx=mkStructMx(szX(1:end-1),structMx);
end
if ( ndims(structMx)>2 ) 
  structMx=reshape(structMx,[],size(structMx,ndims(structMx))); % work with vector X
  %if ( sum(structMx(:)==0)>numel(structMx)/2 ) structMx=sparse(structMx); end;
end
if ( numel(R)==1 ) 
  if ( R<0 ) opts.nFeat=-R; R=1; end % neg regularisation constant => number of features to have
  if ( ~isempty(structMx) ) R=R*structMx; end;
end

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);
szR=size(R);
if ( numel(R)==1 )
  szRw=[nf 1]; RType=1; % scalar
elseif ( isequal(szR,szX(1:end-1)) || numel(R)==nf ) % weighting vector
  szRw=[nf 1]; RType=2;
elseif ( szR(end)>1 && (isequal(szR(1:end-1),szX(1:end-1)) || size(R,1)==nf) ) % group l1 reg
  szRw=[nf szR(end)]; RType=3;
else
  error('Huh, dont know how to use this regularisor');
end
if ( ndims(R)>2 ) R=reshape(R,szRw); end;

% check for degenerate inputs
if ( all(Y>=0) || all(Y<=0) )
  warning('Degnerate inputs, 1 class problem');
end

% N.B. this form of loss weighting has no true probabilistic interpertation!
wght=opts.wght;wghtY=Y;
if ( ~isempty(opts.wght) ) % point weighting -- only needed in wghtY
   if ( numel(wght)==1 ) % weight ratio between classes
     wght=zeros(size(Y));
     wght(Y<0)=1./sum(Y<0)*opts.wght; wght(Y>0)=(1./sum(Y>0))*opts.wght;
     wght = wght*sum(abs(Y))./sum(abs(wght)); % ensure total weighting is unchanged
   elseif ( numel(opts.wght)==2 ) % per class weights
     wght=zeros(size(Y));
     wght(Y<0)=opts.wght(1); wght(Y>0)=opts.wght(2);
   elseif ( numel(opts.wght)==N )
   else
     error('Weight must be 2 or N elements long');
   end
   wghtY=wght.*Y;
end

% check if it's more efficient to sub-set the data, because of lots of ignored points
oX=X; oY=Y;
incIdx=Y(:)~=0;
if ( sum(incIdx)./numel(Y) < opts.incThresh ) % if enough ignored to be worth it
   if ( sum(incIdx)==0 ) error('Empty training set!'); end;
   X=X(:,incIdx); Y=Y(incIdx); if ( numel(wght)==numel(Y) ) wght=wght(incIdx); end;
end

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  w=zeros(nf,1);b=0;
  if ( ~opts.zeroStart ) 
    % prototype classifier seed
    alpha=zeros(numel(Y),1);
    alpha(Y>0)=.5./sum(wghtY(Y>0)); alpha(Y<0)=.5./sum(wghtY(Y<0)); 
    w = X*alpha;
    switch ( RType ) % diff types regularisor
     case {1,2};
      if ( isequal(Rconst,0) ) Rg=w+Rconst(:); else Rg=w; end; % include effect of the constant in the regulisor
      nrm=abs(Rg);
      nrmeta=max(max(nrm)*opts.eta,nrm); varR=R./nrmeta/2;      dR=R(:).*sign(Rg);
     case 3;    
      nrm=sqrt((w.^2)'*R +Rconst(:)'); 
      nrmeta=max(max(nrm)*opts.eta,nrm); varR=R*(1./nrmeta)'/2; dR=2*varR.*w;
    end
    wRw   = C*2*w'*(varR.*w);
    wX    = w'*oX; wX=wX(incIdx); % only included points in seed
    % find least squares optimal scaling and bias
    sb = pinv([wRw+wX*wX' sum(wX); sum(wX) sum(Y~=0)])*[wX*oY(incIdx); sum(wghtY)];
    w=w*sb(1); b=sb(2);
  end
else
  w=wb(1:end-1); b=wb(end);
end 

switch ( RType ) % diff types regularisor
 case {1,2};
  if ( isequal(Rconst,0) ) Rg=w+Rconst(:); else Rg=w; end; % include effect of the constant in the regulisor

  nrm=abs(Rg);           nrmeta=max(max(nrm)*opts.eta,nrm); 
  varR=R./nrmeta/2;      dR=R(:).*sign(Rg);
 case 3;    
  nrm=sqrt((w.^2)'*R +Rconst(:)');         nrmeta=max(max(nrm)*opts.eta,nrm); 
  varR=R*(1./nrmeta)'/2; dR=2*varR.*w;
end
wX   = w'*X;
dv   = wX+b;
g    = 1./(1+exp(-Y'.*dv)); % =Pr(x|y), max to stop log 0
Yerr = wghtY'.*(1-g);
dL   = X*Yerr';
X2 = X.*X;

nFeat = sum(nrm>max(nrm).*opts.eta); 
if ( ~isempty(opts.nFeat) ) 
  wght=g(:).*(1-g(:));
  %dL = -X*Yerr';
  ddL= X2*(wght+.01); % add a ridge to the hessian estimate
  dL0= dL-ddL.*w;
  if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
    %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
    %     norm of the weight change in the same direction then the component will grow.
    %     Thus we only need to see if the loss-gradient is bigger then 1
    % N.B. istructMx effectively changes variables for each group independently such that
    %      in the transformed space the regularisor is a normal sqrt(w.^2) = |w|_2
    %      Then we can use the normal reasoning to find the optimal C in this case, i.e.
    %      C*dR > dL where as dR(w)=1 when w=0 we have C>dL
    istructMx = structMx; istructMx(structMx~=0)=1./istructMx(structMx~=0);
    dLRg= sqrt((double(dL0(:)).^2)'*istructMx);
  else
    dLRg = abs(dL0);
  end
  [sdLRg]=sort(dLRg,'descend');         
  C=sdLRg(opts.nFeat);
  fprintf('%d) nF=%d C=%g\n',0,nFeat,C); 
end

% set the pre-conditioner
% N.B. the Hessian for this problem is:
%  H  =[X*diag(wght)*X'+ddR  (X*wght');...
%       (X*wght')'           sum(wght)];
% where wght=g.*(1-g) where 0<g<1
% So: diag(H) = [sum(X.*(wght.*X),2) + 2*diag(ddR);sum(wght)];
% Now, the max value of wght=.25 = .5*(1-.5) and min is 0 = 1*(1-1)
% So approx assuming average wght(:)=.25/2; (i.e. about 1/2 points are on the margin)
%     diag(H) = [sum(X.*X,2)*.25/2+2*diag(R);N*.25/2];
wPC  = opts.wPC; bPC=opts.bPC;
if ( isempty(wPC) ) 
  wght=g(:).*(1-g(:));
  ddL= X2*(max(wght,.01)); % add a ridge to the hessian estimate
  wPCx=ddL;%(X.*X)*max(1e-2,(g(:).*(1-g(:)))); % H=X'*diag(g*(1-g))*X -> diag(H) = (X.^2)'*(g*(1-g)) 
  %wPCx=tprod(X,[1 -2],[],[1 -2])*.25/2;% H=X'*diag(g*(1-g))*X -> diag(H)=(X.^2)'*(g*(1-g)) ~= .25/2*sum(X.^2,2)
  wPC =wPCx;
  % include the effect of the regularisor
  wPCr=2*varR;
  wPC =C*wPCx+wPCr;
  wPC(wPC<eps) = 1; 
  wPC=1./wPC;
end;
if ( isempty(bPC) ) 
  bPC=1./(size(X,2)*.25/2);   %bPC=sqrt((X(:)'*X(:)))/numel(X);
end % ==mean(diag(cov(X)))
if ( isempty(wPC) ) 
  wPC=ones(size(X,1),1); 
elseif( numel(wPC)==1 ) 
  wPC=wPC*ones(size(X,1),1);
end;
PC = [wPC(:);bPC];

dJ   = [C*dR+dL; ...
        -sum(Yerr)];
% precond'd gradient:
%  [H  0  ]^-1 [ dR-X'((1-g).Y))] 
%  [0  bPC]    [   -1'((1-g).Y))] 
MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d'*dJ);
r2   = dtdJ;

Ed   = -log(max(g,eps))*(Y.*wghtY); % -ln P(D|w,b,fp)
switch ( RType ) % diff types regularisor % -ln P(w,b|R);
 case 1; Ew = R*sum(abs(Rg));
 case 2; Ew = R(:)'*abs(Rg);
 case 3; Ew = sum(nrm);
end
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
%if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] |x|=%2d J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           0,neval,w(1),w(2),sum(nrm>max(nrm).*opts.eta),J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
w0=w; b0=b;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

   oJ= J; oMr  = Mr; or2=r2; ow=w; ob=b; % record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 2 )
      fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ew,Ed); 
      if ( opts.verb>3 ) 
         hold off;plot(0,dtdJ,'r*');hold on;text(0,double(dtdJ),num2str(0)); 
         grid on;
      end
   end;
   ostep=inf;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   % pre-compute for speed later
   wX0 = wX;
   dw  = d(1:end-1); db=d(end);
   dX  = dw'*X;
   dvarR = dw.*varR(:); 
   dvarRw= dvarR'*w; 
   dvarRd= dvarR'*dw;
   % initial values
   dtdJ  = -(2*C*dvarRw - dX*Yerr' - db*sum(Yerr));
   if ( opts.verb > 2 )
     fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,C*Ew,Ed); 
   end
   odtdJ=dtdJ;      % one step before is same as current
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      wX    = wX0+tstep*dX;%w'*X;
      g     = 1./(1+exp(-Y'.*(wX+(b+tstep*db))));
      Yerr  = wghtY'.*(1-g);
      dtdJ  = -(2*C*(dvarRw+tstep*dvarRd) - dX*Yerr' - db*sum(Yerr));
      %fprintf('.%d step=%g ddR=%g ddgdw=%g ddgdb=%g  sum=%g\n',j,tstep,2*(dRw+tstep*dRd),-dX*Yerr',-db*sum(Yerr),-dtdJ);
      
      if ( opts.verb > 2 )
         Ed   = -log(max(g,eps))*(Y.*wghtY);        % P(D|w,b,fp)
         Ew   = (w(:).*varR)'*w(:)+tstep*2*dvarRw+tstep.^2*dvarRd;  % w'*(R*reshape(w,szR));       % P(w,b|R);
         J    = Ed + C*Ew + opts.Jconst;              % J=neg log posterior
         fprintf('.%d %g=%g @ %g (%g+%g)\n',j,tstep,dtdJ,J,Ew,Ed); 
         if ( opts.verb > 3 ) 
            plot(tstep,dtdJ,'*'); text(double(tstep),double(dtdJ),num2str(j));
         end
      end;

      % convergence test, and numerical res test
      if(iter>1||j>2) % Ensure we do decent line search for 1st step size!
         if ( abs(dtdJ) < opts.lstol0*abs(dtdJ0) || ... % Wolfe 2, gradient enough smaller
              abs(dtdJ*step) <= opts.tol(1) )              % numerical resolution
            break;
         end
      end
      
      % now compute the new step size
      % backeting check, so it always decreases
      if ( oodtdJ*odtdJ < 0 & odtdJ*dtdJ > 0 ...      % oodtdJ still brackets
           & abs(step*dtdJ) > abs(odtdJ-dtdJ)*(abs(ostep+step)) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         %but prev points gradient, this is necessary stop very steep orginal gradient preventing decent step sizes
         odtdJ = -sign(odtdJ)*sqrt(abs(odtdJ))*sqrt(abs(oodtdJ)); % geometric mean
      end
      ostep = step;
      % *RELATIVE* secant step size
      ddtdJ = odtdJ-dtdJ; 
      if ( ddtdJ~=0 ) nstep = dtdJ/ddtdJ; else nstep=1; end; % secant step size, guard div by 0
      nstep = sign(nstep)*max(opts.minStep,min(abs(nstep),opts.maxStep)); % bound growth/min-step size
      step  = step * nstep ;           % absolute step
      tstep = tstep + step;            % total step size      
   end
   if ( opts.verb > 2 ) fprintf('\n'); end;
   
   % Update the parameter values!
   % N.B. this should *only* happen here!
   w  = w + tstep*dw; 
   b  = b + tstep*db;

   % compute the other bits needed for CG iteration
   oR=R; ovarR=varR;
   switch ( RType ) % diff types regularisor
    case {1,2};
     if ( isequal(Rconst,0) ) Rg=w+Rconst(:); else Rg=w; end; % include effect of the constant in the regulisor
     nrm=abs(Rg);                  nrmeta=max(max(nrm)*opts.eta,nrm); 
     varR=R*(1./nrmeta)'/2;        dR=2*varR(:).*w;%dR=R(:).*sign(Rg); % prevents convergence testing issues..
     %dR=R(:).*sign(Rg);
    case 3;    
     nrm=sqrt((w.^2)'*R +Rconst(:)');  nrmeta=max(max(nrm)*opts.eta,nrm); 
     varR=R*(1./nrmeta)'/2;        dR=2*varR(:).*w;
   end
   % update the pre-conditioner
   odJ= dJ; % keep so can update pre-conditioner later if wanted...
   %odtdJ = d'*[2*C*varR.*w-X*Yerr';-sum(Yerr)]
   dL = -X*Yerr';
   dJ = [C*dR+dL;...
         -sum(Yerr)];
   MdJ= PC.*dJ;
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   
   % compute the function evaluation
   Ed   = -log(max(g,eps))*(Y.*wghtY);    % P(D|w,b,fp)
   switch ( RType ) % diff types regularisor % -ln P(w,b|R);
    case 1; Ew = R*sum(abs(w));
    case 2; Ew = R(:)'*abs(w);
    case 3; Ew = sum(nrm);
   end
   J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior
   % other useful information
   onFeat= nFeat; nFeat = sum(nrm>max(nrm).*opts.eta);
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] |x|=%2d J=%5f (%5f+%5f) |r2|=%8g |dJ|=%8g' lend],...
              iter,neval,w(1),w(2),nFeat,J,Ew,Ed,r2,madJ);
   end   

   if ( ~newC && ( J > oJ*(1.001) || isnan(J) ) ) % check for stuckness
      if ( opts.verb>=1 ) warning(sprintf('%d) Line-search Non-reduction - aborted',iter)); end;
      J=oJ; w=ow; b=ob; 
      wX   = w'*X;
      break;
   end;
   
   %------------------------------------------------
   % convergence test
   if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(r02,r2);
   elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,r2); % conv if smaller than best single step
   end
   madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
   if ( r2<=opts.tol(1) || ... % small gradient + numerical precision
        r2< r02*opts.tol0 || ... % Wolfe condn 2, gradient enough smaller
        neval > opts.maxEval || ... % abs(odtdJ-dtdJ) < eps || ... % numerical resolution
        madJ <= opts.objTol || madJ < opts.objTol0*dJ0 ) % objective function change
      break;
   end;    
   
   % pre-conditioner update
   condest=(2*varR(:))./wPCr;
   if ( opts.verb>=2 ) fprintf('%d) pc*varR=[%g,%g]\n',iter,max(condest),min(condest)); end;
   if ( max(condest)./min(condest) > opts.PCcondtol || ...
        mod(iter,ceil(nf/2))==0 )
     if ( 1 || opts.verb>=2 ) fprintf('%d) pc*varR=[%g,%g] -- update\n',iter,max(condest),min(condest)); end;
     wght=g(:).*(1-g(:));
     ddL= X2*(max(wght,.01)); % add a ridge to the hessian estimate
     wPCx=ddL;%(X.*X)*max(1e-2,(g(:).*(1-g(:)))); % H=X'*diag(g*(1-g))*X -> diag(H) = (X.^2)'*(g*(1-g)) 
     %bPC =1./sum(max(1e-2,g(:).*(1-g(:))));
     wPCr=2*varR(:);
     wPC =C*wPCr+wPCx;
     wPC(wPC<eps) = 1; 
     wPC=1./wPC;
     bPC=1./sum(wght);
     PC(1:end-1)=wPC;
     PC(end)=bPC;
     oMr = -PC.*odJ; 
     Mr  = -PC.*dJ;
     or2 = abs(oMr'*odJ); 
   end   

   %------------------------------------------------
   % conjugate direction selection
   % N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
   %      PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
   delta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   %delta = max(r2/or2,0); % Fletcher-Reeves
   d     = Mr+delta*d;     % conj grad direction
   dtdJ  = -d'*dJ;         % new search dir grad.
   if( dtdJ <= 0 )         % non-descent dir switch to steepest
     if ( opts.verb >= 1 ) fprintf('%d) non-descent dir\n',iter); end;      
     d=Mr; dtdJ=-d'*dJ; 
   end; 

    %-----------------------------------------------------
    % C search for correct number of features
    if ( ~isempty(opts.nFeat) ) 
      % est current number of active features
      newC  = false;
      if ( nFeat~=opts.nFeat && nFeat==onFeat && mod(iter,3)==0 ) % only when stable...
        wght=g(:).*(1-g(:));
        %dL = -X*Yerr';
        ddL= X2*(max(wght,.01)); % add a ridge to the hessian estimate
        dL0= dL-ddL.*w; % est loss gradient if this weight was set to 0
        if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
          %dR0 = abs(2.*varR.*w)+(abs(w)<mineta); % with the new R
          %dRg = (double(dRg)'*structMx)';
          %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
          %     norm of the weight change in the same direction then the component will grow.
          %     Thus we only need to see if the loss-gradient is bigger then 1
          %N.B. only if all elm in group have same weight in structMx
          dLRg= sqrt((double(dL0(:)).^2)'*istructMx);
        else
          dLRg = abs(dL0);%./R;
        end
        [sdLRg]=sort(dLRg,'descend');         
        estC=sdLRg(opts.nFeat);
        if ( nFeat>opts.nFeat && estC>C*(1+1e-2))     C=estC; newC=true;
        elseif ( nFeat<opts.nFeat && estC<C*(1-1e-2)) C=estC; newC=true;
        end
        if ( newC ) 
          dJ  = [C*dR - X*Yerr';...
                 -sum(Yerr)];
          MdJ = PC.*dJ;
          Mr  =-MdJ;
          d   = Mr; dtdJ=-d'*dJ; % reset to grad descent
          fprintf('%d) nF=%d C=%g estC=%g\n',iter,nFeat,C,estC); 
        end;
      end
    end
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

% compute the final performance with untransformed input and solutions
dv   = wX+b;
g    = 1./(1+exp(-Y'.*dv));         % Pr(x|y)
Ed   = -log(max(g,eps))*(Y.*wghtY); % -ln P(D|w,b,fp)
switch ( RType ) % diff types regularisor % -ln P(w,b|R);
 case 1; Ew = R*sum(abs(w));
 case 2; Ew = R(:)'*abs(w);
 case 3; Ew = sum(nrm);
end
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] |x|=%2d J=%5f (%5f+%5f) |r2|=%8g |dJ|=%8g\n'],...
          iter,neval,w(1),w(2),nFeat,J,Ew,Ed,r2,madJ);
end

% compute final decision values.
if ( all(size(X)==size(oX)) ) f=dv; else f   = w'*oX + b; end;
f = reshape(f,size(oY));
obj = [J Ew Ed];
wb=[w(:);b];
return;

%-----------------------------------------------------------------------
function [opts,varargin]=parseOpts(opts,varargin)
% refined and simplified option parser with structure flatten
i=1;
while i<=numel(varargin);  
   if ( iscell(varargin{i}) ) % flatten cells
      varargin={varargin{1:i} varargin{i}{:} varargin{i+1:end}};
   elseif ( isstruct(varargin{i}) )% flatten structures
      cellver=[fieldnames(varargin{i})'; struct2cell(varargin{i})'];
      varargin={varargin{1:i} cellver{:} varargin{i+1:end} };
   elseif( isfield(opts,varargin{i}) ) % assign fields
      opts.(varargin{i})=varargin{i+1}; i=i+1;
   else
      error('Unrecognised option');
   end
   i=i+1;
end
return;

%-----------------------------------------------------------------------------
function []=testCase()
%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([zeros(1,47) -1 0 zeros(1,47); zeros(1,47) 1 0 zeros(1,47); zeros(1,47) .2 .5 zeros(1,47)],[400 400 50],1,[],[-1 1 1]);[dim,N]=size(X);

%Make a chxtime test case
z=jf_mksfToy(); X=z.X; Y=z.Y;

tic,[wbl2,f,Jlr]=lr_cg(X,Y,50,'verb',1,'objTol0',1e-10);toc
tic,[wb,f,Jlr]=l1lr_cg(X,Y,5,'verb',1,'objTol0',1e-10);toc
clf;subplot(211);plot(log10(abs([wbl2(1:end-1) wb(1:end-1)])));subplot(212);plot([wbl2(1:end-1) wb(1:end-1)])

% test with group regulisor
if ( ndims(X)<3 ) 
  structMx=mkStructMx([4 size(X,1)/4],2); structMx=reshape(structMx,[size(X,1) size(structMx,3)]);
else 
  szX=size(X);
  structMx=mkStructMx(szX(1:end-1),1); 
end
tic,[wb,f,Jlr]=l1lr_cg(X,Y,50*structMx,'verb',1,'objTol0',1e-10);toc
tic,[wb,f,Jlr]=l1lr_cg(X,Y,50,'structMx',structMx,'verb',1,'objTol0',1e-10);toc
szX=size(X);W=reshape(wb(1:end-1),[szX(1:end-1) 1]);clf;subplot(211);imagesc(W);subplot(212);imagesc(log10(abs(W)))


% test with rotated group regularisor


% test with group constant (simulate cross-subject group l1 regularised)
tic,[wb,f,Jlr]=l1lr_cg(X,Y,50*structMx,'verb',1,'objTol0',1e-10,'Rconst',[10;zeros(size(structMx,ndims(structMx))-1,1)]);toc


% test with covariance and sensor selection
C=tprod(X,[1 -2 3],[],[2 -2 3]);
structMx=mkStructMx(size(C),'covCh');
tic,[wb,f,J]=l1lr_cg(C,Y,1*structMx,'verb',1);toc
tic,[wb,f,J]=l1lr_cg(C,Y,1,'structMx',structMx,'verb',1);toc
szX=size(C);W=reshape(wb(1:end-1),[szX(1:end-1) 1]);clf;subplot(211);imagesc(W);subplot(212);imagesc(log10(abs(W)))

% convex regions selection
structMx=mkStructMx(size(X,1),'ascend+descend');
tic,[wb,f,J]=l1lr_cg(X,Y,structMx*1000,'verb',1,'objTol0',1e-10);toc
clf;subplot(211);plot(wb(1:end-1));subplot(212);plot(log10(abs(wb(1:end-1))))

% test re-seeding for increasing Cs
Cs=2.^(1:7);
% without re-seeding
for ci=1:numel(Cs);
  [wbs(:,ci),f,Jlr]=l1lr_cg(X,Y,Cs(ci),'verb',1);
end
% with re-seeding
[wbs(:,1),f,Jlr]=l1lr_cg(X,Y,Cs(1),'verb',1,'wb',[]);
for ci=2:numel(Cs);
  [wbs(:,ci),f,Jlr]=l1lr_cg(X,Y,Cs(ci),'verb',1,'wb',wbs(:,ci-1));
end

% test with automatic regularisor search
tic,[wb,f,Jlr]=l1lr_cg(X,Y,5,'verb',1,'objTol0',1e-10);toc, sum(wb(1:end-1)>1e-3)
tic,[wb,f,Jlr]=l1lr_cg(X,Y,-57,'verb',1,'objTol0',1e-10);toc, sum(wb(1:end-1)>1e-3)
clf;subplot(211);plot(log10(abs([wbl2(1:end-1) wb(1:end-1)])));subplot(212);plot([wbl2(1:end-1) wb(1:end-1)])

% and the group version
structMx=mkStructMx([4 size(X,1)/4],2); % time-point selection
structMx=mkStructMx([4 size(X,1)/4],1); % channel selection
tic,[wb,f,Jlr]=l1lr_cg(X,Y,500,'verb',1,'objTol0',1e-10,'structMx',structMx);toc, sum(wb(1:end-1)>1e-3)
tic,[wb,f,Jlr]=l1lr_cg(X,Y,-8,'verb',1,'objTol0',1e-10,'structMx',structMx);toc, sum(wb(1:end-1)>1e-3)
% group version where we weight different groups differently...
structMx=mkStructMx([4 size(X,1)/4],2); structMx=reshape(structMx,[],size(structMx,3));% time-point selection
structMx(:,1:10)=structMx(:,1:10)/10;
tic,[wb,f,Jlr]=l1lr_cg(X,Y,500,'verb',1,'objTol0',1e-10,'structMx',structMx);toc, sum(wb(1:end-1)>1e-3)
tic,[wb,f,Jlr]=l1lr_cg(X,Y,-8,'verb',1,'objTol0',1e-10,'structMx',structMx);toc, sum(wb(1:end-1)>1e-3)


% and with overlapping groups
structMx=mkStructMx(size(X,1),'ascend');