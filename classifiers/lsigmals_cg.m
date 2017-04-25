function [wb,f,J,obj,tstep]=lsigmals_cg(X,Y,C,varargin);
% trace norm Regularised linear Logistic Regression Classifier
%
% [wb,f,J,obj]=lsigmals_cg(X,Y,C,varargin)
% trace-norm Regularised Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J = C*\sum eig(w) + sum_i (y_i = w'*X_i+b).^2
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
%  C       - trace norm regularisation constant (0)
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
%  eta     - [float] relative smoothing constant for quadratic approx to l1 loss   (1e-3)
%                   thus, anything <eta*max(nrm) has been effectively set to 0.

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
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-8,'objTol',0,'objTol0',1e-3,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'bPC',[],'wPC',[],'incThresh',.75,'optBias',1,'maxTr',inf,...
              'getOpts',0,'eta',1e-3,'zeroStart',0,'eigDecomp',0,'PCcondtol',1,'method','PR','initSoln','proto');
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
szX=size(X); nd=numel(szX); N=szX(end); nf=prod(szX(1:end-1));
Y=Y(:); % ensure Y is col vector

% Ensure all inputs have a consistent precision
if(isa(X,'double') & isa(Y,'single') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence
opts.tol(end+1:2)=opts.tol(1);
if ( isempty(opts.maxEval) ) opts.maxEval=5*max(prod(szX(1:end-1)),szX(end)); end

if ( opts.eigDecomp && szX(1)~=szX(2) ) error('Cant do eig with non-symetric feature inputs'); end;

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);

% check for degenerate inputs
if ( all(Y>=0) || all(Y<=0) )
  warning('Degnerate inputs, 1 class problem');
end

% N.B. this form of loss weighting has no true probabilistic interpertation!
wght=opts.wght;
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
else
  wght=1;
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
  W=zeros(szX(1:2));b=0;
  if ( ~opts.zeroStart ) 
    if ( isequal(opts.initSoln,'lr') )
      wb=lr_cg(X,Y,10*C,'objTol0',1e-3);
      W = reshape(wb(1:end-1),szX(1:2)); b=wb(end);
    else
      alpha=zeros(numel(Y),1);
      alpha(Y>0)=.5./sum(wght.*Y(Y>0)); alpha(Y<0)=.5./sum(wght.*Y(Y<0)); 
      W = reshape(X*alpha,szX(1:2)); b=0;
    end
    % initial scaling and bias estimate
    wX    = W(:)'*X;
    %muP=mean(wX(Y>0)); muN=mean(wX(Y<=0));
    %m = (2/abs(muP-muN)); b=-.5*(muP+muN)*m;
    %W = W*m; wX=wX*m;
    if ( opts.eigDecomp )
      [U,s]  =eig(W); V=U;  s=diag(s); 
    else
      [U,s,V]=svd(W,'econ'); s=diag(s);      
    end
    nrms=abs(s);
    nrmseta=max(max(nrms)*opts.eta,nrms);
    if (szX(1)<=szX(2))
      varR= U*diag(1./nrmseta)*U'/2; % leading dim
      wRw = 2*W(:)'*reshape(varR*W,[numel(W),1]);
    else
      varR= V*diag(1./nrmseta)*V'/2; % leading dim
      wRw = 2*reshape((W'*varR),[numel(W),1])'*W(:);
    end
    % find least squares optimal scaling and bias
    sb = pinv([wRw+wX*wX' sum(wX); sum(wX) sum(Y~=0)])*[wX*Y; sum(wght.*Y)];
    W=W*sb(1); s=s*sb(1); b=sb(2)/2; 
  else
    if ( opts.eigDecomp )
      [U,s]  =eig(W); V=U;  s=diag(s); 
    else
      [U,s,V]=svd(W,'econ'); s=diag(s);      
    end
  end
else
  W=reshape(wb(1:end-1),szX(1:2)); b=wb(end);
  if ( opts.eigDecomp )
    [U,s]  =eig(W); V=U;  s=diag(s); 
  else
    [U,s,V]=svd(W,'econ'); s=diag(s); 
  end
  nrms=abs(s);
end 

nrms=abs(s);
eta=max(nrms)*min(1,opts.eta*1e2); % start not really trusting the nrms on the current solution
nrmseta=max(eta,nrms);
if (szX(1)<=szX(2)) % N.B. check this is right way round!!!  
  varR=U*diag(1./nrmseta)*U'/2; % leading dim
  dR  =2*varR*W;
else
  varR=V*diag(1./nrmseta)*V'/2; % leading dim
  dR  =2*W'*varR;
end
wX   = W(:)'*X;
dv   = wX'+b;
err  = Y-dv;
werr = wght.*err;

% set the pre-conditioner
% N.B. the Hessian for this problem is:
%  H  =[2*X*wght*X'+ddR  2*wght'*X;...
%       (2*wght'*X)'     sum(wght)];
%  so diag(H) = [sum(X.*X,2)*wght+2*diag(R);sum(wght)];
wPC  = opts.wPC; bPC=opts.bPC;
if ( isempty(wPC) ) 
  if ( numel(wght)>1 )
    wPCx=tprod(X,[1 -2],[],[1 -2])*wght(:); 
  else
    wPCx=X(:)'*X(:)*wght;
  end
  wPC =wPCx;
  wPCr=nrmseta;
  % approx pre-conditioner ... can do better! by applying the reg-pc and loss-pc separately
  if ( 0 ) 
    wPCx2=mean(abs(U'*reshape(wPCx,size(W))),2);
    wPC=U*diag(1./(C./nrmseta+wPCx2))*U';
  else
    wPC=U*diag(nrmseta)*U'/C/2;
  end
  if ( 1 ) 
    if (szX(1)<=szX(2))   tmp=repmat(diag(varR) ,1,nf./size(varR,1)); % leading dim
    else                  tmp=repmat(diag(varR)',nf./size(varR,1),1); % trailing dim
    end  
    wPC = wPCx + 2*tmp(:)*C;
    wPC(wPC<eps) = 1; 
    wPC=1./wPC;
  end
end;
if ( isempty(bPC) ) % Q: Why this pre-conditioner for the bias term?
  if ( numel(wght)>1 )
    bPC=1./sum(wght);
  else
    bPC=1./size(X,2);
  end
end % ==mean(diag(cov(X)))
if ( size(wPC,2)==1 ) PC = [wPC(:);bPC]; else PC=bPC; end;
 
dJ   = [C*dR(:) - X*werr; ...
        -sum(werr)];
% precond'd gradient:
%  [H  0  ]^-1 [ dR-X'*(y-f)] 
%  [0  bPC]    [   -1'*(y-f)] 
MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d'*dJ);
r2   = dtdJ;

Ed   = err'*werr;
Ew   = sum(abs(s));
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
%if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,s(1),s(2),J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
W0=W; b0=b;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

  oJ= J; oMr  = Mr; or2=r2; oW=W; ob=b; % record info about prev result we need

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
   dw  = reshape(d(1:end-1),size(W)); db=d(end);
   %if ( opts.eigDecomp ) dw=.5*(dw+dw'); end;
   dX  = dw(:)'*X;
   % N.B. R = trace(D'*varR*W) = \sum_i,j D.*(varR*W) = D(:)'*reshape(varR*W,[nf,1]) 
   %      R = trace(D'*varR*W) = \sum_i,j (D'*varR).*W = reshape((D'*varR)',[nf,1])'*W(:) 
   %                           = reshape(varR*D,[nf,1])'*W(:) % for symetric varR
   if (szX(1)<=szX(2)) % N.B. check this is right way round!!!     
     dvarR = (dw'*varR)'; % == varR*dw (because varR=varR' by construction
     dvarRw= dvarR(:)'*W(:); 
     dvarRd= dvarR(:)'*dw(:);
   else % BODGE: Hmmm, not sure if this is actually correct!
     dvarR = (varR*dw)';
     dvarRw= dvarR(:)'*W(:); 
     dvarRd= dvarR(:)'*dw(:);
   end
   % initial values
   dtdJ  = -(2*C*dvarRw - dX*werr - db*sum(werr));
   if ( opts.verb > 2 )
     Ed   = err'*werr; 
     Ew   = 2*reshape(varR*W,[numel(W),1])'*W(:); % P(w,b|R);
     J    = Ed + C*Ew + opts.Jconst;              % J=neg log posterior
     fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ew,Ed); 
   end
   odtdJ=dtdJ;      % one step before is same as current
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      wX    = wX0+tstep*dX;%w'*X;
      err   = Y-(wX'+(b+tstep*db));
      werr  = wght.*err;
      dtdJ  = -(2*C*(dvarRw+tstep*dvarRd) - dX*werr - db*sum(werr));
      %fprintf('.%d step=%g ddR=%g ddgdw=%g ddgdb=%g  sum=%g\n',j,tstep,2*(dRw+tstep*dRd),-dX*Yerr',-db*sum(Yerr),-dtdJ);
      
      if ( opts.verb > 2 )
        Ed   = err'*werr;
         Ew   = 2*(reshape(varR*W,[numel(W),1])'*W(:)+tstep*2*dvarRw+tstep.^2*dvarRd); % P(w,b|R);           
         J    = Ed + C*Ew + opts.Jconst;              % J=neg log posterior
         Wp   = W+tstep*dw;
         Ew2  = 2*reshape(varR*Wp,[numel(W),1])'*Wp(:); % P(w,b|R);
         fprintf('.%d %g=%g @ %g (%g+%g)\n',j,tstep,dtdJ,J,Ew,Ed); 
         %fprintf('.%d %g / %g\n',j,Ew,Ew2);
         if ( opts.verb > 3 ) 
            plot(tstep,dtdJ,'*'); text(double(tstep),double(dtdJ),num2str(j));
         end
      end;

      % convergence test, and numerical res test
      if(iter>1||j>2) % Ensure we do decent line search for 1st step size!
         if ( abs(dtdJ) < opts.lstol0*abs(dtdJ0) || ... % Wolfe 2, gradient enough smaller
              abs(dtdJ*step) <= opts.tol(2) )              % numerical resolution
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
   W  = W + tstep*dw; 
   if( opts.eigDecomp ) W=.5*(W+W'); end; % enforce symetry of the matrix
   b  = b + tstep*db;

   % update the information for the variational approx to the regularisor
   os=s; 
   % compute the new basis set and scaling
   if ( 1 )
     if ( opts.eigDecomp )
       [U,s]  =eig(W); V=U;  s=diag(s); 
     else
       [U,s,V]=svd(W,'econ'); s=diag(s);      
     end   
   else % update the scaling using the previous basis
     s=tprod(U'*W,[1 -1],V,[-1 1]); nrms=abs(s);
     %nrms=sqrt(sum((W'*U).^2)); s=nrms; % N.B. s now has lost any sign information it previously had!
   end
   nrms=abs(s);
   eta=max(nrms)*opts.eta; % adapt smoothing factor based on the current solution
   nrmseta=max(eta,nrms);
   if (szX(1)<=szX(2)) % N.B. check this is right way round!!!
     varR=U*diag(1./nrmseta)*U'/2; % update the variational l1 approx
     dR  =2*varR*W;
   else
     varR=V*diag(1./nrmseta)*V'/2; % leading dim
     dR  =2*W'*varR;
   end

   % compute the other bits needed for CG iteration
   odJ= dJ; % keep so can update pre-conditioner later if wanted...
   dJ = [C*dR(:) - X*werr;...
         -sum(werr)];
   if ( size(wPC,2)>1 ) 
     MdJ(1:end-1) = wPC*reshape(dJ(1:end-1),size(W));
     MdJ(end)     = bPC*dJ(end);
   else
     MdJ= PC.*dJ;     
   end
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   
   % compute the function evaluation
   Ed   = err'*werr; 
   Ew   = sum(abs(s));
   J    = Ed + C*Ew + opts.Jconst; 
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,s(1),s(2),J,Ew,Ed,madJ);
   end   

   if ( iter>2 && (J > oJ*(1.001) || isnan(J)) ) % check for stuckness
      if ( opts.verb>=1 ) warning(sprintf('%d) Line-search Non-reduction - aborted',iter)); end;
      J=oJ; W=oW; b=ob; 
      wX   = W(:)'*X;
      break;
   end;
   
   %------------------------------------------------
   % convergence test
   if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(r02,r2);
   elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,r2); % conv if smaller than best single step
   end
   madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
   if ( r2<=opts.tol(1) || ... % small gradient + numerical precision
        r2< r02*opts.tol0(1) || ... % Wolfe condn 2, gradient enough smaller
        neval > opts.maxEval || ... % abs(odtdJ-dtdJ) < eps || ... % numerical resolution
        madJ <= opts.objTol(1) || madJ < opts.objTol0(1)*dJ0 ) % objective function change
      break;
   end;    
      
   %------------------------------------------------
   % pre-conditioner update
   condest=nrmseta(:)./wPCr(:); condest=max(condest)./min(condest);%N.B. ignores effects due to rotation of the basis!
   if ( iter>1 ) condest=abs(sum(sum(abs(wPC*2*C*varR)))-size(varR,1)); end;
   if ( opts.verb>=2 ) 
     fprintf('%d) pc*varR=[%g,%g]\n',iter,max(condest),min(condest));
   end;
   if ( condest > opts.PCcondtol || ...
        mod(iter,ceil(nf/2))==0 )   
     if ( opts.verb>=2 ) 
       if ( opts.verb<2 ) fprintf('%d) pc*varR=[%g,%g] ',iter,max(condest),min(condest)); end;
       fprintf('%d) pc update\n',iter); 
     end;
     % approx pre-conditioner ... can do better! by applying the reg-pc and loss-pc separately
     wPCx=(X.*X)*werr;
     wPCr=nrmseta; % used to decide when to update the pre-conditioner
     if ( 0 )
       wPCx2=mean(abs(U'*reshape(wPCx,size(W))),2); % HACK: Hmmm, why doesn't this work????
       wPC=U*diag(1./(C./nrmseta+wPCx2))*U';
     else
       wPC=U*diag(nrmseta)*U'/C/2;
     end
     PC=PC(end);
     oMr(1:end-1) = -wPC*reshape(odJ(1:end-1),size(W)); 
     Mr(1:end-1)  = -wPC*reshape(dJ(1:end-1),size(W));       
     or2 = abs(oMr'*odJ); 
   end   
         
   %------------------------------------------------
   % conjugate direction selection
   % N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
   %      PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
   if ( isequal(opts.method,'PR') )
     delta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   else
     delta = max(r2/or2,0); % Fletcher-Reeves -- seems better for this case
   end
   d     = Mr+delta*d;     % conj grad direction
   dtdJ  = -d'*dJ;         % new search dir grad.
   if( dtdJ <= 0 )         % non-descent dir switch to steepest
     if ( opts.verb >= 1 ) fprintf('%d) non-descent dir\n',iter); end;      
     d=Mr; dtdJ=-d'*dJ; 
   end;      
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   W=W0; b=b0;
end;

% compute the final performance with untransformed input and solutions
dv   = wX'+b;
err  = Y-dv;
Ed   = err'*(wght.*err);
Ew   = sum(abs(s));
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
   fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,s(1),s(2),J,Ew,Ed,madJ);
end

% compute final decision values.
if ( all(size(X)==size(oX)) ) f=dv; else f   = w'*oX + b; end;
f = reshape(f,size(oY));
obj = [J Ew Ed];
wb=[W(:);b];
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
% non-sym pos def matrices
wtrue=randn(40,50); [utrue,strue,vtrue]=svd(wtrue,'econ'); strue=diag(strue);
% sym-pd matrices
wtrue=randn(40,500); wtrue=wtrue*wtrue'; [utrue,strue]=eig(wtrue); strue=diag(strue); vtrue=utrue;
% re-scale components and make a dataset from it
strue=sort(randn(numel(strue),1).^2,'descend');
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
noise=randn([size(wtrue),size(Xtrue,3)])*sum(strue)/10; 
if(size(Xtrue,1)==size(Xtrue,2))noise=tprod(noise,[1 -2 3],[],[2 -2 3]); end;
X    =Xtrue + noise;
wb0  =randn(size(X,1),size(X,2));

%Make a chxtime test case
z=jf_mksfToy(); X=z.X; Y=z.Y;
% sym-pd
X=tprod(z.X,[1 -2 3],[],[2 -2 3]);

% simple l2
tic,[wbl2,f,J]=lr_cg(X,Y,1,'verb',1);toc

%low rank
tic,[wb,f,J]=lsigmals_cg(X,Y,2000,'verb',1);toc
tic,[wb,f,J]=lsigmals_rereg(X,Y,2000,'verb',1);toc
szX=size(X);W=reshape(wb(1:end-1),szX(1:2)); [U,s,V]=svd(W); s=diag(s);
szX=size(X);W=reshape(wb(1:end-1),szX(1:2)); [U,s]=eig(W); V=U; s=real(diag(s));
clf;subplot(221);plot(s);subplot(222);plot(log10(abs(s)+eps));subplot(212);imagesc(W);


% test with re-seeding solutions
Cscale=.1*sqrt(CscaleEst(X,2));
[wb10,f,J] =lsigmals_cg(X,Y,Cscale*2.^4,'verb',1,'wb',[]);  
[wb,f,J]=lsigmals_cg(X,Y,Cscale*2.^3,'verb',1);  
[wb102,f,J]=lsigmals_cg(X,Y,Cscale*2.^3,'verb',1,'wb',wb10);  

% test the symetric version
tic,[wb,f,J] =lsigmals_cg(X,Y,10,'verb',1);toc
tic,[wbs,f,J]=lsigmals_cg(X,Y,10,'verb',1,'eigDecomp',1);toc

% compare on real dataset
jf_cvtrain(z,'objFn','lsigmals_cg','Cs',5.^(-3:5),'reorderC',0,'outerSoln',0,'seedNm','wb')




% test the numerics of the variational approx to the gradient
Ut=orth(randn(10,10)); st=(10.^(rand(10,1)*10)).*sign(randn(10,1));st=st./sum(abs(st)); % high variance in eigenvalues
W=Ut*diag(st)*Ut';

if ( size(W,1)==size(W,2) ) 
  [U,s]=eig(W);s=diag(s); V=U;
else
  [U,s,V]=svd(W,'econ'); s=diag(s);
end
varR=(U*diag(1./abs(s))*U');
trace(W*(U*diag(1./abs(s))*U')*W')
trace(W'*(U*diag(1./abs(s))*U')*W)

% N.B. for symetric W these are all equivalent!
reshape(W'*varR,[numel(W),1])'*W(:)
W(:)'*reshape(varR*W,[numel(W),1])  % true in all cases!
reshape((W'*varR)',[numel(W),1])'*W(:) % true in all cases!
reshape(W*varR,[numel(W),1])'*W(:)
reshape(varR*W',[numel(W),1])'*W(:)

% for non-symetric W only this is true...
D=randn(size(W));
mad(D(:)'*reshape(varR*W,[numel(W),1]),trace(D'*(varR*W)))
mad(reshape((D'*varR)',[numel(W),1])'*W(:),trace(D'*varR*W))
