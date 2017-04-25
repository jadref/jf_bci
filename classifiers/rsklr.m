function [wb,J]=klr(K,Y,C,beta,varargin);
% [alphab,J]=krl(K,Y,lambda,varargin)
% Regularised Kernel Logistic Regression using a pre-conditioned conjugate
% gradient solver so extends to large input kernels.
%
% J = C(1) w' K w + sum_i log( (1 + exp( - y_i ( w'*K_i + b ) ) )^-1 ) 
%
% Inputs:
%  K       - NxN kernel matrix
%  Y       - Nx1 matrix of +1,-1 labels
%  C       - the regularisation parameter
%  beta    - indices (in K) of the points to use for solution construction
%
% Outputs:
%  alphab  - (|beta|+1)x1 matrix of the kernel weights and the bias b
%  J       - the final objective value
%
% Options:
%  alphab  - initial guess at the kernel parameters, [alpha;b]
%  nobias  - flag we don't want the bias computed (x2 faster!)
%  maxIter - max number of CG steps to do
%  tol     - absolute error tolerance
%  tol0    - relative error tolerance, w.r.t. initial gradient
%  verb    - verbosity
%  step    - initial step size guess
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
if ( nargin < 3 ) C(1)=0; end;
if ( nargin < 4 ) beta=[]; end
opts=struct('alpha',[],'nobias',0,'maxIter',inf,'tol',1e-6,'tol0',0,...
            'verb',0,'step',0,'wght',[],'X',[],'ridge',1e-9);
if(isstruct(varargin{1})) % struct->cell
   varargin=[fieldnames(varargin{1})'; struct2cell(varargin{1})'];
end
for i=1:2:numel(varargin); % process the options 
   if( ~isfield(opts,varargin{i}) ) error('Unrecognised option'); end;
   opts.(varargin{i})=varargin{i+1};
end

[dim,N]=size(K);

if ( islogical(beta) ) beta=find(beta);  
elseif ( isempty(beta) ) beta=int32(1:N); end; 
% get set and logical version beta
beta=int32(beta); ibeta=false(1,N); ibeta(beta)=true; M=numel(beta);

% N.B. adding this ridge means we can't guarantee convergence below 1e-8
if ( M==N ) % K is NxN so we don't need the inverse
   ;
elseif ( dim==N ) % assume K is N x N
   invKbb = inv(K(beta,beta)+eye(numel(beta))*opts.ridge); 
elseif ( dim==M ) % assume K is beta x N
   invKbb = inv(K(:,beta)+eye(numel(beta))*opts.ridge); 
else 
   error('K must be NxN or beta x N');
end

wb=opts.alpha; if ( isempty(wb) ) wb=zeros(numel(beta)+1,1); end 
if ( opts.nobias ) wb(end)=0; end;

if( dim==M ) wK   = wb(1:end-1)'*K; else   wK   = wb(1:end-1)'*K(beta,:); end
wK(beta)= wK(beta)+opts.ridge*wb(1:end-1)';
g    = 1./(1+exp(-Y'.*(wK+wb(end)))); g=max(g,eps); % stop log 0

Yerr   = Y'.*(1-g);
% precond'd gradient K^-1 (lambda*wK-K((1-g).Y)) = lambda w - (1-g).Y
MdJ   = [(C(1)*2*wb(1:end-1) - Yerr(beta)');... 
         -sum(Yerr) ];
if(M==N) % not reduced-set
   % non-pcond gradient
   dJ    = [K*MdJ(1:end-1)+opts.ridge*MdJ(1:end-1); ...
            MdJ(end)];
elseif(dim==N) % K is N X N   
   % include non-beta bits
   MdJ(1:end-1)=MdJ(1:end-1)- invKbb*K(beta,~ibeta)*Yerr(~ibeta)';         
   dJ  = [(2*C(1)*K(beta,beta)*wb(1:end-1) + 2*C(1)*opts.ridge*wb(1:end-1) ...
           - K(beta,:)*Yerr' - opts.ridge*Yerr(beta)');...
          MdJ(end)];
elseif(dim==M) % K is beta x N
   % include the non-beta bits
   MdJ(1:end-1)=MdJ(1:end-1)- invKbb*K(:,~ibeta)*Yerr(~ibeta)';         
   dJ  = [(2*C(1)*K(:,beta)*wb(1:end-1) + 2*C(1)*opts.ridge*wb(1:end-1) ...
           - K*Yerr' - opts.ridge*Yerr(beta)');...
          MdJ(end)];
end
Mr    =-MdJ;
d     = Mr;
ddJ   =-d'*dJ;
r2    = ddJ;
r02   = r2;

Ed   = -sum(log(g));       % -ln P(D|w,b,fp)
Ew   = wK(:,beta)*wb(1:end-1);             % -ln P(w,b|R);
J    = Ed + C(1)*Ew;       % J=neg log posterior

step=opts.step;
if( step<=0 ) step=abs(J/ddJ); end  % init step assuming opt is at 0
step=abs(step); tstep=step;

neval=1;
if(opts.verb>0)   % debug code      
   fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],0,neval,wb(1),wb(2),J,r2);
end

% pre-cond non-lin CG iteration
for i=1:opts.maxIter;

   oJ= J; oMr  = Mr; or2=r2; % record info about prev result we need

   % Secant method for the root search.
   if ( opts.verb > 1 )
     fprintf('%g=%g \n',0,ddJ);hold on;plot(0,ddJ,'r*');text(0,ddJ,num2str(0));
   end;
   step=tstep; % prev step size is first guess!
   oddJ=ddJ; % one step before is same as current
   wb = wb + step*d;
   for j=1:20;
      ooddJ=oddJ; oddJ=ddJ; % prev and 1 before grad values
      
      % Eval the gradient at this point.  N.B. only gradient needed for secant
      wK   = wb(1:end-1)'*K(beta,:);
      wK(beta)= wK(beta)+opts.ridge*wb(1:end-1)';
      g    = 1./(1+exp(-Y'.*(wK+wb(end)))); g=max(g,eps); % stop log 0

      Yerr   = Y'.*(1-g);
      % precond'd gradient K^-1 (lambda*wK-K((1-g).Y)) = lambda w - (1-g).Y
      MdJ   = [(C(1)*2*wb(1:end-1) - Yerr(beta)');... 
               -sum(Yerr) ];
      if(M==N) % not reduced-set
         % non-pcond gradient
         dJ    = [K*MdJ(1:end-1)+opts.ridge*MdJ(1:end-1); ...
                  MdJ(end)];
      elseif(dim==N) % K is N X N   
         % include non-beta bits
         MdJ(1:end-1)=MdJ(1:end-1)- invKbb*K(beta,~ibeta)*Yerr(~ibeta)';
         dJ  = [(2*C(1)*K(beta,beta)*wb(1:end-1) ...
                 + 2*C(1)*opts.ridge*wb(1:end-1) ...
                 - K(beta,:)*Yerr' - opts.ridge*Yerr(beta)');...
                MdJ(end)];
      elseif(dim==M) % K is beta x N
         % include the non-beta bits
         MdJ(1:end-1)=MdJ(1:end-1)- invKbb*K(:,~ibeta)*Yerr(~ibeta)';         
         dJ  = [(2*C(1)*K(:,beta)*wb(1:end-1) ...
                 + 2*C(1)*opts.ridge*wb(1:end-1) ...
                 - K*Yerr' - opts.ridge*Yerr(beta)');...
                MdJ(end)];
      end
      if ( opts.nobias ) MdJ(end)=0; dJ(end)=0; end
      ddJ   =-d'*dJ;  % gradient along the line

      % convergence test, and numerical res test
      if ( abs(ddJ) <= opts.tol || abs(ddJ*step)<eps ) break; end; 
      
      if ( opts.verb > 1 )
         fprintf('%g=%g\n',tstep,ddJ); 
         hold on; plot(tstep,ddJ,'*'); text(tstep,ddJ,num2str(j));
      end;
      % now compute the new step size
      % backeting check, so it always decreases
      if ( ooddJ*oddJ < 0 && oddJ*ddJ > 0 ...          % ooddJ still brackets
           && abs(step*oddJ/(oddJ-ddJ)) > abs(ostep) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         oddJ = ooddJ;
      end
      ostep=step;
      step = step * ddJ / (oddJ-ddJ); % *RELATIVE* secant step size
      tstep= tstep + step;
      
      % move to the new point
      wb    = wb + step*d ;
      neval=neval+1;
      
   end
   if ( opts.verb > 1 ) fprintf('\n'); end;

   % compute the other bits needed for CG iteration
   Mr    =-MdJ;
   r2    =abs(Mr'*dJ);

   
   if(opts.verb>0)   % debug code      
      % compute the function evaluation
      Ed   = -sum(log(g));               % P(D|w,b,fp)
      Ew   = wK(:,beta)*wb(1:end-1);     % P(w,b|R);
      J    = Ed + C(1)*Ew;               % J=neg log posterior
      
      fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],...
              i,neval,wb(1),wb(2),J,r2);
   end   
   if ( r2<=opts.tol || r2<=r02*opts.tol0 ) break; end; % term tests

   %delta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   delta = max(r2/or2,0); % Fletcher-Reeves
   d    = Mr+delta*d;  % conj grad direction
   ddJ  =-d'*dJ;      % gradient along the line   
   if( ddJ >= 0 ) d=Mr; ddJ=-d'*dJ; end;% if non-descent dir switch to steepest
end;

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);

[dim,N]=size(X);K=X'*X; % N.B. add 1 to give implicit bias term
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
trnSet=find(trnIdx);

beta=randperm(sum(trnIdx));beta=beta(1:ceil(end/8));beta=int32(sort(beta));
[alphab,J]=rsklr(K(trnIdx,trnIdx),Y(trnIdx),1,beta,'verb',1);

dv=K(tstIdx,trnSet(beta))*alphab(1:end-1)+alphab(end);
dv2conf(dv,Y(tstIdx))

% for linear kernel
alpha=zeros(N,1);alpha(trnSet(beta))=alphab(1:end-1); % equiv alpha
plotLinDecisFn(X,Y,X(:,trnSet(beta))*alphab(1:end-1),alphab(end),alpha);

% for linear kernel
plotLinDecisFn(X(:,trnIdx),Y(trnIdx),X(:,trnIdx)*alphab(1:end-1),alphab(end),alphab(1:end-1));

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
