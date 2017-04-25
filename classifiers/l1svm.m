function [wb,J]=svm(K,Y,C,varargin);
% [alphab,J]=svm(K,Y,C,varargin)
% Linear Loss Support Vector machine using a pre-conditioned conjugate
% gradient solver so extends to large input kernels.
%
% J = C(1) w' K w + sum_i max(0 , 1 - y_i ( w'*K_i ) ) 
% 
% Inputs:
%  K       - NxN kernel matrix
%  Y       - Nx1 matrix of +1,-1 labels
%  C       - the regularisation parameter
%
% Outputs:
%  alphab  - (N+1)x1 matrix of the kernel weights and the bias b
%  J       - the final objective value
%
% Options:
%  alphab  - initial guess at the kernel parameters
%  maxIter - max number of CG steps to do
%  tol     - absolute error tolerance
%  tol0    - relative error tolerance, w.r.t. initial gradient
%  verb    - verbosity
%  step    - initial step size guess
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
%  h       - huber L1 interp width
if ( nargin < 3 ) C=0; end;
opts=struct('alphab',[],'maxIter',inf,'tol',1e-3,'tol0',1e-9,'verb',0,...
            'step',0,'wght',[],'h',.3,'X',[]);
for i=1:2:numel(varargin); % process the options 
   if ( ~isstr(varargin{i}) || ~isfield(opts,varargin{i}) )
      error('Unrecognised option');
   else
      opts.(varargin{i})=varargin{i+1};
   end
end
[N,dim]=size(K);

wb=opts.alphab;
if ( isempty(wb) ) % KFDA seed -- doesn't help!
   wb=zeros(N+1,1);
   %wb(Y>0)=1/sum(Y>0);wb(Y<=0)=-1/sum(Y<=0);wb(end)=wb(1:end-1)'*K*wb(1:end-1);
end 
%wb=opts.alphab; if ( isempty(wb) ) wb=zeros(N+1,1); end 


h = opts.h;
wK     = wb(1:end-1)'*K;
err    = 1-Y'.*(wK+wb(end)); l1svs=err>=h; svs=abs(err)<h;
errph  = err+h;
Ed     = sum(errph(svs).^2)/4/h + sum(err(l1svs));  
Ew     = wK*wb(1:end-1)/2;
J      = Ed + C(1)*Ew; % SVM objective

% pre-condinationed gradient, 
Yerr = Y'.*errph; % Y.*(1-Y.*f)
% K^-1*dJdw = K^-1(C(1)Kw - 2 K I_sv(Y-f) / 4h - K I_1sv Y) 
%           = C(1)w - Isv (Y-f) / 2h - I_1sv Y
% N.B. include additional diag(K) pre-conditioner to balance with b terms
MdJ  = [(C(1)*wb(1:end-1) - (Yerr.*svs)'/2/h - (Y'.*l1svs)')./diag(K);...
        -(sum(Yerr(svs))/2/h+sum(Y(l1svs)))/N ];
dJ   = [K*(MdJ(1:end-1).*diag(K)); MdJ(end)*N ];
Mr   =-MdJ;
d    = Mr;
ddJ  =-d'*dJ;
r2   = ddJ;
r02  = r2;

step=opts.step;
if( step<=0 ) step=abs(J/ddJ); end  % init step assuming opt is at 0
step=abs(step); tstep=step;

neval=1;
if(opts.verb>0)   % debug code      
   fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],0,neval,wb(1),wb(2),J,r2);
end

% alt pre-cond non-lin CG iteration
oddJ=ddJ;
for i=1:opts.maxIter;

   oJ= J; oMr  = Mr; or2=r2; % record info about prev result we need

   % Secant method for the root search.
   if ( opts.verb > 1 )
    fprintf('%g=%g @ %g\n',0,ddJ,J);
    if ( opts.verb>2 ) hold on;plot(0,ddJ,'r*');text(0,ddJ,num2str(0));end
   end;
   step=tstep; % prev step size is first guess!
   oddJ=ddJ; % one step before is same as current
   wb = wb + step*d;
   for j=1:50;
      ooddJ=oddJ; oddJ=ddJ; % prev and 1 before grad values
      
      % Eval the gradient at this point.  N.B. only gradient needed for secant
      wK     = wb(1:end-1)'*K;
      err    = 1-Y'.*(wK+wb(end)); l1svs=err>=h; svs=abs(err)<h;
      errph  = err+h;
      Yerr   = Y'.*errph; % Y.*(1-Y.*f)
      MdJ  = [(C(1)*wb(1:end-1) - (Yerr.*svs)'/2/h - (Y'.*l1svs)')./diag(K);...
              -(sum(Yerr(svs))/2/h+sum(Y(l1svs)))/N ]; % pre-cond grad
      dJ   = [K*(MdJ(1:end-1).*diag(K)); MdJ(end)*N ]; % true grad
      ddJ    =-d'*dJ;
      
      if ( opts.verb > 1 )
         fprintf('%g=%g @ %g \n',tstep,ddJ,J); 
         if ( opts.verb > 2 )
            hold on; plot(tstep,ddJ,'*'); text(tstep,ddJ,num2str(j));
         end;
      end;
      % now compute the new step size
      % backeting check, so it always decreases
      if ( ooddJ*oddJ < 0 && oddJ*ddJ > 0 ...          % ooddJ still brackets
           && abs(step*oddJ/(oddJ-ddJ)) > abs(ostep) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         oddJ = ooddJ;
      end
      ostep = step;
      step  = step * ddJ / (oddJ-ddJ); % *RELATIVE* secant step size
      tstep= tstep + step;            % total step size
      
      % move to the new point
      wb    = wb + step*d ;
      neval=neval+1;
      
      % convergence test, and numerical res test
      if ( abs(ddJ) <= opts.tol || abs(ddJ*step)<eps ) break; end; 
   end
   if ( opts.verb > 1 ) fprintf('\n'); end;
   % compute the function evaluation
   Ed     = sum(errph(svs).^2)/4/h + sum(err(l1svs));  
   Ew     = wK*wb(1:end-1)/2;
   J      = Ed + C(1)*Ew; % SVM objective

   % compute the other bits needed for CG iteration
   Mr    =-MdJ;
   r2    =abs(Mr'*dJ);

   if(opts.verb>0)   % debug code      
    fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],i,neval,wb(1),wb(2),J,r2);
   end   
   if ( r2<=opts.tol || r2<=r02*opts.tol0 ) break; end; % term tests

   %beta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   beta = max(r2/or2,0); % Fletcher-Reeves
   d    = Mr+beta*d;  % conj grad direction
   ddJ  =-d'*dJ;         % new search dir grad.
   if( ddJ <= 0 ) d=Mr; ddJ=-d'*dJ; end; % non-descent dir switch to steepest
end;

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);


K=X'*X+1; % N.B. add 1 to give implicit bias term
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
[alpha,b]=l1svm(K(trnIdx,trnIdx),Y(trnIdx),1,'verb',2);
dv=K(tstIdx,trnIdx)*alpha;
dv2conf(dv,Y(tstIdx))

% for linear kernel
w=X(:,trnIdx)*alpha; b=sum(alpha); plotLinDecisFn(X,Y,w,b);

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
