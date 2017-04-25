function [w,C,J]=l2RMsvm(K,Y,C,varargin);
% [alpha,J]=svm(K,Y,C,varargin)
% Quadratic Loss Support Vector machine using a pre-conditioned conjugate
% gradient solver so extends to large input kernels.
% Including the radius-margin optimisation of the regularisation parameter!
%
% N.B. this is without BIAS, so you must include it in the kernel yourself
%
% J = (varK/C(1)+1)*(C(1) w' K w + sum_i max(0 , 1 - y_i ( w'*K_i ) ).^2) 
% 
% Options:
%  alpha   - initial guess at the kernel parameters
%  maxIter - max number of CG steps to do
%  tol     - absolute error tolerance
%  tol0    - relative error tolerance, w.r.t. initial gradient
%  verb    - verbosity
%  beta    - back-step size used in the line-search
%  sigma   - threshold for the armijo line search (not used)
%  step    - initial step size guess
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
if ( nargin < 3 ) C=0; end;
opts=struct('alpha',[],'maxIter',inf,'tol',1e-3,'tol0',1e-9,'verb',0,...
            'beta',.1,'sigma',1e-1,'step',0,'wght',[],'ridge',1e-4);
for i=1:2:numel(varargin); % process the options 
   if ( ~isstr(varargin{i}) || ~isfield(opts,varargin{i}) )
      error('Unrecognised option');
   else
      opts.(varargin{i})=varargin{i+1};
   end
end

[N,dim]=size(K);

w=opts.alpha;
if ( isempty(w) ) % regularised kernel least squares seed 
   if ( N > 800 ) 
      w=zeros(N,1);
      prm=randperm(N);prm=prm(1:1000);
      w(prm)=(K(prm,prm)+eye(1000,1000)*opts.ridge)\Y(prm);
   else      
      w=(K+eye(size(K))*opts.ridge)\Y; 
   end
end 
lam = log(C(1)); % true parameter is log input so C=exp(lam);


varK   = sum(diag(K))/N - sum(K(:))/N/N; % data variance

wK     = w'*K;
err    = 1-Y'.*wK; svs=err>0;
Ed     = sum(err(svs).^2);  
Ew     = wK*w/2;
Jw     = Ed + C(1)*Ew; % SVM objective
Jv     = (varK/C(1)+1);% data variance objective
J      = Jv*Jw;        % Variance margin objective

% pre-condinationed gradient, 
Yerr = Y'.*err; % Y.*(1-Y.*f)
% K^-1*dJdw = K^-1(2 C(1)Kw - 2 K I_sv(Y-f)) = 2*(C(1)w - Isv (Y-f) )
MdJ   = [Jv*(C(1)*w - 2*(Yerr.*svs)'); Ew*C(1)-Ed*varK/C(1)]; %PC'd grad
dJ    = [K*MdJ(1:end-1); MdJ(end)];                           %true grad
Mr    =-MdJ;
d     = Mr;
ddJ   =-d'*dJ;
r2    = ddJ;
r02   = r2;

step=opts.step;
if( step<=0 ) step=abs(J/ddJ); end  % init step assuming opt is at 0
step=abs(step); tstep=step;

neval=1;
if(opts.verb>0)   % debug code      
   fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],0,neval,w(1),w(2),J,r2);
end

% alt pre-cond non-lin CG iteration
for i=1:opts.maxIter;

   oJ= J; oMr  = Mr; or2=r2; % record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 1 )
    fprintf('%g=%g @ %g\n',0,ddJ,J);
    if ( opts.verb>2 ) hold on;plot(0,ddJ,'r*');text(0,ddJ,num2str(0));end
   end;
   step=tstep; % prev step size is first guess!
   w = w + step*d(1:end-1); lam = lam + step*d(end); C=exp(lam);
   for j=1:20;
      oddJ=ddJ;
      
      % Eval the gradient at this point.  N.B. only gradient needed for secant
      wK    = w'*K;
      err   = 1-Y'.*wK; svs=err>0;
      Yerr  = Y'.*err; 
      MdJ   = [Jv*(C(1)*w - 2*(Yerr.*svs)'); Ew*C(1)-Ed*varK/C(1)]; %PC'd grad
      dJ    = [K*MdJ(1:end-1); MdJ(end)];                           %true grad
      ddJ   =-d'*dJ;
      
      if ( opts.verb > 1 )
         fprintf('%g=%g @ %g \n',tstep,ddJ,J); 
         if ( opts.verb > 2 )
            hold on; plot(tstep,ddJ,'*'); text(tstep,ddJ,num2str(j));
         end;
      end;
      % now compute the new step size
      step = step * ddJ / (oddJ-ddJ); % *RELATIVE* secant step size
      tstep= tstep + step;            % total step size
      
      % move to the new point
      w   = w + step*d(1:end-1); 
      lam =lam+ step*d(end);  C=exp(lam); % update in log, use real      
      neval=neval+1;
      
      if ( abs(ddJ) <= opts.tol  ) break; end; % convergence test;      
   end
   if ( opts.verb > 1 ) fprintf('\n'); end;
   % compute the function evaluation
   Ed     = sum(err(svs).^2);  
   Ew     = wK*w/2;
   Jw     = Ed + C(1)*Ew; % SVM objective
   Jv     = (varK/C(1)+1);% data variance objective
   J      = Jv*Jw;        % Variance margin objective

   % compute the other bits needed for CG iteration
   Mr    =-MdJ;
   r2    =-Mr'*dJ;
      
   if(opts.verb>0)   % debug code      
     fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],i,neval,w(1),w(2),J,r2);
   end   
   if ( r2<=opts.tol || r2<=r02*opts.tol0 ) break; end; % term tests

   %beta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   beta = max(r2/or2,0); % Fletcher-Reeves
   d    = Mr+beta*d;  % conj grad direction
   if( d'*dJ >= 0 ) d=Mr; end; % if non-descent dir switch to steepest
end;

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);


K=X'*X+1; % N.B. add 1 to give implicit bias term
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
[alpha,b]=l2RMsvm(K(trnIdx,trnIdx),Y(trnIdx),1,'verb',2);
dv=K(tstIdx,trnIdx)*alpha;
dv2conf(dv,Y(tstIdx))

% for linear kernel
w=X(:,trnIdx)*alpha; b=sum(alpha); plotLinDecisFn(X,Y,w,b);

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
