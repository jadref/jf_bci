function [wb,J]=klr_nm(K,Y,C,varargin);
% [alphab,J]=krl(K,Y,lambda,varargin)
% Regularised Kernel Logistic Regression using a newton iteration
%
% Inputs:
%  K       - NxN kernel matrix
%  Y       - Nx1 matrix of +1,-1 labels
%  C       - the regularisation parameter
%            good default is = .1*(mean(diag(K))-mean(K(:))))
%
% Outputs:
%  alphab  - (N+1)x1 matrix of the kernel weights and the bias b
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
opts=struct('alphab',[],'nobias',0,...
            'maxIter',inf,'maxEval',inf,'tol',1e-6,'tol0',0,'objTol',0,...
            'verb',0,'step',0,'wght',[],'X',[],'ridge',1e-9);
if(isstruct(varargin{1})) % struct->cell
   varargin=[fieldnames(varargin{1})'; struct2cell(varargin{1})'];
end
for i=1:2:numel(varargin); % process the options 
   if( ~isfield(opts,varargin{i}) ) error('Unrecognised option'); end;
   opts.(varargin{i})=varargin{i+1};
end

[dim,N]=size(K);

wb=opts.alphab; 
if ( isempty(wb) ) 
  wb=zeros(N+1,1); 
%   varK=sum(diag(K))/N-sum(K(:))/N/N;
%   wb(Y>0) = C(1)/sum(Y>0); wb(Y<=0)=-C(1)/sum(Y<=0); % LDA seed?
%   wb(1:end-1)=wb(1:end-1)./(sqrt(wb(1:end-1)'*K*wb(1:end-1)))*sqrt(varK)/10;
end 
if ( opts.nobias ) wb(end)=0; end;

% if ( opts.ridge > 0 ) 
%    K=K+diag(N)*opts.ridge;
% end;

% compute the initial search direction
wK   = wb(1:end-1)'*K;
g    = 1./(1+exp(-Y'.*(wK+wb(end)))); g=max(g,eps); % stop log 0
Yerr = Y'.*(1-g);
% Gradient with [K 0; 0 1] extracted as common factor
MdJ   = [(2*C(1)*wb(1:end-1) - Yerr');...
         -sum(Yerr)];
dJ   = [K*MdJ(1:end-1); MdJ(end)];
if ( opts.nobias ) MdJ(end)=0; dJ(end)=0; end
wght = g.*(1-g);
% Hessian with [K 0; 0 1] extracted as common factor
% N.B. we remove the wght*K on the bottom line for convergence issues!
MddJ  = [2*C(1)*eye(N)+spdiags(wght,0,N)*K     wght';...
         wght                          sum(wght)];
%ddJ   = [K*MddJ];
% compute the new newton search direction
d    = -(MddJ\MdJ);
dtdJ = -d'*dJ;      % gradient along the line   
r2   = dJ'*dJ;
r02  = r2;

% compute the function evaluation
Ed   = -sum(log(g));               % P(D|w,b,fp)
Ew   = wK*wb(1:end-1);             % P(w,b|R);
J    = Ed + C(1)*Ew;               % J=neg log posterior

neval=1;
if(opts.verb>0)   % debug code      
   fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],...
           0,neval,wb(1),wb(2),J,r2);
end
if( dtdJ <= 0 ) error('something funny'); end;% check if its descent dir

for i=1:opts.maxIter; % newton iteration

   tstep=0;
   if ( opts.verb > 1 )      
      fprintf('%g=%g\n',tstep,dtdJ); 
      if ( opts.verb > 2 ) 
         hold on; plot(tstep,dtdJ,'*'); text(tstep,dtdJ,num2str(0));
      end
   end;
   step =1;    % intial step is pure newton
   tstep=step;
   odtdJ=dtdJ; % one step before is same as current
   wb  = wb + step*d;
   for j=1:50; % secant based line-search for the new min location
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      % Eval the gradient at this point.  N.B. only gradient needed for secant
      wK   = wb(1:end-1)'*K;
      g    = 1./(1+exp(-Y'.*(wK+wb(end)))); g=max(g,eps); % stop log 0
      Yerr = Y'.*(1-g);
      MdJ  = [2*C(1)*wb(1:end-1) - Yerr';...
              -sum(Yerr)];
      dJ   = [K*MdJ(1:end-1);MdJ(end)];
      if ( opts.nobias ) MdJ(end)=0; dJ(end)=0; end
      dtdJ  =-d'*dJ;  % gradient along the line      
      
      if ( opts.verb > 1 )
         fprintf('%g=%g\n',tstep,dtdJ); 
         if ( opts.verb > 2 ) 
            hold on; plot(tstep,dtdJ,'*'); text(tstep,dtdJ,num2str(j));
         end
      end;

      if ( abs(dtdJ) <= opts.tol || abs(dtdJ*step)<eps ) break; end; % convergence test;      

      % now compute the new step size
      % backeting check, so it always decreases
      if ( oodtdJ*odtdJ < 0 && odtdJ*dtdJ > 0 ...       % oodtdJ still brackets
           && abs(step*odtdJ/(odtdJ-dtdJ)) > abs(ostep) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         odtdJ = oodtdJ;
      end
      ostep=step;
      step = step * dtdJ / (odtdJ-dtdJ); % *RELATIVE* secant step size
      tstep= tstep + step;
      
      % move to the new point
      wb    = wb + step*d ;
   end
   
   % compute the new search direction from this point
   wght = g.*(1-g);
   MddJ  = [2*C(1)*eye(N)+spdiags(wght,0,N)*K    wght';...
            wght                               sum(wght)];
   %   MddJ  = [(2*C(1)*eye(N) + spdiags(wght,0,N)*K)];
   % compute the new newton search direction
   d    = -(MddJ\MdJ);
   dtdJ = -d'*dJ;      % gradient along the line   
   if( dtdJ <= 0 ) % check if its descent dir
      warning('Somthing funny -- search dir pointed the wrong way!');
      %d=-d;
      d=-dJ; % fall back on GD
      dtdJ=-d'*dJ;
   end;
   
   % compute the function evaluation
   oJ   = J;
   Ed   = -sum(log(g));               % P(D|w,b,fp)
   Ew   = wK*wb(1:end-1);             % P(w,b|R);
   J    = Ed + C(1)*Ew;               % J=neg log posterior
   
   r2   = dJ'*dJ;
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],...
              i,neval,wb(1),wb(2),J,r2);
   end   

   if ( r2<=opts.tol || r2<=r02*opts.tol0 ) break; end; % term tests
   if ( abs(J-oJ) < opts.objTol || neval > opts.maxEval ) break; end;
end;
fprintf(['%3d) %3d x=[%8g,%8g,.] J=%5g |dJ|=%8g\n'],...
        i,neval,wb(1),wb(2),J,r2);

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

K=X'*X; % N.B. add 1 to give implicit bias term
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
trnSet=find(trnIdx);

[alphab,J]=klr2(K(trnIdx,trnIdx),Y(trnIdx),1,'verb',1);
dv=K(tstIdx,trnIdx)*alphab(1:end-1)+alphab(end);
dv2conf(dv,Y(tstIdx))

% for linear kernel
alpha=zeros(N,1);alpha(trnSet)=alphab(1:end-1); % equiv alpha
plotLinDecisFn(X,Y,X(:,trnIdx)*alphab(1:end-1),alphab(end),alpha);

% unbalanced data
wght=[1,sum(Y>0)/sum(Y<=0)];
