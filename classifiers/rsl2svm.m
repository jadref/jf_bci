function [wb,J]=rsl2svm_cg(K,Y,C,beta,varargin);
% [alphab,J]=rsl2svm_cg(K,Y,C,beta,varargin)
% Reduced Set Quadratic Loss Support Vector machine using a pre-conditioned
% conjugate gradient solver so extends to large input kernels.
%
% J = C(1) w' K w + sum_i max(0 , 1 - y_i ( w'*K_i + b ) ).^2 
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
%  alphab  - initial guess at the kernel parameters and bias
%  maxIter - max number of CG steps to do
%  tol     - absolute error tolerance
%  tol0    - relative error tolerance, w.r.t. initial gradient
%  verb    - verbosity
%  step    - initial step size guess
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
if ( nargin < 3 ) C=0; end;
if ( nargin < 4 ) beta=[]; end;
opts=struct('alphab',[],...
            'maxIter',inf,'maxEval',inf,'objTol',0,'tol',1e-6,'tol0',eps,...
            'int',.1,'verb',0,'step',0,'wght',[],'X',[],'ridge',1e-9);
if(isstruct(varargin{1})) % struct->cell
   varargin=[fieldnames(varargin{1})'; struct2cell(varargin{1})'];
end
for i=1:2:numel(varargin); % process the options 
   if( ~isfield(opts,varargin{i}) ) error('Unrecognised option'); end;
   opts.(varargin{i})=varargin{i+1};
end
opts.ridge=opts.ridge(:);

[N,dim]=size(K); 

if ( islogical(beta) ) beta=find(beta);  
elseif ( isempty(beta) ) beta=int32(1:N); end; 
% get set and logical version beta
beta=int32(beta); ibeta=false(1,N); ibeta(beta)=true; 

invKbb = inv(K(beta,beta)+eye(numel(beta))*opts.ridge); 

wb=opts.alphab; if ( isempty(wb) ) wb=zeros(numel(beta)+1,1); end 

wK      = wb(1:end-1)'*K(beta,:); 
wK(beta)= wK(beta)+opts.ridge*wb(1:end-1)'; % include the ridge
err    = 1-Y'.*(wK+wb(end)); svs=err>=0;
% pre-condinationed gradient, 
Yerr = Y'.*err; % Y.*(1-Y.*f)
% K^-1*dJdw = K^-1(2 C(1)Kw - 2 K I_sv(Y-f)) = 2*(C(1)w - Isv (Y-f) )
% N.B. could include additional diag(K) pre-cond here if it would help
%MdJ  = [2*(C(1)*wb(1:end-1) - (Yerr.*svs)')./diag(K); 0];% ohne bias
MdJ  = [(2*C(1)*wb(1:end-1)-2*(Yerr(beta).*svs(beta))')./diag(K(beta,beta));...
        -2*sum(Yerr(svs))/N];
dJ   = [K(beta,beta)*(MdJ(1:end-1).*diag(K(beta,beta)))+opts.ridge*(MdJ(1:end-1).*diag(K(beta,beta))); ...
        N*MdJ(end)];
rsvs = svs & ~ibeta ;
if ( any(rsvs) ) % include the non-beta svs gradient effects
   KbsvYerr=K(beta,rsvs)*Yerr(rsvs)';
   MdJ(1:end-1) = MdJ(1:end-1) -2*invKbb*KbYerr;
   dJ(1:end-1)  = dJ(1:end-1)  -2*KbYerr;
end
Mr   =-MdJ;
d    = Mr;
ddJ  =-d'*dJ;
r2   = ddJ;
r02  = r2;

Ed     = err(svs)*err(svs)';  
Ew     = wK(:,beta)*wb(1:end-1);
J      = Ed + C(1)*Ew; % SVM objective

step=opts.step;
if( step<=0 ) step=abs(J/ddJ); end  % init step assuming opt is at 0
step=abs(step); tstep=step;

neval=1;
if(opts.verb>0)   % debug code      
   fprintf(['%3d) %3d x=[%8.6g,%8.6g,.] J=%5g |dJ|=%8.6g\n'],...
           0,neval,wb(1),wb(2),J,r2);
end

% alt pre-cond non-lin CG iteration
for i=1:opts.maxIter;

   oJ= J; or2=r2; oMr=Mr;% record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 1 )
      fprintf('%g=%g @ %g\n',0,ddJ,J);
      if ( opts.verb>2 ) hold on;plot(0,ddJ,'r*');text(0,ddJ,num2str(0));end
   end;
   step=tstep; % prev step size is first guess!
   oddJ=ddJ; % one step before is same as current
   wb = wb + step*d;
   % cache this so don't comp dJ
   Kd = [K(beta,beta)*d(1:end-1)+opts.ridge.*d(1:end-1);d(end)]; 
   for j=1:50;
      neval = neval+1;
      ooddJ=oddJ; oddJ=ddJ; % prev and 1 before grad values
      
      % Eval the gradient at this point.  N.B. only gradient needed for secant
      wK     = wb(1:end-1)'*K(beta,:) ;%+ opts.ridge*wb(1:end-1)';
      wK(beta)= wK(beta)+opts.ridge'.*wb(1:end-1)'; % include the ridge
      err    = 1-Y'.*(wK+wb(end)); svs=err>=0;
      Yerr   = Y'.*err; 
      MdJ    = [2*C(1)*wb(1:end-1) - 2*(Yerr(beta).*svs(beta))'; ...
                -2*sum(Yerr(svs))];
%       dJ     = [K(beta,beta)*MdJ(1:end-1)+opts.ridge*MdJ(1:end-1); ...
%                 MdJ(end)];
      rsvs = svs & ~ibeta ;
      if ( any(rsvs) ) % include the non-beta svs gradient effects
         KbsvYerr     = K(beta,rsvs)*Yerr(rsvs)'; % pre-comp for dJ later
         MdJ(1:end-1) = MdJ(1:end-1) -2*invKbb*KbsvYerr;
%          dJ(1:end-1)  = dJ(1:end-1)  -2*KbsvYerr;
      end
      ddJ =-Kd'*MdJ;            %ddJ =-d'*dJ;      

      if ( opts.verb > 1 )
         Ed     = err(svs)*err(svs)';   Ew     = wK(:,beta)*wb(1:end-1);
         J      = Ed + C(1)*Ew; % SVM objective
         fprintf('%g=%g @ %g \n',tstep,ddJ,J); 
         if ( opts.verb > 2 )
            hold on; plot(tstep,ddJ,'*'); text(tstep,ddJ,num2str(j)); drawnow;
         end;
      end;
      % convergence test, and numerical res test
      if ( abs(ddJ) <= opts.tol || abs(ddJ*step)<eps ) break; end; 

      % now compute the new step size
      % backeting check, so bracket always decreases
      if ( ooddJ*oddJ < 0 && oddJ*ddJ > 0 ...     % ooddJ still brackets
           && abs(step*oddJ/(oddJ-ddJ)) > abs(ostep) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         oddJ = ooddJ;
      end
      ostep = step;
      step  = step * ddJ / (oddJ-ddJ); % *RELATIVE* secant step size
      tstep = tstep + step;            % total step size
%       % various sanity check this proposed step.
%       % backeting check, so bracket always decreases
%       if ( ooddJ*oddJ < 0 && oddJ*ddJ > 0 ...     % ooddJ still brackets
%            && abs(step*oddJ/(oddJ-ddJ)) > abs(ostep) ) % would jump outside 
%          % quadratic test fit has obviously failed! - switch to golden-ratio?
%          step = ostep + step; % make as if we jumped here directly.
%          oddJ = ooddJ;
%          step  = ddJ / (oddJ-ddJ);
%       end
%       ostep = step;
%       step  = ddJ / (oddJ-ddJ); % *RELATIVE* secant step size
%       % ensure sufficient step size
% %       if ( step*ostep > 0 ) % extrap step
% %          step  = max(abs(step),opts.int)*sign(step);
% %       else % internal step
% %          step  = min(max(abs(step),opts.int),1-opts.int)*sign(step);
% %       end
%       % This step looks OK, so use it
%       step  = ostep * step;
%       tstep = tstep + step;            % total step size
      
      % move to the new point
      wb    = wb + step*d ;
   end
   if ( opts.verb > 1 ) fprintf('\n'); end;
   % compute the other bits needed for CG iteration
   dJ     = [K(beta,beta)*MdJ(1:end-1)+opts.ridge*MdJ(1:end-1); ...
             MdJ(end)];
   rsvs = svs & ~ibeta ;
   if ( any(rsvs) ) % include the non-beta svs gradient effects
      dJ(1:end-1)  = dJ(1:end-1)  -2*KbsvYerr;
   end
   Mr    =-MdJ;
   r2    =abs(Mr'*dJ);
      
   delta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   %delta = max(r2/or2,0); % Fletcher-Reeves
   d     = Mr+delta*d;    % conj grad direction
   ddJ   =-d'*dJ;         % new search dir grad.
   if( ddJ <= 0 ) % non-descent dir switch to steepest
      if ( opts.verb > 1 ) fprintf('non-descent dir\n'); end;
      d=Mr; ddJ=-d'*dJ; 
   end; 
   
   % compute the function evaluation
   Ed     = err(svs)*err(svs)';  
   Ew     = wK(:,beta)*wb(1:end-1);
   J      = Ed + C(1)*Ew; % SVM objective
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8.6g,%8.6g,.] J=%5g |dJ|=%8.6g\n'],...
              i,neval,wb(1),wb(2),J,r2);
   end   

   if ( r2<=opts.tol || r2<=r02*opts.tol0 || neval>opts.maxEval ||...
        abs(J-oJ)<opts.objTol) % term tests
      break;
   end; 

end;

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);


K=X'*X; % Simple linear kernel
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
trnSet=find(trnIdx);

beta=randperm(sum(trnIdx));beta=sort(beta(1:ceil(end/8)));beta=int32(sort(beta));
[alphab,J]=rsl2svm(K(trnIdx,trnIdx),Y(trnIdx),1,beta,'verb',2);

dv=K(tstIdx,trnSet(beta))*alphab(1:end-1)+alphab(end);
dv2conf(dv,Y(tstIdx))

% for linear kernel
alpha=zeros(N,1);alpha(trnSet(beta))=alphab(1:end-1); % equiv alpha
w=X(:,trnSet(beta))*alphab(1:end-1); b=alphab(end); plotLinDecisFn(X,Y,w,b,alpha);

% check the primal dual gap!
w=alphab(1:end-1); b=alphab(end);
Jd = C(1)*(2*sum(w.*Y(trnIdx)) - C(1)*w'*w - w'*K(trnIdx,trnIdx)*w);  
