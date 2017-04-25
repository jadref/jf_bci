function [w,b,alpha,neglogevidence,neglogpost]=evidenceLR(X,Y,maxIter,alpha,verb)
% evidence based Logistic regression classification optimisation
if ( nargin < 4 ) alpha=0; end;
if ( nargin < 5 ) verb=0; end;
labels=unique(Y); oY=Y; Y(Y==labels(1))=0; Y(oY~=labels(1))=1;

persistent tcalled; if (isempty(tcalled)) tcalled=0; end;

% convert Y to P(Y==1)
Y=(Y>0); 

[N,dim]=size(X);
wb=ones(dim+1,1); 
% N.B. need the decision function to be near the center otherwise 
% everything buggers up!
% Middle of the data
% sX=sum(X)'; sXX=sum(sum(X.*X)); varX=max((sXX - sX'*sX/N)/N,eps);
% w0=sX/N; w0=w0/norm(w0)/sqrt(varX); b0=-sX'*w0/N;
% LS
sX=sum(X)';sY=sum(Y);w0=(X'*X - sX*sX'/N)\(X'*Y -sY*sX/N); b0=-sX'*w0/N;
wb=[w0;b0]; owb=wb;
wght=ones(N,1); neglogevidence=inf;MEstep=1; oalpha=alpha;
for i=1:maxIter;
   
   % comp MP estimate
%    [w,b,wght,sWght,muX,covXX]=rrwls(X,Y,wght,1,alpha,...
%                                     'reweightFn','logistic','maxIter',10);

   % alt Newton iteration:
   neglogpost=inf; owb=wb;
   for j=1:20

      w=wb(1:end-1); b=wb(end); 
      
      % adaptive step sizer
      d = wb-owb;  % search direction
      oneglogpost= neglogpost;
      if ( verb > 0 ) fprintf('%d)',j); end;
      for MLstep=2.^[0:-1:-10];
         % evalute the current solution
         f   = (1+exp(-X*w-b)).^-1; f=min(max(f,eps),1-eps); % stop log 0
         Ed  = sum( Y.*log(f) + (1-Y).*log(1-f) );
         Ew  = w'*w/2;

         % neg log posterior
         neglogpost = -(Ed - alpha*Ew);
         if ( verb > 0 ) fprintf('%g=%g ',MLstep,neglogpost); end;
         % reduce step size until neglogposterior reduces
         if ( neglogpost < oneglogpost )  break; 
         else                             wb=owb+MLstep*d;
         end
      end;
      if ( verb > 0 ) fprintf('\n'); end;
      if ( neglogpost > oneglogpost ) % couldn't reduce!
         fprintf('posterior didnt reduce: bailing\n');
         break;
      end

      err = Y-f;
      wght= f.*(1-f);
      
      % compute the weight gradients
      dEDdw  = X'*err;
      dLdw   = dEDdw - alpha*w;
      dLdb   = sum(err);
      ddLdwdb= -X'*wght;
      ddLddb = -sum(wght);
      G  = [dLdw; dLdb];
      H  = [-X'*diag(wght)*X-eye(dim,dim)*alpha ddLdwdb;...
            ddLdwdb',ddLddb];
      
      % plot the current decision function
      % plot every X seconds?
      if( verb > 1 & cputime-tcalled > 2)
         hold off; labScatPlot(X,Y); hold on;
         drawLine(w,b,min(X),max(X));
         drawLine(w,b+1,min(X),max(X),'r-');
         drawLine(w,b-1,min(X),max(X),'r-');   
         drawnow
         tcalled=cputime;
      end;
      
      % Update the parameters
      owb=wb;
      if ( det(H) > 0 ) % only do newton step if H is pos-def
         wb=wb+MLstep*G*.001;
      else
         if ( size(H,1) < 300 || i==1 ) % use exact inv first time round
            sd=H\G;
         else
            sd=cgSolve(H,G,sd,20,1e-2);%for speed use CG for approx search dir
         end         
         wb=wb-MLstep*sd;
         %wb=(-H)\(X'*(wght.*(X*w+b)+err));         
      end
      
      if( neglogpost<oneglogpost & norm(owb-wb)/norm(wb)<1e-8 & norm(G)<1e-7 )
         break; 
      end;

   end;
   
   % compute the new weight?
   %ow=w;  if ( alpha>0 )  w=-G_d/alpha/2; end;

   % the hyper-fn to minimise
   oneglogevidence=neglogevidence;
   neglogevidence=-(Ed-alpha*Ew + dim/2*log(max(alpha,eps))-log(det(H))/2);
   if ( verb > 0 ) fprintf('\nO %d) %g -> %g\n\n',i,alpha,neglogevidence); end;
   if ( oneglogevidence < neglogevidence ) 
      alpha=oalpha; MEstep=MEstep/2; continue; 
   end;
   
   % eG
   %eG = -w'*w + dim/(2*max(alpha,eps)) + trace(inv(H));
   
   % and the new regulariser?
   oalpha=alpha;
   % N.B. the b row and column of H are indepenent of alpha!
   invH=inv(H);
   trinvH=trace(invH(1:end-1,1:end-1)));
   alpha=MEstep*(dim./(2*w'*w-trinvH) +(1-MEstep)*alpha;

   if( oneglogevidence - neglogevidence < 1e-4 & norm(oalpha-alpha) < 1e-5 ) 
      break; 
   end;
   
   
end

return;

function [x,fx]=cgSolve(A,b,x,maxIter,tol)
d=-(A*x-b); r=d; r2 = r'*r; r02=r2; fx=x'*r;
for i=1:maxIter;  
  t=A*d; 
  nf=d'*t;
  alpha= r2 / nf;
  x=x+alpha*d;
  or=r;ofx=fx;
  %r=-(A*x-b);
  r=r-alpha*t;
  fx=x'*r;
  or2=r2;
  r2=r'*r; % udpate residual norm
  if ( r2 <= tol*r02 || or2<r2 ) break; end; % stop when close enough
  % use Polak-Ribiere update from non-lin cg to improve convergence
  beta=max(-r'*t/nf,0);%  beta=max(r'*(r-or)/or2,0);%
  d=r+beta*d;
end
if( or2<r2) x=x-alpha*d; end; % back up if messed up

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
[w,b]=evidenceLR(X,Y,10);

[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5; -.2 -.5],[400 400 20 20],[.3 .3; .3 .3; .2 .2; .2 .2],[],[-1 1 1 -1]);
