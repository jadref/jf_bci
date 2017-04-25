function [f,df,ddf,wb]=evidenceCSPLR(hps,X,Y,grads,iwb,featFn,verb)
% [f,df,ddf,wb]=evidenceCSPLR(hps,X,Y,grads)
% N.B. doesn't compute second derivatives!
% X -- [dim x dim x examples] matrix of per example, per band covariance matrices
% Y -- [examples x 1] matrix of 0/1 labels
% sf -- [dim x featDim] matrix of spatial filters
% evidence based Logistic regression classification objective function

persistent owb; persistent onlp; % cache old solution to re-use if possible
persistent tcalled; if (isempty(tcalled)) tcalled=0; end;

if ( nargin < 4 ) grads=3; end; % alpha,sf
% reset starting loc if wanted
if ( nargin >=5 ) owb=iwb; onlp=inf; else iwb=[]; end; 
if ( nargin < 6 | isempty(verb) ) verb=1; end;
if ( nargin < 7 ) featFn='cspFeatFn'; end; % feature fn to use

% deal with banded input
if ( ndims(X)<=3 )  [indim,indim,N]=size(X); nband=1;
elseif(ndims(X)==4) [indim,indim,nband,N]=size(X); 
end;

% extract the parameters
featDim=(numel(hps)-1)/indim; %N.B. this MUST be exact!
sf=reshape(hps(1:end-1),indim,featDim);
alpha=exp(hps(end));          %N.B. exp() so has arbitary sign & unconstrained!

% make the features and get their hyper-parameter gradients
if bitand(grads,2)
   [phiX dphiXdsf ddphiXdsf]=feval(featFn,sf,X);
else
   [phiX]=feval(featFn,sf,X);
end

if ( ~isempty(owb) ) 
   w=owb(1:end-1);b=owb(end);
   if( verb>0 & cputime-tcalled > .1)
      hold off; labScatPlot(phiX',Y); hold on;
      drawLine(w,b,min(phiX'),max(phiX'));
      drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
      drawLine(w,b-1,min(phiX'),max(phiX'),'r-');
      drawnow; hold off;
      tcalled=cputime;
   end;
end

% Use Newtons method to get a ML decision boundary estimate
wght=ones(N,1);
% set the bias to get us in the ball-park
sX=sum(phiX')';sXX=sum(sum(phiX'.*phiX'));varX=max((sXX - sX'*sX/N)/N,eps);
sY=sum(Y);
%w=sX/N; w=w/norm(w)/sqrt(varX); b=-sX'*w/N; wb=[w;b];
% LS solution to get us in the ball-park
wls=(phiX*phiX' - sX*sX'/N)\(phiX*Y -sY*sX/N);
bls=-sX'*wls/N;
%wbls=[phiX*phiX'-sX*sX'/N sX; sX' N]\[(phiX*Y -sY*sX/N);sY];

% use old solution as seed if a) still valid, b) close enough to old one
w=wls;b=bls;
% evalute the LS solution
fi  = (1+exp(-phiX'*w-b)).^-1; fi=min(max(fi,eps),1-eps); % stop log 0
Ed  = sum( Y.*log(fi) + (1-Y).*log(1-fi) );
Ew  = w'*w/2;
% neg log posterior
neglogpost = -(Ed - alpha*Ew);
if ( ~isempty(iwb) || ( numel(owb)==indim+1 && onlp < neglogpost ) ) 
   wb=owb;
   neglogpost=onlp;  % use the 
else
   wb=[wls;bls]; % if in doubt use the LS solution
   neglogpost=inf;
end

% PART I --- Optimise the decision function
owb=wb;
for iter=1:1

   % adaptive step sizer
   d = wb-owb;  % search direction
   onlp= neglogpost;
   if ( verb>0 ) fprintf('%d)',iter); end
   for MLstep=2.^[0:-1:-9];
      w=wb(1:end-1); b=wb(end);

      % evalute the current solution
      fi  = (1+exp(-phiX'*w-b)).^-1; fi=min(max(fi,eps),1-eps); % stop log 0
      Ed  = sum( Y.*log(fi) + (1-Y).*log(1-fi) );
      Ew  = w'*w/2;

      % neg log posterior
      neglogpost = -(Ed - alpha*Ew);
      if ( verb>0 ) fprintf('%g=%g  ',MLstep,neglogpost); end;
      % reduce step size until neglogposterior reduces
      if ( neglogpost <= onlp )  break; 
      else                             wb=owb+MLstep*d;
      end
   end;
   
   % compute the weight gradients
   err = Y-fi;
   wght= fi.*(1-fi);
   dEDdw  = phiX*err;
   dLdw   = dEDdw - alpha*w;
   dLdb   = sum(err);
   ddLdwdb= -phiX*wght;
   ddLddb = -sum(wght);
   G  = [dLdw; dLdb];
   H  = [-phiX*diag(wght)*phiX'-eye(featDim,featDim)*alpha ddLdwdb;...
         ddLdwdb',ddLddb];
   
   if ( neglogpost > onlp & MLstep == 2^-9 ) 
      fprintf('Couldnt decrease log likelihood: bailing\n'); 
      break;
   end;
   %fprintf('\n');

   % plot the current decision function
   % plot every X seconds?
   if( verb>0 & cputime-tcalled > 0)
      hold off; labScatPlot(phiX',Y); hold on;
      drawLine(w,b,min(phiX'),max(phiX'));
      drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
      drawLine(w,b-1,min(phiX'),max(phiX'),'r-');
      drawnow; hold off;
      tcalled=cputime;
   end;
   
   % Update the parameters
   owb=wb;
%    if ( det(H) > 0 | rcond(H) < 1e-12 | isnan(rcond(H)) ) % only do newton step if -H is pos-def -- i.e. concave fn
%       wb=wb+G./norm(G)*norm(wb)*.001;
%    else
%       wb=wb-(H\G);
%       %wb=(-H)\(X'*(wght.*(X*w+b)+err));         
%   end
   
   % Convergence test
   if( neglogpost<onlp & norm(owb-wb)/norm(wb)<1e-2 & norm(G)<1e-3 ) 
      break; 
   end;
end;
owb=wb; onlp=neglogpost;
if ( verb>0 ) fprintf('\n'); end;
if( verb > 2 | (verb>0 & cputime-tcalled > 2) )
   hold off; labScatPlot(phiX',Y); hold on;
   drawLine(w,b,min(phiX'),max(phiX'));
   drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
   drawLine(w,b-1,min(phiX'),max(phiX'),'r-');
   drawnow; hold off;
   tcalled=cputime;
end;

% Part II: Now were at the ML setting compute the evidence approx and its
% gradient.

% get its chol factor
Sigma=-H;    % Covariance of the posterior is its negative hessian
R=chol(Sigma);  % N.B. -H to make sure its pos-def 
% inverse of the chol factor 
% N.B. more efficient to do this using the BLAS routine
invR=R\eye(size(R)); 
   
logdetSigma = 2*sum(log(diag(R)));
trinvSigma  = sum(invR(:).*invR(:));
%invSigma    = invR'*invR;
   
% compute the neg log evidence, i.e. our objective value
J2= Ed - alpha*Ew + featDim/2*log(max(alpha,eps)) - logdetSigma/2 ;
f = -J2;
if ( verb>0 ) fprintf('O %d) %g = %g\n',i,alpha,f); end;
   
% Compute the derivates
if ( nargout > 1 )
   ddf=zeros(size(hps));
   dJ2dsf=zeros(size(sf)); dJ2dalpha=0;
   
   % compute the regulasier gradient
   if ( bitand(grads,1) )
      % N.B. dSigmadalpha = [eye(dim,dim) 0;0 0] as dJ2db = 0.
      dJ2dalpha = -alpha*(-Ew + featDim/2/max(alpha,eps) - (trinvSigma-invR(:,end)'*invR(:,end))/2);
   end
   
   % compute the sf gradient
   if ( bitand(grads,2) )
      % compute the bit of the inverse hessian we need
      invSigma1    = invR'*invR(:,1);
      % update the sp directions
      % Compute the gradient of the Loss w.r.t. the sf's 
      % N.B. we ignore dependence of wght and w_mp on the hyperparameters.
      for fd=1:featDim;

         % do the bits which are vectors
         % dE_d/dsf = sum_i (y_i-f(x_i))'*dg_i/ds_d
         dEDdsf=w(fd)*squeeze(dphiXdsf(:,fd,:))*err;
   
         % compute the gradient of the hessian w.r.t. the sf's (ignore wght)
         phiXDdphiXdsf=phiX*diag(wght)*squeeze(dphiXdsf(:,fd,:))';

         % diff of the b bits (ignoreing wght)
         dphiXDdsf = (squeeze(dphiXdsf(:,fd,:))*wght)';
         
         % compute the combined gradient...          
         dJ2dsf(:,fd)=-(dEDdsf + [phiXDdphiXdsf;dphiXDdsf]'*invSigma1);
         
      end % loop fd
   end
   
   df=[dJ2dsf(:); dJ2dalpha];
end

return;

%-----------------------------------------------------------------------------
function []=testCase()
% Toy data
N=500; indim=2;
Y=sign(randn(N,1));X=randn(1000,indim,N); 
fX=rcovFilt(X); trnIdx(2:2:N,1)=true;
d=randn(indim,1);
for i=1:N; % inflate dir d.. 
   if ( Y(i)>0 ) 
      fX(:,:,i)=fX(:,:,i)*(eye(indim,indim)+abs(randn)/20*d*d'/(d'*d)); 
   end; 
end

% BCI data
z=loadprep(bci('hm'),'c_dsd'); 
z=filterfft(z,'mode','bandpass','bands',[3 6 35 40],'downsample',true);
z=reserve(z,.5);
[N,indim,nEx]=size(z.x); 
Y=z.y; trnIdx=false(size(z.y));trnIdx(fold(z.outerfoldguide))=true; 
fX=rcovFilt(permute(z.x,[3 2 1]));

% init 
featDim=2;

% SF:  init
% CSP
if ( ndims(fX)>3 ) 
   XX1=sum(squeeze(fX(:,:,end,trnIdx & Y>0)),3);
   XX=sum(squeeze(fX(:,:,end,trnIdx)),3); 
else XX1=sum(fX(:,:,trnIdx & Y>0),3);XX=sum(fX(:,:,trnIdx),3);
end;
[U,D]=eig(XX1,XX); [sD,sI]=sort(abs(.5-diag(D)));D=diag(D);D=D(sI);U=U(:,sI);
sf0=U(:,end:-1:end-featDim+1); 
% Rand
sf0=randn(size(X,2),featDim);  
% Normalise and fix up the directions
sf0=orth(sf0);                                         % orthonogal directions
sf0=sf0*diag(sign(sum(sf0,1))./sqrt(sum(sf0.*sf0,1))); % Normalise

% ALPHA: init
% evidence max
if(ndims(fX)>3) 
   phiX=cspFeatFn(sf0,fX(:,:,bandIdx,:)); else phiX=cspFeatFn(sf0,fX); 
end;
[w,b,alpha0]=evidenceLR(phiX(:,trnIdx)',Y(trnIdx)>0,5,0);

% SFA: init
sfa0=[sf0(:);log(alpha0)]; sfa=sfa0;

[f,df,ddf,wb]=evidenceCSPLRFn(sfa,fX(:,:,trnIdx),Y(trnIdx)>0);

checkgrad(@(x) evidenceCSPLRFn(x,fX,Y>0,3,wb),sfa,1e-5,0,1);
checkgrad(@(x) evidenceCSPLRFn(x,fX,Y>0,3,wb),[orth(randn(numel(sf0),1));1],1e-5,0,1);

checkgrad(@(x) evidenceCSPLRFn(x,fX(:,:,bandIdx,:),Y>0),[randn(size(sf0(:)));log(.1)],1e-3);

% Optimisation
% N.B. need the pre-conditioner as the hessian w.r.t. alpha is *MUCH* smaller 
% than for sf
sfa=nonLinConjGrad(@(w) evidenceCSPLRFn(w,fX(:,:,trnIdx),Y(trnIdx)>0),sfa0,'plot',0,'verb',1,'maxEval',5000,'maxIter',inf,'curveTol',1e-3,'pCond','adapt','alpha0',1e-4);
sfa=nonLinConjGrad(@(w) evidenceCSPLRFn(w,fX(:,:,bandIdx,trnIdx),Y(trnIdx)>0),sfa0,'plot',0,'verb',1,'maxEval',5000,'maxIter',inf,'curveTol',1e-3,'pCond','adapt','alpha0',1e-4);

sfa=nonLinConjGrad(@(w) evidenceCSPLRFn(w,fX(:,:,trnIdx),Y(trnIdx)>0),sfa0,'plot',0,'verb',1,'maxEval',5000,'maxIter',inf,'curveTol',1e-3,'pCond',[ones(numel(sfa)-1,1);1e-4],'alpha0',1e-4);
sfa=nonLinConjGrad(@(w) evidenceCSPLRFn(w,fX(:,:,bandIdx,trnIdx),Y(trnIdx)>0),sfa0,'plot',0,'verb',1,'maxEval',5000,'maxIter',inf,'curveTol',1e-3,'pCond',[ones(numel(sfa)-1,1);1e-4],'alpha0',1e-4);

% results extraction
[f,df,ddf,wb]=evidenceCSPLRFn(sfa,fX(:,:,trnIdx),Y(trnIdx)>0,3,3);
w=wb(1:end-1);b=wb(end);sf=reshape(sfa(1:end-1),indim,featDim);alpha=exp(sfa(end));

% Performance assessment
if(ndims(fX)>3) 
   phiX=cspFeatFn(sf,fX(:,:,bandIdx,:)); else phiX=cspFeatFn(sf,fX); 
end;
clf;labScatPlot(phiX',Y); hold on;
drawLine(w,b,min(phiX'),max(phiX'));
drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
drawLine(w,b-1,min(phiX'),max(phiX'),'r-');
dv=phiX'*w+b;
dv2conf(dv(trnIdx),Y(trnIdx));
dv2conf(dv(~trnIdx),Y(~trnIdx));

