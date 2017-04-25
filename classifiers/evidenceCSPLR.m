function [w,b,sf,lambda,js]=evidenceCSPLR(X,Y,sf,maxIter)
% X -- [dim x dim x examples] matrix of per example, per band covariance matrices
% Y -- [examples x 1] matrix of 0/1 labels
% sf -- [dim x featDim] matrix of spatial filters
% evidence based Logistic regression classification optimisation
if ( nargin < 4 ) maxIter=10; end;
nu=.00001;
persistent tcalled; 

labels=unique(Y); oY=Y; Y(Y==labels(1))=0; Y(oY~=labels(1))=1;

[indim,indim,N]=size(X);
[indim,featDim]=size(sf);
w=ones(featDim,1);b=0; lambda=0; wght=ones(N,1);
logevidence=-inf;
for i=1:maxIter;
   [phiX,dphiXdsf,ddphiXdsf]=cspFeatFn(sf,X); % compute the feature set
   
   hold off; labScatPlot(phiX',Y); hold on;
   drawLine(w,b,min(phiX'),max(phiX'));
   drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
   drawLine(w,b-1,min(phiX'),max(phiX'),'r-');   
   drawnow
   tcalled=cputime;
   
     % comp MP estimate
%    [w,b,wght,sWght,muX,covXX]=rrwls(phiX',Y,wght,1,lambda,...
%                                     'reweightFn','logistic','maxIter',4);
   % alt Newton iteration:
   % set the bias to get us in the ball-park
   if ( i == 1 ) 
     sX=sum(phiX')';sXX=sum(sum(phiX'.*phiX'));varX=max((sXX - sX'*sX/N)/N,eps);
     w=sX/N; w=w/norm(w)/sqrt(varX); b=-sX'*w/N; wb=[w;b];
     %w=mean(phiX')';b=-norm(w);w=w/norm(w);wb=[w;b];
     %w=ones(indim,1);
   end

   % Optimise the decision function
   MLstep=1; neglogpost=inf; oneglogpost=neglogpost; owb=wb;
   H=1;G=0; % cause the first update to do nowthing!
   for j=1:20

      % adaptive step sizer
      d = wb-owb;  % search direction
      oneglogpost= neglogpost;
      fprintf('%d)',j);
      for MLstep=2.^[0:-1:-5];
         w=wb(1:end-1); b=wb(end);

         % evalute the current solution
         f   = (1+exp(-phiX'*w-b)).^-1; f=min(max(f,eps),1-eps); % stop log 0
         Ed  = sum( Y.*log(f) + (1-Y).*log(1-f) );
         Ew  = w'*w/2;

         % neg log posterior
         neglogpost = -(Ed - lambda*Ew);
         fprintf('%g=%g  ',MLstep,neglogpost);
         % reduce step size until neglogposterior reduces
         if ( neglogpost <= oneglogpost )  break; 
         else                             wb=owb+MLstep*d;
         end
      end;
      fprintf('\n');
      
      % compute the weight gradients
      err = Y-f;
      wght= f.*(1-f);
      dEDdw  = phiX*err;
      dLdw   = dEDdw - lambda*w;
      ddLdwdb= -phiX*wght;
      dLdb   = sum(err);
      ddLddb = -sum(wght);
      G  = [dLdw; dLdb];
      H  = [-phiX*diag(wght)*phiX'-eye(featDim,featDim)*lambda ddLdwdb;...
            ddLdwdb',ddLddb];
   
      % plot the current decision function
      % plot every X seconds?
      if (isempty(tcalled)) tcalled=0; end;
      if( cputime-tcalled >= 0)
         hold off; labScatPlot(phiX',Y); hold on;
         drawLine(w,b,min(phiX'),max(phiX'));
         drawLine(w,b+1,min(phiX'),max(phiX'),'r-');
         drawLine(w,b-1,min(phiX'),max(phiX'),'r-');   
         drawnow
         tcalled=cputime;
      end;
      
      % Update the parameters
      owb=wb;
      if ( det(H) > 0 ) % only do newton step if H is pos-def
         wb=wb+G./norm(G)*norm(wb)*.001;
      else
         wb=wb-(H\G);
         %wb=(-H)\(X'*(wght.*(X*w+b)+err));         
      end
      
      if( neglogpost<oneglogpost & norm(owb-wb)/norm(wb)<1e-2 & norm(G)<1e-4 ) 
         break; 
      end;

   end;

   % get its chol factor
   Sigma=-H;    % Covariance of the posterior is its negative hessian   
   R=chol(Sigma);  % N.B. -H to make sure its pos-def 
   % inverse of the chol factor 
   % N.B. more efficient to do this using the BLAS routine
   invR=R\eye(size(R)); 
   
   logdetSigma = 2*sum(log(diag(R)));
   trinvSigma  = -sum(invR(:).*invR(:));
   invSigma    = invR'*invR;
   
   % compute the evidence
   ologevidence=logevidence;
   logevidence = Ed - lambda*Ew + featDim/2*log(max(lambda,eps)) - logdetSigma/2;
   js(i)=logevidence;
   fprintf('O %d) %g = %g\n',i,lambda,logevidence);
   
   % update regulariser
   dEdlambda = -Ew + featDim/(2*max(lambda,eps)) - trinvSigma/2;
   lambda    = featDim./(2*Ew);% - trinvSigma)
   
   % update the sp directions
   % Compute the gradient of the Loss w.r.t. the sf's
   for fd=1:featDim;

      % do the bits which are vectors
      % dE_d/dsf = sum_i (y_i-f(x_i))'*dg_i/ds_d
      dEDdsf(:,fd)=w(fd)*squeeze(dphiXdsf(:,fd,:))*err;
      
      % compute the gradient of the hessian w.r.t. the sf's
      % first and last components
      phiXDdphiXdsf(:,:,fd)=phiX*diag(wght)*squeeze(dphiXdsf(:,fd,:))';

      % the rest are a matrix per coefficient so do them one by one.
      for id=1:indim; % loop over the input dimensions

         % the gradient of the point weight with the sf
         dwghtdsf = zeros(size(wght)); % BODGE: no weight gradients!
                                       % dwghtdsf = wght.*(1-2*fi).*squeeze(dphiXdsf(id,fd,:));

         % the middle component of the hessian derivative
         phiXdDdsfphiX(:,:,id,fd)=phiX*diag(dwghtdsf)*phiX';

         % diff of the b bits..
         dphiXDdsf= phiX*dwghtdsf;
         dphiXDdsf(fd) = squeeze(dphiXdsf(id,fd,:))'*wght;
         
         % put both together to get the hessian gradient
         dSigmadsf(:,:,id,fd) =[-phiXdDdsfphiX(:,:,id,fd) -dphiXDdsf;...
                             -dphiXDdsf', sum(dwghtdsf)];
         dSigmadsf(:,1,id,fd) = dSigmadsf(:,1,id,fd) -[phiXDdphiXdsf(:,id,fd);0];
         dSigmadsf(1,:,id,fd) = dSigmadsf(1,:,id,fd) -[phiXDdphiXdsf(:,id,fd);0]';
         
         % finally now got all the bits compute the gradient of the 
         % evidence w.r.t. the sf's
         dJ2dsf(id,fd)=dEDdsf(id,fd) - 1/2*trace(invSigma*dSigmadsf(:,:,id,fd));
         
      end % loop id
   end % loop fd

   if ( ologevidence > logevidence ) nu=nu/2; end;
   osf=sf;
   sf=sf + nu*dJ2dsf;  % simple gradient step -- N.B. to max Ed

   
   fprintf('%d) %g %g %g\n',i,logevidence,norm(osf-sf)/norm(sf),norm(dJ2dsf)*nu);
   if( logevidence>=ologevidence & norm(osf-sf)/norm(sf)<1e-6 & norm(dJ2dsf)*nu<1e-4 ) 
      break; 
   end;
   
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
   if ( Y(i)==1 ) 
      fX(:,:,i)=fX(:,:,i)*(eye(indim,indim)+abs(randn)/20*d*d'/(d'*d)); 
   end; 
end

% BCI data
z=loadprep(bci('hm'),1); 
z=filterfft(z,'mode','bandpass','bands',[3 6 35 40],'downsample',true);
X=permute(z.x,[3 2 1]);Y=z.y;
[nSamp,indim,N]=size(X); 
fX=rcovFilt(X); trnIdx(2:2:N,1)=true;

% CSP seeding..
XX1=mxcat(X(:,:,trnIdx & Y==Y(1)),3); XX1=XX1'*XX1; 
XX=mxcat(X(:,:,trnIdx),3);            XX =XX'*XX;
[U,D]=eig(XX1,XX);
[sD,sI]=sort(abs(.5-diag(D)));D=diag(D);D=D(sI);U=U(:,sI);
U=U*diag(sign(sum(U,1))./sqrt(sum(U.*U,1)));  % normalise the vectors

featDim=2;
sf0=U(:,end-featDim+1:end); sf=sf0;
[w,b,sf]=evidenceCSPLR(fX(:,:,trnIdx),Y(trnIdx),sf0);
[w,b,sf]=evidenceCSPLR(fX(:,:,trnIdx),Y(trnIdx),randn(size(sf0)));

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

% svm direction opt
wb=nonLinConjGrad(@(w) primalL2SVMFn(w,phiX(:,trnIdx)',Y(trnIdx),cost(1)),[w0;b0],'plot',0,'verb',1,'maxEval',5000,'maxIter',inf);w=wb(1:end-1);b=wb(end);