function [wb,f,C] = proximalL1LS(X,Y,C,varargin)
% proximal opt to solve the l1 reg least squares problem
%
% [wb,f] = proximalOpt(X,Y,C,varargin)
% 
%  J = \min_{w,b} |(X*w+b)-y|^2 + C |w|
%
%Options:
%  lineSearchStep  - [4x1] [initStepSize stepIncFactor stepDecFactor minStep maxStep]
opts=struct('stepSize',[],'maxIter',5000,'verb',0,'wb',[],'alphab',[],'objTol',1e-6,'tol',1e-5,'objTol0',0,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',0,'structMx',[],'dim',[],...
            'lipzApprox','maxeig');
opts=parseOpts(opts,varargin);
if ( numel(opts.lineSearchStep)<5 ) opts.lineSearchStep(end+1:5)=1; end;
szX=size(X); X=reshape(X,[],size(X,ndims(X))); % get orginal size, then reshape into [feat x examples]
wb=opts.wb; 
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) ) wb=zeros(size(X,1)+1,1); end
if (numel(wb)==size(X,1) ) wb=[wb(:);0]; end;
if (size(Y,2)==numel(Y) ) Y=Y'; end;

structMx=opts.structMx; 
% N.B. string or single entry struct matrix are type of structure matrix to make
if ( ~isempty(structMx) && (isstr(structMx) || numel(structMx)==1) ) 
  structMx=mkStructMx(szX(1:end-1),structMx);
  structMx=reshape(structMx,[],size(structMx,ndims(structMx))); % work with vector X
  if ( sum(structMx(:)==0)>numel(structMx)/2 ) structMx=sparse(structMx); end;
end

lip0=opts.stepSize; 
if( isempty(lip0) ) % stepSize = Lipschiz constant = approx inv hessian
if ( strcmp(opts.lipzApprox,'hess') )
  % diag-hessian approx for Lipsitch constants?
  ddL = [sum(X.*X,2);size(X,2)];
  %lip0 = ddL*(l./sqrt(sum((ddL(1:end-1).*w).^2)))*2.5;
  ss=1./sqrt(ddL(1:end-1));
  tic
  w=ss.*X(:,1); l=sqrt(w'*w); for i=1:10; w=ss.*(X*((w.*ss)'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  toc  
  lip0 = ddL*l*2.2;
else
  % est of largest eigenvalue of the data covariance matrix
  sX=sum(X,2); N=size(X,2); 
  % tic
  %   H=[X*X' sX;... % N.B. don't forget the factor of 2
  %      sX' N];
  %   w=[X(:,1);1]; l=sqrt(w'*w); for i=1:10; w=H*w/l; l=sqrt(w'*w); end; 
  % toc
  % tic
  %   XX=X*X';
  %   w=X(:,1);b=0; l=sqrt(w'*w+b*b); for i=1:10; w=(XX*w+sX*b)/l; b=(sX'*w+N*b);l=sqrt(w'*w+b*b); end; 
  % toc  
  tic
  w=X(:,1);l=sqrt(w'*w); for i=1:10; w=(X*(w'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  toc
  lip0 = l*2.2;
end;
end;

nFeat=[]; if ( C<0 ) nFeat=-C; C=0; end

lineSearchStep=opts.lineSearchStep(1);
step=1; ostep=step;
J=inf; owb=zeros(size(wb)); oowb=owb; oC=C; zeroFeat=false; actFeat=wb(1:end-1)~=0;
for iter=1:opts.maxIter;

  oJ=J; % prev objective

  % line search acceleration
  oowb=owb; owb=wb; % N.B. save the solutions *after* the prox but before the acceleration step!!
  if( opts.lineSearchAccel && norm(oC-C)./max(1,norm(C))<.05 && iter>2 ) 
    if ( opts.verb>0 ) fprintf('a'); end;
    % N.B. Accel gradient method:
    %  y = x_{k-1} + (k-2)/(k+1)*(x_{k-1}-x_{k-2})
    % x_k= prox_rho(y-dL(y)/rho);
    dwb= owb - oowb; % track the gradient bits, (x_{k-1}-x_{k-2})
    wb = owb + dwb*(iter-1)/(iter+2)*lineSearchStep; % wb=y
  end
  
  wX = (wb(1:end-1)'*X)';
  f  = wX + wb(end);
  err= f-Y;
  R = sum(abs(wb));
  L = err'*err;
  dL = 2*[X*err;            sum(err)]; % = [X*((w'*X+b)'-Y); 1'*((w'*X+b)'-Y)]
  dR = [sign(wb(1:end-1));0 ];
  J = C*R + L;
  
  % progress reporting
  if ( opts.verb>0 )
    fprintf('%3d)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(lip0),lineSearchStep,C,sum(actFeat),sum(~zeroFeat(:)));    
  end
  
  % adapt step size
  if ( norm(oC-C)./max(1,norm(C))<.05 && (oC*R+L)>=oJ && J>=oJ ) %*(1+1e-3) ) 
    if ( opts.verb>0 ) fprintf('*'); end;
    if (  opts.lineSearchAccel ) 
      lineSearchStep=max(opts.lineSearchStep(4),min(opts.lineSearchStep(5),lineSearchStep/opts.lineSearchStep(3)));
    end;
    %rho=rho*1.5; 
  else
    if ( opts.lineSearchAccel ) 
      lineSearchStep=max(opts.lineSearchStep(4),min(opts.lineSearchStep(5),lineSearchStep*opts.lineSearchStep(2)));
    end
  end
  
  % est the final number features?
  % N.B. need a slight margin on dL to identify marginally stable points... which should 0, but have 
  %  v.v.v.v. small gradient
  oC=C;
  dLg  =abs(dL(1:end-1))+2*lip0(1:max(1,end-1)).*min(abs(wb(1:end-1)),mean(abs(wb(1:end-1))));% 2nd order approx est of gradient if this feature had value=0% abs(wb(1:end-1));%
  dRg  =ones(size(dLg));%abs(sign(wb(1:end-1)))+(wb(1:end-1)==0);
  if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
    dRg = (double(dRg)'*structMx)';
    dLRg= sqrt((double(dLg(:)).^2)'*structMx)'./dRg(:); %N.B. only if all elm in group have same weight in structMx
    dLg = dLRg;
  else
    dLRg = abs(dLg);
    dRg  = ones(size(dLg));
  end
  if ( ~isempty(nFeat) ) % est what correct regularisor should be
    [sdLRg]=sort(dLRg,'descend'); 
    % target C is just big enough to prevent nFeat+1 features..
    if ( iter<=2 ) cFF=0; elseif( iter<200 ) cFF=.1; else cFF=.95; end;
    C=(C*(cFF)+sdLRg(nFeat+[0 1])'*[.5;.5]*(1-cFF)); % smooth out sudden value changes
    C=max(C,lip0*1e-8); % stop C=0 problems
  end
  zeroFeat = C*abs(dRg)>abs(dLg)*(1+1e-4);
    
  % Line search for the optimal step size
  step=1;
  if ( opts.lineSearchIter>0 )
  % pre-compute useful stuff
  maxStep=inf; minStep=1;
  step = minStep;
  %wX   = (wb(1:end-1)'*X)';%(wb(1:end-1)'*X)'; % ready computed
  dLXL  = ((dL(1:end-1)./lip0(1:max(1,end-1)))'*X)';
  wbn  = wb-step.*(dL./lip0);
  dRlin= [sign(wbn(1:end-1)).*min(abs(wbn(1:end-1)).*lip0(1:max(1,end-1))/step/C,1); 0];
  %dRlin= dL; % this is a bugger
  dRXL  = ((dRlin(1:end-1)./lip0(1:max(1,end-1)))'*X)';

  %f   = wX + wb(end);
  %err = f-Y;
  %Lw0 = err'*err;
  
  Lw0 = L; % orginal loss
  Lmin= Lw0; 
  stepmin=minStep;
  for ii=1:opts.lineSearchIter;
    % True loss with this step size
    err= (wX-step*dLXL-step*C.*dRXL+wb(end)-step/lip0(end)*dL(end))-Y;    
    Lii = err'*err;
    if ( ii==1 ) Lmin=Lii; end;
    % Taylor approx to the loss: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*(w-w0)'*L*(w-w0)
    % Given that w_lin ~= w0-dL/L-C/L*dRLin = w0 -(dL+C*dRlin)/L we have
    % L(w_lin,w0) = L(w0) - dL'(dL+C*dRlin)/L + 1/2*(dL+C*dRlin)'*1/L*(dL+C*dRlin)
    Lq = Lw0 - step*dL'*((dL+C*dRlin)./lip0) + step.*(((dL+C*dRlin)./lip0)'*(dL+C*dRlin))/2;
    if ( opts.verb>1) 
      fprintf('%d.%d) step=%7.6f\t <%7.6f,%7.6f>\tL=%5.3f Lq=%5.3f',iter,ii,step,minStep,maxStep,Lii,Lq);
    end
    if ( 1 || Lii < Lmin ) % valid good step
      if ( Lq <= Lii ) % invalid step : step is too big
        maxStep = min(maxStep,step);
        step    = min(maxStep,minStep+.6*(step-minStep));
        if ( opts.verb>1) fprintf('i\n'); end;
      else % valid step : could be bigger?
        Lmin=Lii; stepmin=step;
        minStep = max(minStep,step);
        step    = min(step*2,step+.6*(maxStep-step));
        if ( opts.verb>1) fprintf('v\n'); end;
      end
    else % worse than smallest so far....
      if ( opts.verb>1) fprintf('W\n'); end;
      maxStep=min(maxStep,step);
      step=stepmin+(.6*(step-stepmin)); % bi-sect window
    end
    if ( minStep>ostep*6 || (maxStep-minStep)<minStep*.1 ) 
      break; 
    end;
  end
  step=stepmin;
  if ( opts.verb>1 ) 
    fprintf('%d.%d) step*=%7.6f\t <%7.6f,%7.6f>\tL=%5.3f\n',iter,ii,step,minStep,maxStep,Lmin);
  end
  end
  
  % Generalised gradient descent step on C*R+L
  %wX   = (wb(1:end-1)'*X)';%(wb(1:end-1)'*X)'; % ready computed
  %f   = wX + wb(end);
  %err = f-Y;
  %Lw0 = L;%err'*err;
  % Gradient descent on L
  lip= step./lip0; 
  dw = lip(:).*dL; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( ~isempty(structMx) )
    nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
    if ( numel(lip)>1 )
      slip= sqrt(double(lip(1:max(1,end-1)).^2)'*structMx)';
    else
      slip=lip;
    end
    %wb(1:end-1) = wb(1:end-1).*(structMx*(1./nrm)); % normalize
    snrm= max(0,nrm-double(slip.*C)); % shrink
    wb(1:end-1) = double(wb(1:end-1)).*(structMx*(snrm./nrm));% apply shrinkage
  else
    wb(1:end-1)=sign(wb(1:end-1)).*max(0,(abs(wb(1:end-1))-lip(1:max(1,end-1))*C)); % prox step on R
  end
  % N.B. wb=x_k
  actFeat=wb(1:end-1)~=0;
  
  % check the validaty of the choosen step
  if ( 0 ) 
  wX = (wb(1:end-1)'*X)';
  err= wX+wb(end)-Y;    
  L  = err'*err;
  dwb = wb-owb;
  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*L*(w-w0)^2
  Lq = Lw0 + dL'*dwb + 1/2/(dwb.*lip)'*dwb;
  if ( Lq*(1.0001) < L ) 
    fprintf('Warning: Step size invalid!\n');
    %wb = owb+dwb/2;
  end  
  if ( opts.verb>0 ) 
    fprintf('%d.%d) step*=%7.6f\t <%7.6f,%7.6f>\tL=%5.3f Lq=%5.3f\n',iter,ii,step,minStep,maxStep,L,Lq);
  end
  end
  
  % convergence testing
  if ( iter>1 && abs(oC-C)./max(1,abs(C))<5e-3 && ...
       (abs(J-oJ)<opts.objTol || norm(owb-wb)./max(1,norm(wb))<opts.tol ) ) 
    break; 
  end;
  
end

  f  = (wb(1:end-1)'*X)' + wb(end);
  err= f-Y;
  R = sum(abs(wb));
  L = err'*err;
  dL = [X*err;            sum(err)]; % = [X*((w'*X+b)'-Y); 1'*((w'*X+b)'-Y)]
  dR = [sign(wb(1:end-1));0 ];
  J = C*R + L;
  
% progress reporting
if ( opts.verb>0 )
  fprintf('%3d)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(lip0),lineSearchStep,C,sum(actFeat),sum(~zeroFeat(:)));    
end
return

function testCase()
wtrue=randn(100,1); % true weight
X    =randn(size(wtrue,1)*2,1000); % 1/2 relevant, 1/2 not
Ytrue=wtrue'*X(1:size(wtrue,1),:); % true pred
Y    =sign(Ytrue + randn(size(Ytrue))*1e-3); % noisy pred
wb0  =randn(size(X,1),1);

wb = l1ls_prox(X,Y,1);
wb = l1ls_prox(X,Y,-10); % with target num features
% with a seed solution to test the C determination
wb = l1ls_prox(X,Y,-10,'wb',wb0); % with target num features

% with a warm start to test the C determination
wb10 = l1ls_prox(X,Y,-10); 
wb12 = l1ls_prox(X,Y,-12,'wb',wb10); 
wb   = l1ls_prox(X,Y,-12); 

% with a structure matrix
structMx=mkStructMx([4 size(X,1)/4],2); structMx=reshape(structMx,[size(X,1) size(structMx,3)]);
tic,[wb,f,J]=l1ls_prox(X,Y,-2,'verb',1,'structMx',structMx);toc  

% with stiff structure
wb = l1ls_prox(repop(X,'*',[10;ones(size(X,1)-1,1)]),Y,1,'verb',2,'lineSearchAccel',0,'lineSearchStep',0,'lineSearchIter',0,'maxIter',60,'lipzApprox','sphere');
wb = l1ls_prox(repop(X,'*',[10;ones(size(X,1)-1,1)]),Y,1,'verb',2,'lineSearchAccel',1,'lineSearchStep',0,'lineSearchIter',0,'maxIter',60,'lipzApprox','sphere');
wb = l1ls_prox(repop(X,'*',[10;ones(size(X,1)-1,1)]),Y,1,'verb',2,'lineSearchAccel',0,'lineSearchStep',0,'lineSearchIter',0,'maxIter',60,'lipzApprox','hess');
wb = l1ls_prox2(repop(X,'*',[10;ones(size(X,1)-1,1)]),Y,1,'verb',2,'lineSearchAccel',0,'lineSearchStep',0,'lineSearchIter',0,'maxIter',60);
% with structure
[wb,f,J]=l1ls_prox(repop(X,'*',[2;ones(size(X,1)-1,1)]),Y,1,'verb',1,'structMx',structMx,'verb',2,'lineSearchAccel',0,'lineSearchStep',0,'lineSearchIter',0,'maxIter',60);

% with pre-processed data
X2=sqrt(sum(X.^2,3));
dX=repop(X,'/',X2);
wb = l1ls_prox(dX,Y,1,'verb',2,'lineSearchAccel',0,'lineSearchStep',1,'lineSearchIter',0,'maxIter',60,'lipzApprox','sphere');
% vs. with dia hess
wb = l1ls_prox(dX,Y,1,'verb',2,'lineSearchAccel',0,'lineSearchStep',1,'lineSearchIter',0,'maxIter',60,'lipzApprox','hess');



% weird solution problems.
tic,[wb,f,Jlr]=l1ls_prox(X,Y,50,'verb',1,'lineSearchAccel',0,'maxIter',2,'tol',0,'lineSearchIter',20);toc
% but now this function doesn't seem to obey its taylor approximation.
H=2*[X*X'      sum(X,2);... % N.B. don't forget the factor of 2
     sum(X,2)' size(X,2)];
rho=max(abs(eig(H)));
wX =(wb(1:end-1)'*X)';
f  =wX+wb(end);
err=f-Y(:);
Lw0=err'*err;
dL =[2*X*err; 2*sum(err)]; % = [X*((w'*X+b)'-Y); 1'*((w'*X+b)'-Y)]

%dw =randn(size(wb))*1e-3;
%dw =dL/rho;
dw =-dL/rho;
steps=[0:.01:1];
for si=1:numel(steps);
  wb2=wb-dw*steps(si);
  f2 =(wb2(1:end-1)'*X)'+wb2(end);
  Lw2(si)=sum((f2-Y).^2);
  Lq2(si)=Lw0 + dL'*(wb2-wb) + rho/2*sum((wb2-wb).^2);
  LH2(si)=Lw0 + dL'*(wb2-wb) + 1/2*(wb2-wb)'*H*(wb2-wb);
end
fdiffH = (Lw2(1:end-2)-2*Lw2(2:end-1)+Lw2(3:end))./(mean(diff(steps)).^2.*dw'*dw);
clf;plot(steps,[Lw2;Lq2;LH2]');legend('Lw','Lq','LH');
fprintf('Lw2=%g\tLq2=%g\n',Lw2,Lq2);

% shrinkage
wb3=wb2;
wb3(1:end-1)=sign(wb3(1:end-1)).*max(0,(abs(wb3(1:end-1))-C/rho)); % prox step on R
f3 =(wb3(1:end-1)'*X)'+wb3(end);
Lw3=sum((f3-Y).^2);
Lq3=Lw0 + dL'*(wb3-wb) + rho/2*sum((wb3-wb).^2);

fprintf('Lw2=%g\tLq2=%g\tLw3=%g\tLq3=%g\n',Lw2,Lq2,Lw3,Lq3);