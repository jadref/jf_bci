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
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',6,'structMx',[],'dim',[]);
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

rho=opts.stepSize; 
if( isempty(rho) ) % stepSize = Lipschiz constant = approx inv hessian
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
  w=X(:,1);b=1; 
  l=sqrt(w'*w+b*b); for i=1:10; w=(X*(w'*X)'+sX*b)/l;b=(sX'*w+N*b)/l; l=sqrt(w'*w+b*b); end; 
  toc
  rho=l*2.2; % N.B. Don't forget the factor of 2!
  %rho=2*max(abs(eig(H)));
  %rho=sqrt(X(:)'*X(:));  
end;

nFeat=[]; if ( C<0 ) nFeat=-C; C=0; end

% pre-comp some solution info
wX = (wb(1:end-1)'*X)';
f  = wX + wb(end);
err= f-Y;    
L  = err'*err;
dL =zeros(size(wb));
R  = sum(abs(wb));

lineSearchStep=opts.lineSearchStep(1);
step=1/rho; ostep=step;
J=inf; 
w0=wb; owX = zeros(size(wX)); owb=zeros(size(wb));  of=zeros(size(f));
oowX= owX;             oowb=owb;             oof=of;
oC=C; zeroFeat=false; actFeat=wb(1:end-1)~=0;
for iter=1:opts.maxIter;
  bt=false;
  oJ=J; % prev objective

  % line search acceleration
  of =f;
   
  % check the prox-step was valid
  w0X= wX;
  wX  = (wb(1:end-1)'*X)';
  fw0 = f;
  f   = wX+wb(end);
  err = f-Y;
  Lw0 = L;
  L   = err'*err;
  dLw0= dL;
  dwb = wb-owb;
  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*Lip*(w-w0)^2 
  %  iff step<1/L then L(w,w0) <= L(w0)+dL'*(w-w0)+1/2(w-w0).^2/step
  %  thus using 1/2*Lip*(w-w0)^2 <= 1/2*(w-wo)^2/step makes Lq larger 
  %  and the quad requirement *stricter* than necessary
  %Lq = Lw0 + dLw0'*dwb + 1/2*rho*(dwb'*dwb); % if have rho available
  Lq = Lw0 + dLw0'*dwb + 1/2/step*(dwb'*dwb);  % if don't have rho
  fprintf('%d) step=%g  L=%g  Lq=%g\n',iter,step*rho,L,Lq);
  if ( Lq < L ) % **Backtrack** on the prox step
    bt=true;
    fprintf('Warning: Step size invalid!\n');
    step = max(1/rho,step./2);     % *rapidly* reduce the step size
    wb   = w0;% undo the previous step
    f    = fw0;
    L    = Lw0;
    err  = f-Y;    
  else % only try acceleration if the step size was valid
    oowb=owb; owb=wb; % N.B. save the solutions *after* the prox but before the acceleration step!!
    %if ( Lq*1.001 < L ) step = step*1.2; end; % *slowly* reduce step size....
    step=min(step*1.1,4/rho);
    
    % finish evaluating the new solution
    R  = sum(abs(wb));
    dL = 2*[X*err;            sum(err)]; 
    dR = [sign(wb(1:end-1));0 ];
  end    
  J  = C*R + L;
  
  % progress reporting
  if ( opts.verb>0 )
    fprintf('%3d)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(step),lineSearchStep,C,sum(actFeat),sum(~zeroFeat(:)));    
  end
     
  % Generalised gradient descent step on C*R+L
  dw = step*dL; 
  w0 = wb;  
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( ~isempty(structMx) )
    nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
    %wb(1:end-1) = wb(1:end-1).*(structMx*(1./nrm)); % normalize
    snrm= max(0,1-double(step*C)./nrm); % shrink
    wb(1:end-1) = double(wb(1:end-1)).*(structMx*snrm);% apply shrinkage
  else
    wb(1:end-1)=sign(wb(1:end-1)).*max(0,(abs(wb(1:end-1))-step*C)); % prox step on R
    %wb(1:end-1)=wb(1:end-1).*max(0,(1-step*C./abs(wb(1:end-1)))); % prox step on R
  end
  % N.B. wb=x_k
  actFeat=wb(1:end-1)~=0;
  
  % convergence testing
  if ( ~bt && iter>1 && abs(oC-C)./max(1,abs(C))<5e-3 && ...
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
  fprintf('%3d)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(step),lineSearchStep,C,sum(actFeat),sum(~zeroFeat(:)));    
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