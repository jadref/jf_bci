function [wb,f,C] = l1lr_prox3(X,Y,C,varargin)
% proximal opt to solve the l1 reg least squares problem
%
% [wb,f] = proximalOpt(X,Y,C,varargin)
% 
%  J = \min_{w,b} |(X*w+b)-y|^2 + C |w|
%
%Options:
%  lineSearchStep  - [4x1] [initStepSize stepIncFactor stepDecFactor minStep maxStep]
opts=struct('stepSize',[],'maxIter',5000,'verb',0,'wb',[],'alphab',[],...
            'objTol',0,'tol',0,'tol0',1e-4,'objTol0',0,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',6,'structMx',[],'dim',[],...
            'lipzApprox','maxeig','stepLim',[.01 1000],'stepLearnRate',[.5 2],'minIter',10);
opts=parseOpts(opts,varargin);
if ( numel(opts.lineSearchStep)<5 ) opts.lineSearchStep(end+1:5)=1; end;
szX=size(X); X=reshape(X,[],size(X,ndims(X))); % get orginal size, then reshape into [feat x examples]
wb=opts.wb; 
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) ) wb=zeros(size(X,1)+1,1); end
if ( numel(wb)==size(X,1) ) wb=[wb(:);0]; end;
if ( size(Y,2)==numel(Y) ) Y=Y'; end;

structMx=opts.structMx; 
% N.B. string or single entry struct matrix are type of structure matrix to make
if ( ~isempty(structMx) && (isstr(structMx) || numel(structMx)==1) ) 
  structMx=mkStructMx(szX(1:end-1),structMx);
  structMx=reshape(structMx,[],size(structMx,ndims(structMx))); % work with vector X
  if ( sum(structMx(:)==0)>numel(structMx)/2 ) structMx=sparse(structMx); end;
end

iH=opts.stepSize; 
if( isempty(iH) ) % stepSize = Lipschiz constant = approx inv hessian
if ( strcmp(opts.lipzApprox,'hess') )
  % diag-hessian approx for Lipsitch constants?
  ddL = [sum(X.*X,2);size(X,2)];
  l=1;
  ss=1./sqrt(ddL(1:end-1));
  w=ss.*X(:,1); l=sqrt(w'*w); for i=1:10; w=ss.*(X*((w.*ss)'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  niH= 1./(.25/2); % approx norm of the inverse hessian
  iH = 1./(l*ddL)/2;
  if ( ~isempty(structMx) ) % pre-group hessian
    giH= sqrt(double(iH(1:end-1).^2)'*structMx)'/2; % norm in each group
    %iH = [structMx*giH; iH(end)]; % scaled back to full space size
  end
else
  % est of largest eigenvalue of the data covariance matrix
  sX=sum(X,2); N=size(X,2); 
  w=X(:,1);l=sqrt(w'*w); for i=1:3; w=(X*(w'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  niH= 1./(.25/2);
  iH = 1/l/2;
  giH= iH; % group inverse hessian
end;
end;

nFeat=[]; if ( C<0 ) nFeat=-C; C=0; end

% pre-comp some solution info
wX = (wb(1:end-1)'*X)';
f  = wX + wb(end);
g  = 1./(1+exp(-Y.*f));
niH= mean(1./(g.*(1-g))); % est of scale of diag hessian
L  = sum(-log(max(g,eps)));
Yerr= Y.*(1-g);
dL = [-X*Yerr; -sum(Yerr)];
if ( ~isempty(structMx) )
  nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
  dRw = (structMx*(1./(nrm+double(nrm==0)))).*wb(1:end-1); % for convergence testing
else
  nrm = abs(wb(1:end-1));
  dRw = sign(wb(1:end-1)); % for convergence testing
end
R  = sum(nrm);
dw2=norm(dL(1:end-1)+C*dRw); % norm of the gradient
dw20=dw2;

lineSearchStep=opts.lineSearchStep(1);
step=1; 
ostep=step;
J=inf; dJ0=0;
% Information about the solution prior to the prox step, i.e. the basis for the prox step
w0  = wb; fw0 = f; Lw0 = L; dLw0= dL; LLq=1;
% Information about the iterated solutions, i.e. after the prox step, but before any acceleration is applied
owX = zeros(size(wX)); owb=zeros(size(wb));  of=zeros(size(f));
oowX= owX;             oowb=owb;             oof=of;
wbstar=wb; Jstar=J; % best solution so-far
oC=C; zeroFeat=false; actFeat=wb(1:end-1)~=0;
nsteps=0; % number of actual descent steps, i.e. excluding the backtracking steps
for iter=1:opts.maxIter;
  bt=false;
  oJ=J; % prev objective
   
  % check the prox-step was valid
  wX  = (wb(1:end-1)'*X)';
  f   = wX+wb(end);
  g   = 1./(1+exp(-Y.*f));
  L   = sum(-log(max(g,eps)));
  J   = C*R+L;
  % Keep track of best solution so far... as acceleration makes objective decrease non-monotonic
  if ( J<Jstar ) 
    if ( opts.verb>1 ) 
      fprintf('%3d*)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',...
              nsteps,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(step),lineSearchStep,...
              C,sum(actFeat),sum(~zeroFeat(:)));    
    end
    wbstar=wb; Jstar=J; 
  end; 
  dwb = wb-w0;
  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*Lip*(w-w0)^2 
  %  iff step<1/L then L(w,w0) <= L(w0)+dL'*(w-w0)+1/2(w-w0).^2/step
  %  thus using 1/2*Lip*(w-w0)^2 <= 1/2*(w-wo)^2/step makes Lq larger 
  %  and the quad requirement *stricter* than necessary
  Lq = Lw0 + dLw0'*dwb + 1/2/step*((dwb./(iH*niH))'*dwb);  % if don't have rho
  oLLq=LLq; LLq=L/Lq; % info for estimating the correct step size
  if ( opts.verb>1 ) fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g\n',nsteps,step,L,Lq,L/Lq); end;
  if ( Lq < L*(1-1e-3) ) % **Backtrack** on the prox step
    bt=true;
    if ( opts.verb>0 ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g -- Warning: Step size invalid!\n',nsteps,step,L,Lq,L/Lq); 
    end;
    % estimate the intercept we want....
    % shrink rapidly if we over-shoot
    ostep= step; step = max(opts.stepLim(1),step*opts.stepLearnRate(1));
    wb   = w0;% undo the previous step
    f    = fw0;
    L    = Lw0;
    niH  = niHw0; % back-track the scaling also
    err  = f-Y;    
  else % only try acceleration if the step size was valid
    nsteps=nsteps+1;
    % estimate the new step size to achieve the bound on validity of the prox step
    if ( nsteps<2 )
      ostep= step;
      step = min(opts.stepLim(2),step*opts.stepLearnRate(2));
    elseif( LLq<(1-1e-3) ) % newton search for the correct step size
      dstep = abs((step-ostep)/(LLq-oLLq))*(1-LLq)/2; % 1/2 the gap to 1
      ostep = step;
      step  = min([opts.stepLim(2),step*opts.stepLearnRate(2),step+dstep]);
    end
    
    % acceleration step
    if ( opts.lineSearchAccel ) 
      % track the gradient bits, (x_{k-1}-x_{k-2}), N.B. these are 
      % N.B. save the solutions *after* the prox but before the acceleration step!!
      oof =of;  of =f;
      oowb=owb; owb=wb; 
      if ( nsteps>2 ) 
        accstep = (iter-1)/(iter+2)*lineSearchStep;
        wb = wb  +(wb-oowb)*accstep; % N.B. watch for the gota about the updated of=f... owb=wb etc...
        f  = f   +(f-oof)*accstep;    
        g  = 1./(1+exp(-Y.*f));
        niH= 1./mean(g.*(1-g)); % approx scale of the hessian
        L   = sum(-log(max(g,eps)));        
      end
    end
    % finish evaluating the new starting point    
    if ( opts.verb>0 ) % only needed for progress reporting
      if ( ~isempty(structMx) )
        nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
        % N.B. could make a little more comp efficient by only using bits which have >0 nrm
        dRw = (structMx*(1./(nrm+double(nrm==0)))).*wb(1:end-1); % for convergence testing
      else
        nrm = abs(wb(1:end-1));
        dRw = sign(wb(1:end-1)); % for convergence testing
      end
      R  = sum(nrm);
    end
    Yerr= Y.*(1-g);
    dL  = [-X*Yerr; -sum(Yerr)];
    % save info on the starting point
    niHw0=niH;
    w0  = wb;
    fw0 = f;
    Lw0 = L;
    dLw0= dL;
  end    
  J  = C*R + L;
  
  dw2=norm(dL(actFeat)+C*dRw(actFeat)); % norm of the gradient
  % progress reporting
  if ( opts.verb>0 )
    fprintf('%3d(%d)\twb=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p=%d\n',iter,nsteps,sprintf('%5.3f ',wb(1:min(end,3))),L,R,C,J,norm(owb-wb),dw2,min(step),sum(actFeat),sum(~zeroFeat(:)));    
  end
     
  %-------------------------------------------------------------------------------
  % start/end Prox step
  % Generalised gradient descent step on C*R+L
  % i.e. wb = prox_(step*iH){ w0 - step*iH*dL(w0) }
  dw = step.*iH.*niH.*dL; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( ~isempty(structMx) )
    nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
    %wb(1:end-1) = wb(1:end-1).*(structMx*(1./nrm)); % normalize
    onrm=nrm;
    nrm = max(0,nrm-C*step*giH.*niH); % shrink, new group norm
    wb(1:end-1) = double(wb(1:end-1)).*(structMx*(nrm./onrm));% apply shrinkage
    dRw = (structMx*(1./(nrm+double(nrm==0)))).*wb(1:end-1); % for convergence testing
  else
    nrm =abs(wb(1:end-1));
    nrm =max(0,nrm-C*step*iH(1:max(1,end-1)).*niH); % shrink
    wb(1:end-1)=sign(wb(1:end-1)).*nrm; % prox step on R
    dRw = sign(wb(1:end-1)); % for convergence testing
  end
  R  = sum(nrm);
  % N.B. wb=x_k
  actFeat=wb(1:end-1)~=0;
  
  % convergence testing
  if ( nsteps<3 ) dw20=max(dw20,dw2); end;
  if ( nsteps>1 ) dJ0=max(dJ0,abs(J-oJ)); end;
  if ( ~bt && nsteps>opts.minIter && abs(oC-C)./max(1,abs(C))<5e-3 && ...
       (abs(J-oJ)<opts.objTol || abs(J-oJ)<opts.objTol0*dJ0 || ...
        norm(owb-wb)./max(1,norm(wb))<opts.tol || dw2<opts.tol0*dw20 ) ) 
    break; 
  end;
  
end

wb = wbstar; % return best solution found
f  = (wb(1:end-1)'*X)' + wb(end);
g  = 1./(1+exp(-Y.*f));
L  = sum(-log(max(g,eps)));
if ( ~isempty(structMx) )
  nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
else
  nrm =abs(wb(1:end-1));
end
R  = sum(nrm);
J  = C*R + L;
  
% progress reporting
if ( opts.verb>=0 )
    fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p=%d\n',...
            iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,C,J,...
            norm(owb-wb),dw2,min(step),sum(actFeat),sum(~zeroFeat(:)));       
end
return

function testCase()
wtrue=randn(100,1); % true weight
X    =randn(size(wtrue,1)*2,1000); % 1/2 relevant, 1/2 not
Ytrue=wtrue'*X(1:size(wtrue,1),:); % true pred
Y    =sign(Ytrue + randn(size(Ytrue))*1e-3); % noisy pred
wb0  =randn(size(X,1),1);

wb = l1lr_prox(X,Y,-10); % with target num features
% with a seed solution to test the C determination
wb = l1lr_prox(X,Y,-10,'wb',wb0); % with target num features

[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'stepLim',[.1 1024],'lipzApprox','maxeig','lineSearchAccel',0,'maxIter',500);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'stepLim',[.1 1024],'lipzApprox','maxeig','lineSearchAccel',1);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'stepLim',[.1 10240],'lipzApprox','hess','lineSearchAccel',0);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'stepLim',[.1 10240],'lipzApprox','hess','lineSearchAccel',1);


% with ill-conditioning
Xbad = repop(X,'*',[10;ones(size(X,1)-1,1)]);
[wb,f,J]=l1lr_prox3(Xbad,Y,1,'verb',1,'stepLim',[.1 4],'lipzApprox','hess');

% with a warm start to test the C determination
wb10 = l1lr_prox3(X,Y,-10); 
wb12 = l1lr_prox3(X,Y,-12,'wb',wb10); 
wb   = l1lr_prox3(X,Y,-12); 

% with a structure matrix
structMx=mkStructMx([4 size(X,1)/4],2); structMx=reshape(structMx,[size(X,1) size(structMx,3)]);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'structMx',structMx,'stepLim',[1 1],'lineSearchAccel',0);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'structMx',structMx,'stepLim',[1 1],'lineSearchAccel',1);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4],'lipzApprox','hess');

% with ill-conditioning
Xbad=X; Xbad(1,:,:)= Xbad(1,:,:)*10; 
[wb,f,J]=l1lr_prox(Xbad,Y,1,'verb',1,'structMx',structMx);
[wb,f,J]=l1lr_prox2(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4]);
[wb,f,J]=l1lr_prox3(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4],'lipzApprox','hess');

% with harder problem
strue=randn(10,1).^2;
utrue=randn(40,size(strue,1)); utrue=repop(utrue,'./',sqrt(sum(utrue.^2))); 
vtrue=randn(50,size(strue,1)); vtrue=repop(vtrue,'./',sqrt(sum(vtrue.^2))); 
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
X    =Xtrue + randn(size(Xtrue))*1e-1;
wb0  =randn(size(X,1),size(X,2));

%
[wb,f,J]=l1lr_prox(X,Y,1,'verb',1);
[wb,f,J]=l1lr_prox2(X,Y,1,'verb',1,'stepLim',[.1 4]);
[wb,f,J]=l1lr_prox3(X,Y,1,'verb',1,'stepLim',[.1 4],'lipzApprox','hess','lineSearchAccel',0);
[wb,f,J]=l1lr_prox3(Xbad,Y,1,'verb',1,'stepLim',[.1 4],'lipzApprox','maxeig');

% with ill-conditioning
Xbad=X; Xbad(1,:,:)= Xbad(1,:,:)*1000; 
[wb,f,J]=l1lr_prox(Xbad,Y,1,'verb',1);
[wb,f,J]=l1lr_prox2(Xbad,Y,1,'verb',1,'stepLim',[.1 4]);
[wb,f,J]=l1lr_prox3(Xbad,Y,1,'verb',1,'stepLim',[.1 32],'lipzApprox','hess');
[wb,f,J]=l1lr_prox3(Xbad,Y,1,'verb',1,'stepLim',[.1 32],'lipzApprox','maxeig');



% weird solution problems.
tic,[wb,f,Jlr]=l1lr_prox(X,Y,50,'verb',1,'lineSearchAccel',0,'maxIter',2,'tol',0,'lineSearchIter',20);toc
% but now this function doesn't seem to obey its taylor approximation.
H=2*[X*X'      sum(X,2);... % N.B. don't forget the factor of 2
     sum(X,2)' size(X,2)];
rho=max(abs(eig(H)));
wX =(wb(1:end-1)'*X)';
f  =wX+wb(end);
err=f-Y(:);
Lw0=err'*err;
dL =[2*X*err; 2*sum(err)]; % = [X*((w'*X+b)'-Y); 1'*((w'*X+b)'-Y)]