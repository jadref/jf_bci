function [wb,f,C] = lsigmalr3d_prox3(X,Y,C,varargin)
% proximal opt to solve the l1 reg least squares problem
%
% [wb,f] = proximalOpt(X,Y,C,varargin)
% 
%  J = \min_{w,b} |(X*w+b)-y|^2 + C |w|
%
%Options:
%  lineSearchAccel - [bool] do we use line-search acceleration? (true)
%  lineSearchStep  - [4x1] [initStepSize stepIncFactor stepDecFactor minStep maxStep]
%  maxIter
%  verb
%  wb, alphab
%  objTol, objTol0
%  tol
%  dim
%  lipzApprox - {str} one-of: 'hess' - diag hessian approx,  'maxeig' - max-eigenvalue approx
%  stepLim    - [2x1] min/max limits for the step size in each iteration ([.01 8])
%  symDim     - [2 x 1] if non-empty which dimensions of X should be treated as symetric ([])
%               N.B. *DO NOT USE* iteration is numerically unstable -- probably because eig is unstable...
%  minIter    - [int] minimum number of iterations to run before testing convergence     (10)
%                     (prevent early exit when using re-seeding)
opts=struct('stepSize',[],'maxIter',5000,'verb',0,'wb',[],'alphab',[],...
            'objTol',0,'tol',0,'tol0',1e-3,'objTol0',1e-3,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',6,'structMx',[],'dim',[],...
            'lipzApprox','maxeig','stepLim',[.01 1024],'stepLearnRate',[.5 1.3],'symDim',[],'minIter',10);
opts=parseOpts(opts,varargin);
if ( numel(opts.lineSearchStep)<5 ) opts.lineSearchStep(end+1:5)=1; end;
szX=size(X); X=reshape(X,[],size(X,ndims(X))); % get orginal size, then reshape into [feat x examples]
wb=opts.wb; 
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) ) wb=zeros(size(X,1)+1,1); end
if ( numel(wb)==size(X,1) ) wb=[wb(:);0]; end;
if ( size(Y,2)==numel(Y) ) Y=Y'; end;

iH=opts.stepSize; 
if( isempty(iH) ) % stepSize = Lipschiz constant = approx inv hessian
  if ( strcmp(opts.lipzApprox,'hess') )
    % diag-hessian approx for Lipsitch constants?
    ddL= [sum(X.*X,2);size(X,2)];
    l=1;
    ss = 1./sqrt(ddL(1:end-1));
    w  = ss.*X(:,1); l=sqrt(w'*w); for i=1:3; w=ss.*(X*((w.*ss)'*X)')/l;l=sqrt(w'*w); end; w=w/l;
    niH= 1./(.25/2); % approx norm of the inverse hessian
    iH = 1./(l*ddL);
  else
    % est of largest eigenvalue of the data covariance matrix
    sX =sum(X,2); N=size(X,2); 
    w  =X(:,1);l=sqrt(w'*w); for i=1:3; w=(X*(w'*X)')/l;l=sqrt(w'*w); end; w=w/l;
    niH= 1./(.25/2); % approx norm of the inverse hessian
    iH = 1/(l);
    giH= iH; % group inverse hessian
  end;
end;

nFeat=[]; if ( C<0 ) nFeat=-C; C=0; end

% pre-comp some solution info
wX = (wb(1:end-1)'*X)';
f  = wX + wb(end);
g  = 1./(1+exp(-Y.*f));
L  = sum(-log(max(g,eps)));
Yerr= Y.*(1-g);
dL = [-X*Yerr; -sum(Yerr)];
% decompose the weight vector
[U,s,V]=decomp3d(wb,szX,opts.symDim);
% compute the regularisation cost
R  = sum(abs(s(:)));
actFeat=abs(s(:))>eps;

lineSearchStep=opts.lineSearchStep(1);
step=1; 
ostep=step;
J=inf; 
% Information about the solution prior to the prox step, i.e. the basis for the prox step
w0  = wb; fw0 = f; Lw0 = L; dLw0= dL; LLq=1; dJ0=0;
% Information about the iterated solutions, i.e. after the prox step, but before any acceleration is applied
owX = zeros(size(wX)); owb=zeros(size(wb));  of=zeros(size(f));
oowX= owX;             oowb=owb;             oof=of;
wbstar=wb; Jstar=J; % best solution so-far
oC=C; 
zeroFeat=false;
nsteps=0;
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
  if ( J<Jstar ) wbstar=wb; Jstar=J; end; 
  dwb = wb-w0;
  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*Lip*(w-w0)^2 
  %  iff step<1/L then L(w,w0) <= L(w0)+dL'*(w-w0)+1/2(w-w0).^2/step
  %  thus using 1/2*Lip*(w-w0)^2 <= 1/2*(w-wo)^2/step makes Lq larger 
  %  and the quad requirement *stricter* than necessary
  %Lq = Lw0 + dLw0'*dwb + 1/2*rho*(dwb'*dwb); % if have rho available
  Lq = Lw0 + dLw0'*dwb + 1/2/step*((dwb./(iH*niH))'*dwb);  % if don't have rho
  oLLq=LLq; LLq=L/Lq; % info for estimating the correct step size
  if ( opts.verb>1 ) fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g\n',iter,step,L,Lq,L/Lq); end;
  if ( Lq < L*(1-1e-3) ) % **Backtrack** on the prox step
    bt=true;
    if ( opts.verb>0 ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: step size invalid!\n',iter,step,L,Lq,L/Lq); 
    end;
    %if ( Lq < L*(1-1e-5) ) % shrink rapidly
      step = max(opts.stepLim(1),step*opts.stepLearnRate(1));     % *rapidly* reduce the step size
    %else % shrink slowly
    %  step = max(opts.stepLim(1),step*.9);
    %  %step = max(opts.stepLim(1),step*(Lq/L).^2); % grow/shrink proportionally to error
    %end
    wb   = w0;% undo the previous step
    f    = fw0;
    niH  = niHw0; % back-track the scaling also
    L    = Lw0;
  else % only try acceleration if the step size was valid
    if ( Lq < L && opts.verb>0 ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: marginally stable!\n',iter,step,L,Lq,L/Lq); 
    end;
    nsteps=nsteps+1; % track number of actual valid descent steps
    if ( nsteps<2 )
      ostep= step;
      step = min(opts.stepLim(2),step*opts.stepLearnRate(2));
    elseif( LLq<(1-1e-4) ) % newton search for the correct step size
      dstep = abs((step-ostep)/(LLq-oLLq))*(1-LLq)/2; % 1/2 the gap to 1
      ostep = step;
      step  = min([opts.stepLim(2),step*opts.stepLearnRate(2),step+dstep]);
    end
    % %if ( Lq*1.001 < L ) step = step*1.2; end; % *slowly* reduce step size....
    % if ( Lq < L*(1+1e-4) ) % close to limit
    %   step = min(opts.stepLim(2),step*(Lq/L).^2); % grow/shrink proportionally to error
    %   if ( opts.verb>1 ) % don't increase step size if marginally stable
    %     fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: marginally stable!\n',nsteps,step,L,Lq,L/Lq); 
    %   end
    % else % grow fast
    %   step=min(opts.stepLim(2),step*opts.stepLearnRate(2));        
    % end;

    % acceleration step
    if ( opts.lineSearchAccel ) 
      % track the gradient bits, (x_{k-1}-x_{k-2}), N.B. these are 
      % N.B. save the solutions *after* the prox but before the acceleration step!!
      oof =of;  of =f;
      oowb=owb; owb=wb; 
      if ( nsteps>2 ) 
        accstep = (iter-1)/(iter+2)*lineSearchStep;
        wb  = wb  +(wb-oowb)*accstep; % N.B. watch for the gota about the updated of=f... owb=wb etc...
        f   = f   +(f-oof)*accstep;    
        g   = 1./(1+exp(-Y.*f));
        niH= 1./mean(g.*(1-g)); % approx scale of the hessian
        L   = sum(-log(max(g,eps)));        
      end
    end
    % finish evaluating the new starting point
    if ( opts.verb>0 ) % only reallyed needed for progress reporting
      % decompose the weight vector
      [U,s,V]=decomp3d(wb,szX,opts.symDim);
      R  = sum(abs(s(:)));
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
  
  % progress reporting
  if ( opts.verb>0 )
    fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p#act=%d\r',...
            iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,...
            norm(owb-wb),0,min(step),sum(actFeat),sum(~zeroFeat(:)));    
  end
     
  %-------------------------------------------------------------------------------
  % start/end Prox step
  % Generalised gradient descent step on C*R+L
  % i.e. wb = prox_(step*iH){ w0 - step*iH*dL(w0) }
  dw = step.*iH.*niH.*dL; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  % decompose the weight vector
  [U,s,V]= decomp3d(wb,szX,opts.symDim);
  s(:)   = sign(s(:)).*max(0,(abs(s(:))-C*step*iH*niH)); % prox step on s  
  W  = recomp3d(U,s,V,szX); % re-compose the shrunk solution
  wb(1:end-1)=W(:);% insert back into running soution
  % N.B. wb=x_k
  actFeat=abs(s(:))>eps;
  R  = sum(abs(s(:)));
  
  % convergence testing
  if ( nsteps>1 ) dJ0=max(dJ0,abs(J-oJ)); end;
  if ( ~bt && nsteps>opts.minIter && abs(oC-C)./max(1,abs(C))<5e-3 && ...
       (abs(J-oJ)<opts.objTol || abs(J-oJ)<opts.objTol0*dJ0 || ...
        norm(owb-wb)./max(1,norm(wb))<opts.tol ) ) 
    break; 
  end;
  
end

wb = wbstar; % return best solution found
f  = (wb(1:end-1)'*X)' + wb(end);
g  = 1./(1+exp(-Y.*f));
L  = sum(-log(max(g,eps)));
[U,s,V]=decomp3d(wb,szX,opts.symDim);
R  = sum(abs(s(:)));
J  = C*R + L;
  
% progress reporting
if ( opts.verb>=0 )
  fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,norm(owb-wb),0,min(step),sum(actFeat),sum(~zeroFeat(:)));    
end
return

function [Us,ss,Vs]=decomp3d(wb,szX,symDim)
% Decompose the weight vector
for gi=1:szX(3);
  wgi=reshape(wb((1:prod(szX(1:2)))+(gi-1)*(prod(szX(1:2)))),szX(1:2));
  if ( isequal(symDim(:),[1;2]) )
    [U,s]  =eig(wgi); V=U';  s=diag(s); 
  else
    [U,s,V]=svd(wgi,'econ'); s=diag(s);
  end
  Us(:,:,gi)=U;
  ss(:,gi)  =s;
  Vs(:,:,gi)=V;
end
return;

function [W]=recomp3d(U,s,V,szX)
if ( nargin<3 || isempty(szX) ) szX=[size(U,1) size(V,1) size(s,2)]; end;
if ( ndims(U)<3 ) U=reshape(U,[size(U,1) size(U,2)/szX(3) szX(3)]); end;
if ( ndims(V)<3 ) V=reshape(V,[size(V,1) size(V,2)/szX(3) szX(3)]); end;
W=zeros(szX(1:3));
for gi=1:szX(3)
  W(:,:,gi)=U(:,:,gi)*diag(s(:,gi))*V(:,:,gi)'; 
end
return;



function testCase()
strue=randn(10,1).^2;
utrue=randn(40,size(strue,1)); utrue=repop(utrue,'./',sqrt(sum(utrue.^2))); 
vtrue=randn(50,size(strue,1)); vtrue=repop(vtrue,'./',sqrt(sum(vtrue.^2))); 
wtrue=utrue*diag(strue(:,1))*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2 3],Y,[4]);
X    =Xtrue + randn(size(Xtrue))*1e-1;

% simple test
[wb,f,J]=lsigmalr3d_prox3(X,Y,1,'verb',1);
[wb,f,J]=lsigmalr3d_prox3(X,Y,1,'verb',1,'stepLim',[.1 1],'lipzApprox','maxeig','lineSearchAccel',0,'maxIter',500);
[wb,f,J]=lsigmalr3d_prox3(X,Y,1,'verb',1,'stepLim',[.1 1],'lipzApprox','maxeig','lineSearchAccel',1);
[wb,f,J]=lsigmalr3d_prox3(X,Y,1,'verb',1,'stepLim',[.1 16],'lipzApprox','maxeig');
%[wb,f,J]=lsigmalr_prox3(X,Y,1,'verb',1,'stepLim',[.1 8],'lipzApprox','hess');

% now with stacked matrix features
strue=randn(10,2).^2;
wtrue=cat(3,utrue*diag(strue(:,1))*vtrue',utrue*diag(strue(:,2))*vtrue');% true weight
Xtrue=cat(3,tprod(wtrue,[1 2 3],Y,[4]),randn(size(utrue,1),size(vtrue,1),1,size(Y,1))); % irrelevant feat
X    =Xtrue + randn(size(Xtrue))*1e-1;

[wb,f,J]=lsigmalr3d_prox3(X,Y,1,'verb',1);
W=reshape(wb(1:end-1,1),[size(X,1),size(X,2),size(X,3)]);
clf;image3d(W,3);
for di=1:size(W,3); [s(:,di)]=svd(W(:,:,di)); lab{di}=sprintf('%d',di); end; 
clf;plot(s,'LineWidth',1);legend(lab);


% with ill-conditioning
Xbad=X; Xbad(1,:,:)= Xbad(1,:,:)*10; 
[wb,f,J]=lsigmalr_prox(Xbad,Y,1,'verb',1,'structMx',structMx);
[wb,f,J]=lsigmalr_prox2(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4]);
[wb,f,J]=lsigmalr_prox3(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4],'lipzApprox','hess');