function [wb,f,C] = proximalL1LS(X,Y,C,varargin)
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
            'objTol',0,'tol',0,'tol0',1e-3,'objTol0',5e-3,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',6,'structMx',[],'dim',[],...
            'lipzApprox','maxeig','stepLim',[.001 8],'stepLearnRate',[.5 1.1],'symDim',[],'minIter',10);
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
  error('Not supported yet!');
  % diag-hessian approx for Lipsitch constants?
  ddL= [sum(X.*X,2);size(X,2)];
  l=1;
  ss = 1./sqrt(ddL(1:end-1));
  tic
  w  = ss.*X(:,1); l=sqrt(w'*w); for i=1:10; w=ss.*(X*((w.*ss)'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  toc  
  iH = 1./(2*l*(ddL));
else
  % est of largest eigenvalue of the data covariance matrix
  sX=sum(X,2); N=size(X,2); 
  w=X(:,1);l=sqrt(w'*w); for i=1:10; w=(X*(w'*X)')/l;l=sqrt(w'*w); end; w=w/l;
  iH = 1/(l*2.2);
  giH= iH; % group inverse hessian
end;
end;

nFeat=[]; if ( C<0 ) nFeat=-C; C=0; end

% pre-comp some solution info
wX = (wb(1:end-1)'*X)';
f  = wX + wb(end);
err= f-Y;    
L  = err'*err;
dL = 2*[X*err;            sum(err)]; 
if ( isequal(opts.symDim(:),[1;2]) )
  [U,s]  =eig(reshape(wb(1:end-1),szX(1:end-1))); V=U';  s=diag(s); 
else
  [U,s,V]=svd(reshape(wb(1:end-1),szX(1:end-1)),'econ'); s=diag(s);
end
R  = sum(abs(s));
% compute the gradient of the regularisor (with 0 gradient for disabled directions)
actFeat= true(size(s));
dRwact = repop(sign(s(actFeat))','*',V(:,actFeat))'; % efficient when low rank
dw2    = U(:,actFeat)'*reshape(dL(1:end-1),szX(1:2)) + C*dRwact;
dw2    = sqrt(dw2(:)'*dw2(:));
dw20   = dw2;

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
oC=C; zeroFeat=false;
nsteps=0;
for iter=1:opts.maxIter;
  bt=false;
  oJ=J; % prev objective
   
  % check the prox-step was valid  
  wX  = (wb(1:end-1)'*X)';
  f   = wX+wb(end);
  err = f-Y;
  L   = err'*err;
  J   = C*R+L;
  % Keep track of best solution so far... as acceleration makes objective decrease non-monotonic
  if ( J<Jstar ) wbstar=wb; Jstar=J; end; 
  dwb = wb-w0;
  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*Lip*(w-w0)^2 
  %  iff step<1/L then L(w,w0) <= L(w0)+dL'*(w-w0)+1/2(w-w0).^2/step
  %  thus using 1/2*Lip*(w-w0)^2 <= 1/2*(w-wo)^2/step makes Lq larger 
  %  and the quad requirement *stricter* than necessary
  %Lq = Lw0 + dLw0'*dwb + 1/2*rho*(dwb'*dwb); % if have rho available
  Lq = Lw0 + dLw0'*dwb + 1/2/step*((dwb./iH)'*dwb);  % if don't have rho
  oLLq=LLq; LLq=L/Lq; % info for estimating the correct step size
  if ( opts.verb>1 ) fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g\n',iter,step,L,Lq,L/Lq); end;
  if ( Lq < L*(1-1e-3) ) % **Backtrack** on the prox step
    bt=true;
    if ( opts.verb>0 ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: step size invalid!\n',iter,step,L,Lq,L/Lq); 
    end;
    step = max(opts.stepLim(1),step*opts.stepLearnRate(1));     % *rapidly* reduce the step size
    wb   = w0;% undo the previous step
    f    = fw0;
    L    = Lw0;
    err  = f-Y;
  else % only try acceleration if the step size was valid
    if ( Lq < L && opts.verb>0 ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \t1-L/Lq=%g --- Warning: marginally stable!\n',iter,step,L,Lq,1-L/Lq); 
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
        err = f-Y;
        L   = err'*err;
      end
    end
    % finish evaluating the new starting point
    if ( opts.verb>0 ) % only reallyed needed for progress reporting
      if ( isequal(opts.symDim(:),[1;2]) ) % decompose this solution for obj computation
        [U,s]  =eig(reshape(wb(1:end-1),szX(1:end-1))); V=U';  s=diag(s); 
      else
        [U,s,V]=svd(reshape(wb(1:end-1),szX(1:end-1)),'econ'); s=diag(s);
      end    
      R  = sum(abs(s));
    end
    dL = 2*[X*err;            sum(err)]; 
    % save info on the starting point
    w0  = wb;
    fw0 = f;
    Lw0 = L;
    dLw0= dL;
  end    
  J  = C*R + L;
  
  if ( any(actFeat) )   
    dRwact = repop(sign(s(actFeat))','*',V(:,actFeat))'; % efficient when low rank
    dw2    = U(:,actFeat)'*reshape(dL(1:end-1),szX(1:2)) + C*dRwact;
    dw2    = sqrt(dw2(:)'*dw2(:));
  else
    dRwact=0; dw2=0;
  end
  
  % progress reporting
  if ( opts.verb>0 )
    fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p=%d\n',...
            iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,...
            norm(owb-wb),dw2,min(step),sum(actFeat),sum(~zeroFeat(:)));    
  end
     
  %-------------------------------------------------------------------------------
  % start/end Prox step
  % Generalised gradient descent step on C*R+L
  % i.e. wb = prox_(step*iH){ w0 - step*iH*dL(w0) }
  dw = step.*iH.*dL; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( isequal(opts.symDim(:),[1;2]) ) % decompose 
    [U,s]  =eig(reshape(wb(1:end-1),szX(1:end-1))); V=U';  s=diag(real(s)); % sym (pd?) features
  else % non-sym features
     if ( 0 && sum(actFeat)*4 < min(szX(1:end-1)) ) % use faster svds -- never faster!
        [U,s,V]=svds(reshape(wb(1:end-1),szX(1:end-1)),sum(actFeat)+2);
     else
        try
           [U,s,V]=svd(reshape(wb(1:end-1),szX(1:end-1)),'econ'); s=diag(s);
        catch % convert to double and try again if failed.. (sometimes happens on AMD)
           [U,s,V]=svd(double(reshape(wb(1:end-1),szX(1:end-1))),'econ'); s=diag(s);
           % ss=reshape(wb(1:end-1),szX(1:end-1));
           % save(sprintf('./SVD_failed%d',randi(1000)),'ss');
           % error('SVD failed to converge');
           % %[U,s,V]=svds(reshape(wb(1:end-1),szX(1:end-1)),min(szX(1:2))); s=diag(s);
        end
     end
  end
  s = sign(s).*max(0,(abs(s)-C*step*iH)); % prox step on s
  W = U*diag(s)*V'; 
  wb(1:end-1)=W(:);% re-construct solution
  % N.B. wb=x_k
  actFeat=s~=0;
  R  = sum(abs(s));
  
  % convergence testing
  if ( nsteps<3 ) dw20=max(dw20,dw2); end;
  if ( nsteps>1 ) dJ0=max(dJ0,abs(J-oJ)); end;
  if ( ~bt && nsteps>opts.minIter && abs(oC-C)./max(1,abs(C))<5e-3 && ...
       (abs(J-oJ)<opts.objTol || abs(J-oJ)<opts.objTol0*dJ0 || ...
        norm(owb-wb)./max(1,norm(wb))<opts.tol || dw2<opts.tol0*dw20 ) ) 
    break; 
  end;
  
end

wb= wbstar; % return best solution found
f  = (wb(1:end-1)'*X)' + wb(end);
err= f-Y;
if ( isequal(opts.symDim(:),[1;2]) ) % decompose this solution for obj computation
  [U,s]  =eig(reshape(wb(1:end-1),szX(1:end-1))); V=U';  s=diag(s); 
else
  [U,s,V]=svd(reshape(wb(1:end-1),szX(1:end-1)),'econ'); s=diag(s);
end    
R  = sum(abs(s));
L  = err'*err;
J  = C*R + L;
  
% progress reporting
if ( opts.verb>=0 )
  fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) = %5.4f \t|dw|=%5.3f,%5.3f \tL=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,norm(owb-wb),dw2,min(step),sum(actFeat),sum(~zeroFeat(:)));    
end
return

function testCase()
strue=randn(10,1).^2;
utrue=randn(40,size(strue,1)); utrue=repop(utrue,'./',sqrt(sum(utrue.^2))); 
vtrue=randn(50,size(strue,1)); vtrue=repop(vtrue,'./',sqrt(sum(vtrue.^2))); 
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
X    =Xtrue + randn(size(Xtrue))*1e-1;
wb0  =randn(size(X,1),size(X,2));

[wb,f,J]=lsigmals_prox(X,Y,1,'verb',1);
[wb,f,J]=lsigmals_prox3(X,Y,1,'verb',1,'stepLim',[1 1],'lipzApprox','maxeig','lineSearchAccel',0,'maxIter',500);
[wb,f,J]=lsigmals_prox3(X,Y,1,'verb',1,'stepLim',[1 1],'lipzApprox','maxeig','lineSearchAccel',1);
[wb,f,J]=lsigmals_prox3(X,Y,1,'verb',1,'stepLim',[.1 8],'lipzApprox','maxeig');
%[wb,f,J]=lsigmals_prox3(X,Y,1,'verb',1,'stepLim',[.1 8],'lipzApprox','hess');

% with ill-conditioning
Xbad=X; Xbad(1,:,:)= Xbad(1,:,:)*10; 
[wb,f,J]=lsigmals_prox(Xbad,Y,1,'verb',1,'structMx',structMx);
[wb,f,J]=lsigmals_prox2(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4]);
[wb,f,J]=lsigmals_prox3(Xbad,Y,1,'verb',1,'structMx',structMx,'stepLim',[.1 4],'lipzApprox','hess');