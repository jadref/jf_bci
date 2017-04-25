function [wb,f,C] = mmlsigmalr_prox3(X,Y,C,varargin)
% proximal opt to solve the l1 reg least squares problem
%
% [wb,f] = proximalOpt(X,Y,C,varargin)
% 
% J = C*\sum eig(w) + sum_i log( (1 + exp( - y_i ( w'*X_i + b ) ) )^-1 ) 
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
%  C       - trace norm regularisation constant (0)
%           N.B. C<0 -> -C is target solution rank
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
%  stepC, ratioC - [float] two constants used in the search for the correct regulasor strength to
%              fix for a certain output rank.  alphaC is the 'step-size' for updating the C parameter
%              ratioC is the target C based on the current solution.
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
opts=struct('stepSize',[],'maxIter',5000,'verb',0,'wb',[],'alphab',[],...
            'objTol',1e-5,'tol',0,'tol0',1e-4,'objTol0',1e-5,'eps',1e-8,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 1.4],'lineSearchIter',6,...
				'dim',[],'labdim',[],'ydim',[],...
            'lipzApprox','maxeig','stepLim',[.01 1024],'stepLearnRate',[.3 1.1],...
				'symDim',[],'minIter',10,...
				'nFeat',[],'stepC',1,'ratioC',.5); %adaptive step sizer to find optimal C
opts=parseOpts(opts,varargin);
if ( numel(opts.lineSearchStep)<5 ) opts.lineSearchStep(end+1:5)=1; end;

dim=opts.dim; if ( isempty(dim) ) dim=ndims(X); end;
labdim=opts.labdim; if ( isempty(labdim) ) labdim=ndims(X)-1; end;
szX=size(X); nd=numel(szX); N=szX(dim); nf=prod(szX(setdiff(1:end-1,[dim labdim])));
if( size(Y,1)==N && size(Y,2)~=N ) Y=Y'; end; % ensure Y has examples in last dim [L x N]
if( size(Y,1)==1 )
  if( all(Y(:)==-1 | Y(:)==0 | Y(:)==1 | isnan(Y(:))) ) % binary problem
	 Y=cat(1,Y,-Y); % make into 2 label version, with +1 first!
  elseif ( all(Y(:)>0 & Y(:)==ceil(Y(:))) ) % class labels input, convert to indicator matrix
	 Yl=Y;key=unique(Y);key(key==0)=[];
	 Y=zeros(numel(key),N);for l=1:numel(key); Y(l,:)=Yl==key(l); end;	 
  end
end
if ( size(Y,2)~=N ) error('Y should be [LxN]'); end;
Y(:,any(isnan(Y),1))=0; % convert NaN's to 0 so are ignored
L=size(Y,1);
% reshape X to be 3d for simplicity
if ( labdim~=ndims(X)-1 || dim~=ndims(X) ) 
   persistent warned
   if (isempty(warned) ) 
      warning('X has trials in other than the last dimension, permuting to make it so..');
      warned=true;
   end
  X=permute(X,[setdiff(1:ndims(X),[labdim dim]) labdim dim]); 
end;
X=reshape(X,[nf L N]);
incInd=any(Y~=0,1);  exInd=find(~incInd);

wb=opts.wb; 
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) ) 
  wb=zeros(size(X,1)+1,1); 
end
if ( numel(wb)==size(X,1) ) wb=[wb(:);0]; end;
w=wb(1:end-1); b=wb(end);

iH=opts.stepSize; 
X2   = X.*X; % pre-compute for the secant est of the gradient at 0
if( isempty(iH) ) % stepSize = Lipschiz constant = approx inv hessian
  if ( strcmp(opts.lipzApprox,'hess') )
	 % diag-hessian approx for Lipsitch constants?
	 ddL= [sum(X2,2);size(X,2)];
	 l=1;
	 ss = 1./sqrt(ddL(1:end-1));
	 v  = ss.*X(:,1); l=sqrt(v'*v); for i=1:3; v=ss.*(X*((v.*ss)'*X)')/l;l=sqrt(v'*v); end; v=v/l;
	 niH= 1./(.25/2); % approx norm of the inverse hessian
	 iH = 1./(l*ddL)/2; % N.B. Don't forget the factor of 2!
  elseif ( strcmp(opts.lipzApprox,'maxeig') )
	 % est of largest eigenvalue of the data covariance matrix
	 v  =X(:,:,1);l=sqrt(v(:)'*v(:)); 
	 for i=1:3; 
		v=(reshape(X,[],size(X,3))*(v(:)'*reshape(X,[],size(X,3)))')/l;l=sqrt(v(:)'*v(:)); 
	 end; 
	 v=v/l;
	 niH= 1./(.25/2); % approx norm of the inverse hessian
	 iH = 1/(l)/2; % N.B. Don't forget the factor of 2!
  else
     error('Unrecognised Lipz constant estimation procedure');
  end;
end;

nFeat=opts.nFeat; if ( C<0 ) nFeat=-C; C=0; end
onFeat=nFeat;

% build index expression so can quickly get the predictions on the true labels
wghtYnEx= ones(size(Y,2),1); wghtYnEx(exInd)=[];
Yind = Y>0;
Yidx = int32(find(Y>0))'; 

% pre-comp some solution info
wX = tprod(X,[-1 1 2],wb(1:end-1),-1); % [L x N]
f  = wX + wb(end);
dv = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
p  = exp(dv);
p  = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
g  = p(Yidx); % [1xN] = Pr(y_true)
Yerr= Yind-p;  % [LxN] = dp_y, i.e. gradient of the probability of the true class
Yerr(:,exInd)=0; % ensure ignored points aren't included

L  = sum(-log(g));
dL = [-X(:,:)*Yerr(:); -sum(Yerr(:))];
if ( isequal(opts.symDim(:),[1;2]) )
  [U,s]  =eig(reshape(wb(1:end-1),szX(1),[])); V=U';  s=diag(s); 
else
  [U,s,V]=svd(reshape(wb(1:end-1),szX(1),[]),'econ'); s=diag(s);
end
actFeat=abs(s)>opts.eps;if( ~any(actFeat) ) actFeat(1)=true; end;
R  = sum(abs(s));

% Estimate the initial regularisation strength if needed
oactFeat=actFeat;
if ( ~isempty(nFeat) ) 
  wght= g(:).*(1-g(:));
  %ddL = X2*wght; % diagonal entries of the hessian
  dL0 = dL(1:end-1);%-wb(1:end-1).*ddL(:);
  %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
  %     norm of the weight change in the same direction then the component will grow.
  %     Thus we only need to see if the loss-gradient is bigger then 1
  % We can treat the Lsigma as an l1l2 where each component is a group which is independently
  % regularised by transforming to the component space and re-scaling with the inverse reg
  %      C*dR > dL where as dR(w)=1 when w=0 we have C>dL
  dLRg= sqrt(sum((reshape(dL0,szX(1),[])'*U).^2));
  [sdLRg]=sort(dLRg,'descend');         
  % Use the geometric mean, as C scales exponientially
  C=exp(log(sdLRg(min(end,nFeat+[0 1])))*[opts.ratioC;1-opts.ratioC]);% N.B. set C to be enough to *stop* nFeat+1'th weight from becomming non-zero
  if ( opts.verb>0 ) fprintf('%d) nF=%d C=%g\n',0,nFeat,C); end;
end

% N.B. after the inital reg-strength estimation!
% compute the gradient of the regularisor (with 0 gradient for disabled directions)
% Used mainly for convergence testing
dRwact = repop(sign(s(actFeat))','*',V(:,actFeat))'; % efficient when low rank
dw2    = U(:,actFeat)'*reshape(dL(1:end-1),szX(1),[]) + C*dRwact;
dw2    = sqrt(dw2(:)'*dw2(:));
dw20   = dw2; 

lineSearchStep=opts.lineSearchStep(1);
step=1;
ostep=step;
validSteps=0;
J=inf; 
% Information about the solution prior to the prox step, i.e. the basis for the prox step
w0  = wb; fw0 = f; Lw0 = L; dLw0= dL; LLq=1; dJ0=0;
% Information about the iterated solutions, i.e. after the prox step, but before any acceleration is applied
owX = zeros(size(wX)); owb=zeros(size(wb));  of=zeros(size(f));
oowX= owX;             oowb=owb;             oof=of;
oC=C; ooC=oC;
wbstar=wb; Jstar=J; % best solution so-far
for iter=1:opts.maxIter;
  if ( iter > 500 ) 
     if ( iter==500 ) fprintf('\n>500 iter!!\n'); end;
     opts.verb=opts.verb+1; 
  end; % turn on debug info if not converged
  oJ=J; % prev objective
  oactFeat=actFeat; % prev set active features
   
  % check the prox-step was valid  
  wX = tprod(X,[-1 1 2],wb(1:end-1),-1); % [L x N]
  f  = wX + wb(end);
  dv = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
  p  = exp(dv);
  p  = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
  g  = p(Yidx);  % [1xN] = Pr(y_true)
  Yerr= Yind-p;  % [LxN] = dp_y, i.e. gradient of the probability of the true class
  Yerr(:,exInd)=0; % ensure ignored points aren't included

  L  = sum(-log(g));
  dL = [-X(:,:)*Yerr(:);...
		  -sum(Yerr(:))];
  R   = sum(abs(s));
  J   = C*R+L;
  dwb = wb-w0;
  % N.B. do this hear as this is the *only* point where J is the actual value of the current wb
  % Keep track of best solution so far... as acceleration makes objective decrease non-monotonic
  if ( J<Jstar || (C-oC)>1e-4*(C+oC)/2 ) % if C *increases* then auto-best solution
	 wbstar=wb; Jstar=J; 
	 if( opts.verb>0) fprintf('*'); end;
  end; 

  if ( any(actFeat) ) % est distance to the optima under quadratic assumption
    dRwact = repop(sign(s(actFeat))','*',V(:,actFeat))'; % efficient when low rank
    dw2    = U(:,actFeat)'*reshape(dL(1:end-1),szX(1),[]) + C*dRwact;
    dw2    = sqrt(dw2(:)'*dw2(:));
  else
    dRwact=0; dw2=dw20;
  end

  % progress reporting
  printIter=false;
  if ( opts.verb>0 && ( iter<250 || mod(iter,10)==0 ) )
     printIter=true;
    fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f) =%5.4f\t|dw|=%5.3f,%5.3f\tL=%5.3f\t#act=%d\n',...
            iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,...
            norm(owb-wb),dw2,min(step),sum(actFeat));    
  end

  % convergence testing
  if ( iter<3 && dw2>0 )  if ( dw20==0 ) dw20=dw2; else dw20=min(dw20,dw2); end; end;
  if ( iter<5 && iter>1 && abs(J-oJ)>opts.eps ) dJ0 =(dJ0+abs(J-oJ))/2; end;
  if ( validSteps>0 && iter>opts.minIter && ...
       (abs(J-oJ)<=opts.objTol || abs(J-oJ)<=opts.objTol0*dJ0 || ...
        norm(owb-wb)./max(1,norm(wb))<=opts.tol || dw2<=opts.tol0*dw20 ) ) 
    break; 
  end;

  % Taylor approx loss for this step size: L(w,w0) = L(w0) + dL'*(w-w0) + 1/2*Lip*(w-w0)^2 
  %  iff step<1/L then L(w,w0) <= L(w0)+dL'*(w-w0)+1/2(w-w0).^2/step
  %  thus using 1/2*Lip*(w-w0)^2 <= 1/2*(w-wo)^2/step makes Lq larger 
  %  and the quad requirement *stricter* than necessary
  %Lq = Lw0 + dLw0'*dwb + 1/2*rho*(dwb'*dwb); % if have rho available
  Lq = Lw0 + dLw0'*dwb + 1/2/step*((dwb./(iH*niH))'*dwb);  % if don't have rho
  oLLq=LLq; LLq=L/Lq; % info for estimating the correct step size
  if ( opts.verb>1 && printIter) fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g\n',iter,step,L,Lq,L/Lq); end;
  if ( Lq < L*(1-1e-3) ) % **Backtrack** on the prox step
    if ( opts.verb>0  && printIter ) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: step size invalid!\n',iter,step,L,Lq,L/Lq); 
    end;
    %if ( Lq < L*(1-1e-5) ) % shrink rapidly
    step = max(opts.stepLim(1),step*opts.stepLearnRate(1));     % *rapidly* reduce the step size
    %else % shrink slowly
    %  step = max(opts.stepLim(1),step*.9);
    %  %step = max(opts.stepLim(1),step*(Lq/L).^2); % grow/shrink proportionally to error
    %end
    %if ( ~validSteps ) % only backtrack if didn't previous step
       validSteps=0;
       wb   = w0;% undo the previous step
       f    = fw0;
       niH  = niHw0; % back-track the scaling also
       L    = Lw0;
    %end
  else  % only update the step size if the previous step was valid
     validSteps=validSteps+1;
     if ( Lq < L && opts.verb>0  && printIter) 
      fprintf('%d) step=%g  L=%g \tLq=%g \tL/Lq=%g --- Warning: marginally stable!\n',iter,step,L,Lq,L/Lq); 
    end;
    if( iter>3 && LLq<(1-1e-4) ) % newton search for the correct step size for the gradient descent
		if (ostep==step) % try a very small increment
		  dstep = step*1.01;
		else			
        dstep = abs((step-ostep)/(LLq-oLLq))*(1-LLq)/2; % 1/2 the gap to 1
		end
      ostep = step;
      step  = min([opts.stepLim(2),step*opts.stepLearnRate(2),step+dstep]);
    end

    % acceleration step, only accelerate if the step size was valid, and two valid steps in a row
	 % i.e. no acc if marginal stability
    if ( opts.lineSearchAccel && validSteps>1 && Lq>=L ) 
      if ( opts.verb > 0 ) fprintf('a'); end;
      % track the gradient bits, (x_{k-1}-x_{k-2}), N.B. these are 
      % N.B. save the solutions *after* the prox but before the acceleration step!!
      oof =of;  of =f;
      oowb=owb; owb=wb; 
      if ( validSteps>2 ) 
        accstep = (iter-1)/(iter+2)*lineSearchStep;
        wb = wb  +(wb-oowb)*accstep; % N.B. watch for the gota about the updated of=f... owb=wb etc...
        f  = f   +(f-oof)*accstep;    
		  dv = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
		  p  = exp(dv);
		  p  = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
		  g  = p(Yidx); % [1xN] = Pr(y_true)
        %niH= 1./mean(g.*(1-g)); % approx scale of the hessian
        L   = sum(-log(g));        
      end
    end
    % finish evaluating the new starting point
    if ( opts.verb>0 ) % only reallyed needed for progress reporting / convergence testing
      if ( isequal(opts.symDim(:),[1;2]) ) % decompose this solution for obj computation
        [U,s]  =eig(reshape(wb(1:end-1),szX(1),[])); V=U';  s=diag(s); 
      else
         if ( 0 && sum(actFeat)*4 < min(szX(1:end-1)) ) % use faster svds -- never faster!
            [U,s,V]=svds(reshape(wb(1:end-1),szX(1),[]),sum(actFeat)+2);
         else
            [U,s,V]=svd(reshape(wb(1:end-1),szX(1),[]),'econ'); s=diag(s); 
         end
      end    
		%actFeat=abs(s)>opts.eps;
      %R  = sum(abs(s));
    end
	 Yerr= Yind-p;  % [LxN] = dp_y, i.e. gradient of the probability of the true class
	 Yerr(:,exInd)=0; % ensure ignored points aren't included
	 dL = [-X(:,:)*Yerr(:); -sum(Yerr(:))];
    % save info on the starting point
    niHw0=niH;
    w0  = wb;
    fw0 = f;
    Lw0 = L;
    dLw0= dL;
  end    
       
  %-------------------------------------------------------------------------------
  % start/end Prox step
  % Generalised gradient descent step on C*R+L
  % i.e. wb = prox_(step*iH){ w0 - step*iH*dL(w0) }
  dw = step.*iH.*niH.*dL; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( isequal(opts.symDim(:),[1;2]) ) % decompose 
    [U,s]  =eig(reshape(wb(1:end-1),szX(1),[])); V=U';  s=diag(real(s)); % sym (pd?) features
  else % non-sym features
     try
        [U,s,V]=svd(reshape(wb(1:end-1),szX(1),[]),'econ'); s=diag(s);
     catch % convert to double and try again if failed.. (sometimes happens on AMD)
        [U,s,V]=svd(double(reshape(wb(1:end-1),szX(1),[])),'econ'); s=diag(s);
        % ss=reshape(wb(1:end-1),szX(1),[]);
        % save(sprintf('./SVD_failed%d',randi(1000)),'ss');
        % error('SVD failed to converge');
        % %[U,s,V]=svds(reshape(wb(1:end-1),szX(1),[]),min(szX(1:2))); s=diag(s);
     end
  end
  % do the proximal step
  s  = sign(s).*max(0,(abs(s)-C*step*iH*niH)); % prox step on s
  W  = U*diag(s)*V'; 
  wb(1:end-1)=W(:);% re-construct solution
  % N.B. wb=x_k
  actFeat=abs(s)>opts.eps; % update here so is correct when deciding how to update C
  
  %-----------------------------------------------------
  % C search for correct number of features  
  if ( ~isempty(nFeat) && ...
		 validSteps>3 && sum(oactFeat)==sum(actFeat) && dw2<dw20 ) % if soln stable.... otherwise C est is rubbish
    % est current number of active features
    %wght=g(:).*(1-g(:));
    %ddL= X2*wght; % add a ridge to the hessian estimate
    dL0= dL(1:end-1);%-wb(1:end-1).*ddL(:); % estimated gradient if this feature had value=0
    %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
    %     norm of the weight change in the same direction then the component will grow.
    %     Thus we only need to see if the loss-gradient is bigger then 1
    %N.B. only if all elm in group have same weight in structMx
	 dLRg= sqrt(sum((reshape(dL0,szX(1),[])'*U).^2));
	 [sdLRg]=sort(dLRg,'descend');         
	 % N.B. set C to be enough to *stop* nFeat+1'th weight from becomming non-zero
	 % Use the geometric mean, as C scales exponientially
	 estC=exp(log(sdLRg(min(end,nFeat+[0 1])))*[opts.ratioC;1-opts.ratioC]);
    if ( opts.verb>0  && printIter) fprintf('%d) nF=%d C=%g estC=%g\n',iter,nFeat,C,estC); end;
	 % Dampen the update rate to prevent ossilations, only if 10% different
	 if ( abs(estC-C)>.05*C ) % more than 10% change
		validSteps=0; % reset as invalid to stop acceleration
		% Adaptive step size to converge to correct C fairly fast		  
		if ( (oC-C)*(C-estC)>=0 ) % same direction
		  opts.stepC=min(1,opts.stepC*1.05); if ( opts.verb>0  && printIter) fprintf('/\\'); end;
		else % different directions = overshoot = backtrack
		  opts.stepC=max(.1,opts.stepC*.67); if ( opts.verb>0  && printIter) fprintf('\\/'); end;
		end
      oC=C; 
		C =max(0,C*(1-opts.stepC)+opts.stepC*estC);
      if ( opts.verb>0  && printIter) 
		  fprintf('%d) nF=%d step=%g oC=%g estC=%g C=%g\n',iter,nFeat,opts.stepC,oC,estC,C); 
      end;
	 end
  end   
end

wb = wbstar; % return best solution found
wX = tprod(X,[-1 1 2],wb(1:end-1),-1); % [L x N]
dv = wX + wb(end);
dv = repop(dv,'-',max(dv,[],1)); % re-scale for numerical stability
p  = exp(dv);
p  = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
g  = p(Yidx); % [1xN] = Pr(y_true)
L  = sum(-log(g));
if ( isequal(opts.symDim(:),[1;2]) ) % decompose this solution for obj computation
  [U,s]  =eig(reshape(wb(1:end-1),szX(1),[])); V=U';  s=diag(s); 
else
  [U,s,V]=svd(reshape(wb(1:end-1),szX(1),[]),'econ'); s=diag(s);
end    
actFeat=abs(s)>opts.eps;
R  = sum(abs(s));
J  = C*R + L;
  
% progress reporting
if ( opts.verb>=-1 )
	sAct=''; if(~isempty(nFeat)) sAct=sprintf('Cact=%d\t',nFeat); end;
   fprintf('%3d)\ts=[%s]\t%5.3f + %5.3f C(%5.3f)=%5.4f\t|dw|=%5.3f,%5.3f\tL=%5.3f\t%s#act=%d(~%d)\n',...
           iter,sprintf('%5.3f ',s(1:min(end,3))),L,R,C,J,norm(owb-wb),dw2,min(step),...
           sAct,sum(actFeat),sum(abs(s)>1e-3*max(abs(s))));
end
return

%-------------------------------------------------------------------------------
function testCase()
%simple 2d 4 class problem
cents=[-1 0;1 0;0 1;0 -1];
[X,Yl]=mkMultiClassTst(cents,[400 400 400 400],[.2 .2]);[dim,N]=size(X);
clf;labScatPlot(X,Yl,'linewidth',1)
Y =lab2ind(Yl)';

% pre-transform X in a class specific way
S =randn(size(X,1),2,size(Y,1));
S =reshape(cents',size(X,1),1,size(Y,1));
XS=tprod(X,[-1 3],S,[-1 1 2]); % [d x L x N]

XSmu=tprod(XS,[1 2 -3],cat(3,single(Y>0),single(Y==0)),[2 -3 3]);clf;imagesc('cdata',shiftdim(XSmu));

[wb,f,Jlr]=mmlr_cg(XS,Y,0,'verb',1);
[wb,f,Jlr]=mmlsigmalr_prox3(XS,Y,0,'verb',1);
[wb,f,Jlr]=mmlsigmalr_prox3(XS,Y,-1,'verb',1);

% try a stim-classification problem
