function [wb,f,C] = l1lr_prox(X,Y,C,varargin)
% proximal opt to solve the l1 reg least logistic regression
%
% [wb,f,C] = l1lr_prox(X,Y,C,varargin)
% 
%  J = \min_{w,b} |(X*w+b)-y|^2 + C |w|
opts=struct('stepSize',[],'maxIter',5000,'verb',0,'wb',[],'alphab',[],'objTol',1e-8,'tol',1e-5,'objTol0',0,...
            'lineSearchAccel',1,'lineSearchStep',[1 1.05 4 .1 2],'structMx',[],'dim',[]);
opts=parseOpts(opts,varargin);
if ( numel(opts.lineSearchStep)<5 ) opts.lineSearchStep(end+1:5)=1; end;
szX=size(X); X=reshape(X,[],size(X,ndims(X))); % get orginal size, then reshape into [feat x examples]
wb=opts.wb; 
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) ) wb=zeros(size(X,1)+1,1); end
if (numel(wb)==size(X,1) ) wb=[wb(:);0]; end;

wX   = (wb(1:end-1)'*X)';
dv   = wX+wb(end);
g    = 1./(1+exp(-Y.*dv)); % =Pr(x|y), max to stop log 0
Yerr = Y.*(1-g);
dL = [-X*Yerr; -sum(Yerr)];
X2   = X.*X;

structMx=opts.structMx; 
% N.B. string or single entry struct matrix are type of structure matrix to make
if ( ~isempty(structMx) && (isstr(structMx) || numel(structMx)==1) ) 
  structMx=mkStructMx(szX(1:end-1),structMx);
  structMx=reshape(structMx,[],size(structMx,ndims(structMx))); % work with vector X
  if ( sum(structMx(:)==0)>numel(structMx)/2 ) structMx=sparse(structMx); end;
end

rho=opts.stepSize; 
if( isempty(rho) ) % stepSize = Lipschiz constant = approx inv hessian
  % N.B. 
  %  H  =[X*diag(wght)*X'+2*C(1)*R  (X*wght');...
  %       (X*wght')'                sum(wght)];
  % where; wght=g*(1-g) and g=Pr(+|X)
  % Thus, assume wght=.5, then H = [X*X'/4 + 2*C*R sum(X,2)/4;sum(X,2)'/4 size(X,2)/4]
  % also if assume X is centered, then sum(X,2)~=0
  % est of largest eigenvalue of the data covariance matrix, X*X'
  w=X(:,1); l=sqrt(w'*w); for i=1:5; w=X*(w'*X)'/l; l=sqrt(w'*w); end;
  rho = l*.25/2 ;%+ sqrt((X(:)'*X(:)))/numel(X);
end;

tgtFeat=[]; 
if ( C<0 ) 
  tgtFeat=-C;
  wght=g(:).*(1-g(:));
  %dL = -X*Yerr';
  ddL= X2*(wght+.01); % add a ridge to the hessian estimate
  dL0= dL(1:end-1);%-ddL.*wb(1:end-1);
  if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
    %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
    %     norm of the weight change in the same direction then the component will grow.
    %     Thus we only need to see if the loss-gradient is bigger then 1
    % N.B. istructMx effectively changes variables for each group independently such that
    %      in the transformed space the regularisor is a normal sqrt(w.^2) = |w|_2
    %      Then we can use the normal reasoning to find the optimal C in this case, i.e.
    %      C*dR > dL where as dR(w)=1 when w=0 we have C>dL
    istructMx = structMx; istructMx(structMx~=0)=1./istructMx(structMx~=0);
    dLRg= sqrt((double(dL0(:)).^2)'*istructMx);
  else
    dLRg = abs(dL0);
  end
  [sdLRg]=sort(dLRg,'descend');         
  C=sdLRg(tgtFeat);
  fprintf('%d) nF=%d C=%g\n',0,tgtFeat,C);   
end



lineSearchStep=opts.lineSearchStep(1);
J=inf; owb=zeros(size(wb)); oowb=owb; oC=C; zeroFeat=false; nFeat=sum(wb(1:end-1)~=0); onFeat=0;
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

  % eval current solution
  f  = (wb(1:end-1)'*X)'+wb(end);
  g  = 1./(1+exp(-Y.*f));
  Yerr= Y.*(1-g);
  R  = sum(abs(wb(1:end-1)));
  L  = sum(-log(max(g,eps)));
  dL = [-X*Yerr; -sum(Yerr)];
  dR = [sign(wb(1:end-1));0];
  J  = C*R + L;

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
  if ( ~isempty(tgtFeat) ) % est what correct regularisor should be
    newC  = false;
    if ( nFeat~=tgtFeat && abs(nFeat-onFeat)<1 && mod(iter,1)==0 ) % only when stable...
        wght=g(:).*(1-g(:));
        %ddL= X2*(max(wght,.01)); % add a ridge to the hessian estimate
        ddL= X2*wght; % add a ridge to the hessian estimate        
        dL0= dL(1:end-1)-ddL.*wb(1:end-1); % est loss gradient if this weight was set to 0
        if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
          %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
          %     norm of the weight change in the same direction then the component will grow.
          %     Thus we only need to see if the loss-gradient is bigger then 1
          %N.B. only if all elm in group have same weight in structMx
          dLRg= sqrt((double(dL0(:)).^2)'*istructMx);
        else
          dLRg = abs(dL0);%./R;
        end
        [sdLRg]=sort(dLRg,'descend');         
        estC=sdLRg(tgtFeat);
        if ( nFeat>tgtFeat && estC>C*(1+1e-3))     C=estC; newC=true;
        elseif ( nFeat<tgtFeat && estC<C*(1-1e-3)) C=estC; newC=true;
        end
        if ( newC ) 
          fprintf('%d) nF=%d C=%g estC=%g\n',iter,tgtFeat,C,estC); 
        end;
      end
      zeroFeat = sum(sdLRg<C);
  end

  % Generalised gradient descent step on C*R+L
  % Gradient descent on L
  dw = dL/rho; 
  wb = wb - dw; % grad descent step on L
  % proximal step on C*R+L -- i.e. shrinkage
  if ( ~isempty(structMx) )
    nrm = sqrt(double(wb(1:end-1)).^2'*structMx)';
    %wb(1:end-1) = wb(1:end-1).*(structMx*(1./nrm)); % normalize
    snrm= max(0,nrm-double(C/rho)); % shrink
    wb(1:end-1) = double(wb(1:end-1)).*(structMx*(snrm./nrm));% apply shrinkage
  else
    wb(1:end-1)=sign(wb(1:end-1)).*max(0,(abs(wb(1:end-1))-C/rho)); % prox step on R
  end
  % N.B. wb=x_k
  onFeat=nFeat;  nFeat=sum(wb(1:end-1)~=0);

  % progress reporting and convergence testing
  J = C*R+L;
  if ( opts.verb>0 )
    fprintf('%3d)\twb=[%s]\t%5.3f + C * %5.3f = %5.4f \t|dw|=%5.3f \tL=%5.3f/%5.3f\tC=%5.3f\t#act=%d p#act=%d\n',iter,sprintf('%5.3f ',wb(1:min(end,3))),L,R,J,norm(owb-wb),min(rho),lineSearchStep,C,nFeat,sum(~zeroFeat(:)));    
  end
  if ( iter>1 && abs(oC-C)./max(1,abs(C))<5e-3 && ...
       (abs(J-oJ)<opts.objTol || norm(owb-wb)./max(1,norm(wb))<opts.tol ) ) 
    break; 
  end;
  
end
return

function testCase()
wtrue=randn(100,1); % true weight
Y    =sign(randn(1000,1)); % true pred
X    =randn(size(wtrue,1)*2,size(Y,1)); X(1:size(wtrue,1),:)=X(1:size(wtrue,1),:)+wtrue*Y';% 1/2 relevant, 1/2 not
wb0  =randn(size(X,1),1);

wb = l1lr_prox(X,Y,1);
wb = l1lr_prox(X,Y,-10); % with target num features
% with a seed solution to test the C determination
wb = l1lr_prox(X,Y,-10,'wb',wb0); % with target num features

% with a warm start to test the C determination
wb10 = l1lr_prox(X,Y,-10); 
wb12 = l1lr_prox(X,Y,-12,'wb',wb10); 
wb   = l1lr_prox(X,Y,-12); 

% with a structure matrix
structMx=mkStructMx([4 size(X,1)/4],2); structMx=reshape(structMx,[size(X,1) size(structMx,3)]);
tic,[wb,f,J]=l1lr_prox(X,Y,1,'verb',1,'structMx',structMx);toc  
