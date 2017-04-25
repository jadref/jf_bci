function [wb,f,J,obj,Js]=lsigmalr_cg(X,Y,C,varargin);
% trace norm Regularised linear Logistic Regression Classifier
%
% [wb,f,J,obj]=lsigmalr_cg(X,Y,C,varargin)
% trace-norm Regularised Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J = C*\sum eig(w) + sum_i log( (1 + exp( - y_i ( w'*X_i + b ) ) )^-1 ) 
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
%  C       - trace norm regularisation constant (0)
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
%
% Options:
%  wb      - [(N+1)x1] initial guess at the weights parameters, [W;b]     ([])
%  maxEval - [int] max number for function evaluations                    (N*5)
%  maxIter - [int] max number of CG steps to do                           (inf)
%  maxLineSrch - [int] max number of line search iterations to perform    (50)
%  objTol0 - [float] relative objective gradient tolerance                (1e-5)
%  objTol  - [float] absolute objective gradient tolerance                (0)
%  tol0    - [float] relative gradient tolerance, w.r.t. initial value    (0)
%  lstol0  - [float] line-search relative gradient tolerance, w.r.t. initial value   (1e-4)
%  tol     - [float] absolute gradient tolerance                          (0)
%  verb    - [int] verbosity                                              (0)
%  step    - initial step size guess                                      (1)
%  wght    - point weights [Nx1] vector of label accuracy probabilities   ([])
%            [2x1] for per class weightings
%            [1x1] relative weight of the positive class
%  eta     - [float] relative smoothing constant for quadratic approx to l1 loss   (1e-4)
%                   thus, anything <eta*max(nrm) has been effectively set to 0.
%               N.B. eta<0 = absolute threshold.  eta>0 = relative to max nrm
%  CGmethod -- [str] type of congagite gradient direction compution to use, one of:
%                'PR' 'FR' 'HS' 'MPRP'
%  PCmethod -- method used to compute the pre-conditioner for the CG, one of:      ('wb0+Rmin')
%              none,wb0,wb -- loss based pre-conditioning based on, no/wb(:)=0/current solution
%             regularizor pre-conditioner based on:
%               d1 - 1st dimension diagonal approx, d2 - 2nd dimension diagonal approx, 
%               d12 - both dims, dmin- smallest of 1,2nd dim quad-approx
%               R1 - 1st dimension matrix approx, R2 - 2nd dimension matrix approx,
%               Rmin - smallest of 1,2nd dim matrix approx
%  adaptPC  -- [int] iterations between updates of the pre-conditioner                      (1)
%

% TODO: [] Include effect of the loss in the R1,R2,Rmin pre-conditioner
%       [] Do we need to include/set eta at all?
%

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied
if ( nargin < 3 ) R(1)=0; end;
if( numel(varargin)==1 && isstruct(varargin{1}) ) % shortcut eval option procesing
  opts=varargin{1};
else
  opts=struct('wb',[],'alphab',[],'dim',[],'Jconst',0,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-4,'objTol',0,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,'quadApprox',1,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'bPC',[],'wPC',[],'adaptPC',1,'PCcondtol',2,'actConsPC',0,'PCmethod','wb0+Rmin',...
              'decompInterval',1,...
				  'incThresh',.75,'optBias',1,'maxTr',inf,...
              'getOpts',0,'eta',1e-4,'zeroStart',0,'eigDecomp',0,...
				  'CGmethod','PR','method',[],'initSoln','proto',...
              'nFeat',[]);
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( isempty(opts.CGmethod) && ~isempty(opts.method) ) opts.CGmethod=opts.method; end;
szX=size(X); nd=numel(szX); N=szX(end); nf=prod(szX(1:end-1));
Y=Y(:); % ensure Y is col vector

% Ensure all inputs have a consistent precision
if(isa(X,'double') && isa(Y,'single') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence
opts.tol(end+1:2)=opts.tol(1);
if ( isempty(opts.maxEval) ) opts.maxEval=5*max(prod(szX(1:end-1)),szX(end)); end

if ( opts.eigDecomp && szX(1)~=szX(2) ) error('Cant do eig with non-symetric feature inputs'); end;

if ( C<0 ) opts.nFeat=-C; C=1; end % neg regularisation constant => number of features to have

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);

% check for degenerate inputs
if ( all(Y>=0) || all(Y<=0) )
  warning('Degnerate inputs, class problem');
end

% N.B. this form of loss weighting has no true probabilistic interpertation!
wght=opts.wght;
if ( ~isempty(opts.wght) ) % point weighting
   if ( numel(wght)==1 ) % weight ratio between classes
     wght=zeros(size(Y));
     wght(Y<0)=1./sum(Y<0)*opts.wght; wght(Y>0)=(1./sum(Y>0))*opts.wght;
     wght=wght*sum(abs(Y))./sum(abs(wght)); % ensure total weighting is unchanged
   elseif ( numel(opts.wght)==2 ) % per class weights
     wght=zeros(size(Y));
     wght(Y<0)=opts.wght(1); wght(Y>0)=opts.wght(2);
   elseif ( numel(opts.wght)==N )
   else
     error('Weight must be 2 or N elements long');
   end
end
Yi=Y;
if(max(size(Yi))==numel(Yi)) % convert to indicator
  Yi=cat(2,Yi(:)>0,Yi(:)<0); 
  if( isa(X,'single') ) Yi=single(Yi); else Yi=double(Yi); end; % ensure is right data type
end 
if ( ~isempty(wght) )        Yi=repop(Yi,'*',wght); end % apply example weighting

% check if it's more efficient to sub-set the data, because of lots of ignored points
oX=X; oY=Y;
incInd=any(Yi~=0,2);
if ( sum(incInd)./size(Yi,1) < opts.incThresh ) % if enough ignored to be worth it
   if ( sum(incInd)==0 ) error('Empty training set!'); end;
   X=X(:,incInd); Yi=Yi(incInd,:);
end
% pre-compute stuff needed for gradient computation
Y1=Yi(:,1); sY=sum(Yi,2);

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  W=zeros(szX(1:2));b=0;
  if ( ~opts.zeroStart ) 
    if ( isequal(opts.initSoln,'lr') )
      wb=lr_cg(X,Yi,10*C,'objTol0',1e-3);
      W = reshape(wb(1:end-1),szX(1:2)); b=wb(end);
    else
		alpha=Yi(:,1)./sum(Yi(:,1))/2-Yi(:,2)./sum(Yi(:,2))/2;
      W = reshape(X*alpha,szX(1:2)); b=0;
		% initial scaling and bias estimate
		wX    = W(:)'*X;
		% center and scale to +/-1
		b     = -mean(wX); sd=max(1,sqrt(wX*wX'/numel(wX)-b*b));
		W     = W/sd;    b=b/sd;  wX=wX/sd;
    end
    % % find least squares optimal scaling and bias
    % sb = pinv([wRw+wX*wX' sum(wX); sum(wX) sum(Y~=0)])*[wX*Y; sum(wghtY)];
    % W=W*sb(1); s=s*sb(1); b=sb(2)/2; 
  end
else
  W=reshape(wb(1:end-1),szX(1:2)); b=wb(end);
end 

if ( opts.eigDecomp )
  [U,s]  =eig(W); V=U;  s=diag(s); 
else
  [U,s,V]=svd(W,'econ'); s=diag(s); 
end
nrms=abs(s);
% Round to nearest power of 10 to slow down the rate at which eta changes....
eta=abs(opts.eta); if(opts.eta>0) eta=10.^(ceil(log(max(nrms))/log(10)))*abs(opts.eta); end
if(eta==0) eta=max(abs(opts.eta),eps); end; % guard against eta=0
si     =nrms>=eta; % huber-approx near the origin
% update the loss to refect the huberization around the origin in both methods
nrms(~si)= (nrms(~si).^2/eta+eta)/2;%update-loss: s'=s if |s|>h, ((s/h).^2+1)*h/2 =1/2*(s^2/h+h)
nrmseta=max(eta,nrms); 
if ( opts.quadApprox ) 	
  % alternative way to compute this:
  % si = s~=0; dR = repop(U(:,si),'*',sign(s(si))')*V(:,si)'; % efficient when low rank
  %            dRw= repop(U(:,si).'*',abs(s(si))')*U(:,si)';
  % quad var approx:  1/2* (x^2/|x|+x) for x>eta, 1/2(x^2/eta+eta) otherwise
  nrmseta=max(eta,nrms); 
  if (szX(1)<=szX(2)) % N.B. check this is right way round!!!
	 varR=U*diag(1./nrmseta)*U'/2; % update the variational l1 approx
	 dR  =2*varR*W;
  else
	 varR=V*diag(1./nrmseta)*V'/2; % trailing dim
	 dR  =2*W*varR;
  end
  
else % compute the current true gradient
  ds   = sign(s); ds(~si)=s(~si)/eta; %update-grad: ds = sign(s) if |s|>h, s/h otherwise
  dR   = U*diag(ds)*V';
end

wX   = W(:)'*X;
dv   = wX+b;
p    = 1./(1+exp(-dv(:))); % =Pr(x|y+)
Yerr = Y1-p.*sY; % [N x nCls]
dLw  = -X*Yerr;
Xmu  = sum(X,2)./size(X,2);
X2   = X.*X;

nFeat = sum(abs(s)>max(abs(s)).*abs(opts.eta)); newC=false;
if ( ~isempty(opts.nFeat) ) 
  %H=X'*diag(wght)*X -> diag(H)=wght.*sum(X.^2,2)~= sum(X.^2,2)
  %g    = 1./(1+exp(-Y'.*dv)); 
  g    = (p.*Yi(:,1) + (1-p).*Yi(:,2)); % expected prob we are correct
  wght=g(:).*(1-g(:));
  %dLw = -X*Yerr;
  %ddLw= X2*max(wght,.01); % add a ridge to the hessian estimate
  dLw0= dLw;%-W(:).*ddLw(:);
  %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
  %     norm of the weight change in the same direction then the component will grow.
  %     Thus we only need to see if the loss-gradient is bigger then 1
  % We can treat the Lsigma as an l1l2 where each component is a group which is independently
  % regularised by transforming to the component space and re-scaling with the inverse reg
  % N.B. istructMx effectively changes variables for each group independently such that
  %      in the transformed space the regularisor is a normal sqrt(w.^2) = |w|_2
  %      Then we can use the normal reasoning to find the optimal C in this case, i.e.
  %      C*dR > dLw where as dR(w)=1 when w=0 we have C>dLw
  dLwRg= sqrt(sum((reshape(dLw0(:),size(W))'*U).^2));
  [sdLwRg]=sort(dLwRg,'descend');         
  C=sdLwRg(opts.nFeat);
  fprintf('%d) nF=%d C=%g\n',0,nFeat,C); 
end

% set the pre-conditioner
% N.B. the Hessian for this problem is:
%  H  =[X*diag(wght)*X'+ddR  (X*wght');...
%       (X*wght')'           sum(wght)];
% where wght=g.*(1-g) where 0<g<1
% So: diag(H) = [sum(X.*(wght.*X),2) + 2*diag(ddR);sum(wght)];
% Now, the max value of wght=.25 = .5*(1-.5) and min is 0 = 1*(1-1)
% So approx assuming average wght(:)=.25/2; (i.e. about 1/2 points are on the margin)
%     diag(H) = [sum(X.*X,2)*.25/2+2*diag(R);N*.25/2];
wPC  = opts.wPC; bPC=opts.bPC;
minD='1'; if ( szX(2)<=szX(1) ) minD='2'; end;
if ( isstr(opts.PCmethod) ) 
  pidx=strfind(opts.PCmethod,'+');
  if ( ~isempty(pidx) ) opts.PCmethod={opts.PCmethod(1:pidx(1)-1) opts.PCmethod(pidx(1)+1:end)}; end;
  if ( numel(opts.PCmethod)==1 ) opts.PCmethod{2}=''; end;
  if ( strcmp(opts.PCmethod{2},'dmin') ) opts.PCmethod{2}=['d' minD]; end;
  if ( strcmp(opts.PCmethod{2},'Rmin') ) opts.PCmethod{2}=['R' minD]; end;
end
if ( isempty(wPC) ) 
  %H=X'*diag(wght)*X -> diag(H)=wght.*sum(X.^2,2)~= sum(X.^2,2)
  %g    = 1./(1+exp(-Y'.*dv)); 
  g    = (p.*Yi(:,1) + (1-p).*Yi(:,2)); % expected prob we are correct
  wght = g(:).*(1-g(:));

  % include the effect of the Loss
  switch opts.PCmethod{1};
	 case 'wb0';                        wPCL = sum(X2,2)*.25/2;
	 case {'wb','adapt'};wght=g.*(1-g); wPCL = X2*wght(:);
	 otherwise;                         wPCL = ones(size(X,1),1);
  end

  % include the effect of the regularisor
  switch opts.PCmethod{min(2,numel(opts.PCmethod))};
	 case 'd1';   
		diagddRdw1=(U.*U)*(1./nrmseta)/2; 
		wPCR = C*reshape(repmat(diagddRdw1,[1,szX(2)]),[],1);
	 case 'd2';
		diagddRdw2=(V.*V)*(1./nrmseta)/2;
		wPCR = C*reshape(repmat(diagddRdw2',[szX(1),1]),[],1);
	 case 'd12';
		diagddRdw1=(U.*U)*(1./nrmseta)/2; 
		diagddRdw2=(V.*V)*(1./nrmseta)/2;
		wPCR = C/2*(reshape(repmat(diagddRdw1,[1,szX(2)]),[],1)+...
						reshape(repmat(diagddRdw2',[szX(1),1]),[],1));
	 case 'R1';
		if ( any(strcmp(opts.PCmethod{1},{'wb0'})) ) % include effect of the loss
		  covX1= tprod(reshape(X,[szX(1:2) size(X,2)]),[1 -2 -3],[],[2 -2 -3])./szX(2);
		  dLds = sum((U'*covX1).*U',2)*.25/2; % diag squared norm in the basis for the regularizor
		  wPCR = U*diag(nrmseta./(1+nrmseta./2./C.*dLds))*U'/2/C;
		else % don't include the loss terms
		  wPCR = U*diag(nrmseta)*U'/2/C;
		end
	 case 'R2';			
		if ( any(strcmp(opts.PCmethod{1},{'wb0','wb'})) ) % include effect of the loss
		  covX2= tprod(reshape(X,[szX(1:2) size(X,2)]),[-1 1 -3],[],[-1 2 -3])./szX(1);
		  dLds = sum((V'*covX2).*V',2)*.25/2; % diag squared norm in the basis for the regularizor
		  wPCR = V*diag(nrmseta./(1+nrmseta./2./C.*dLds))*V'/2/C;
		else % don't include the loss terms
		  wPCR = V*diag(nrmseta)*V'/2/C;
		end
	 otherwise; wPCR = 2*C;
  end
  if ( size(wPCR,2)==1 ) % diag PC -> invert the diag hessian estimate to get the pre-conditioner
	 wPC = wPCL + wPCR; % combine the two components
	 wPC(wPC<eps) = 1;   
	 wPC=1./wPC;
  else
	 wPC = wPCR;
  end
end
if ( isempty(bPC) ) % Q: Why this pre-conditioner for the bias term?
  switch lower(opts.PCmethod{1})
	 case 'none';          bPC=1;
	 case 'wb0';           bPC=1./(size(X,ndims(X))*.25/2);
	 case {'wb','adapt'};  bPC=1./sum(wght);
  end
end
if ( isempty(wPC) ) wPC=ones(size(X,1),1); end;
if ( isempty(bPC) ) bPC=1; end;
wPCr=max(eta,nrms); 
 
%dLw   = -X*Yerr;
dJw   = C*dR(:)+dLw;
dJb   = -sum(Yerr);
% precond'd gradient:
%  [H  0  ]^-1 [ dR-X'((1-g).Y))] 
%  [0  bPC]    [   -1'((1-g).Y))] 
% pre-conditioned gradient
if ( size(wPC,1)==numel(dJw) ) MdJw = reshape(wPC.*dJw,szX(1:2));% diag PC
elseif ( size(wPC,1)==szX(1))  MdJw = wPC*reshape(dJw,szX(1:2)); % pre-mult PC
elseif ( size(wPC,1)==szX(2))  MdJw = reshape(dJw,szX(1:2))*wPC; % post-mult PC		
elseif ( iscell(wPC) ) 			MdJw = wPC{1}*reshape(dJw,szX(1:2))*wPC{2}; % pre and post..
end
MdJb = bPC*dJb;
Mrw  =-MdJw;        Mrb  =-MdJb;
dw   = Mrw;         db   = Mrb;
dtdJ =-(dw(:)'*dJw(:))   -db*dJb;
r2   = dtdJ;

% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
Ed   = -(log(max(p,eps))'*Yi(:,1)+log(max(1-p,eps))'*Yi(:,2)); 
Ew   = sum(nrms);
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
%if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] |x|=%2d J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,s(1),s(2),nFeat,J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
Uact=[]; adaptPC=[];
nStuck=0;
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
W0=W; b0=b;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

  oJ= J; oMrw  = Mrw; oMrb=Mrb; or2=r2; oW=W; ob=b; % record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 2 )
      fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ew,Ed); 
      if ( opts.verb>3 ) 
         hold off;plot(0,dtdJ,'r*');hold on;text(0,double(dtdJ),num2str(0)); 
         grid on;
      end
   end;
   ostep=inf;tstep=tstep/2;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   % pre-compute for speed later
	W0  = W;
   wX0 = wX;
   %dw  = reshape(d(1:end-1),size(W)); db=d(end);
   %if ( opts.eigDecomp ) dw=.5*(dw+dw'); end;
   dX  = dw(:)'*X;
	if ( opts.quadApprox ) 
     % N.B. R = trace(D'*varR*W) = \sum_i,j D.*(varR*W) = D(:)'*reshape(varR*W,[nf,1]) 
     %      R = trace(D'*varR*W) = \sum_i,j (D'*varR).*W = reshape((D'*varR)',[nf,1])'*W(:) 
     %                           = reshape(varR*D,[nf,1])'*W(:) % for symetric varR
     if (szX(1)<=szX(2)) % N.B. check this is right way round!!!     
		 dvarR = varR*dw; % == varR*dw (because varR=varR' by construction
		 dvarRw= dvarR(:)'*W0(:); 
		 dvarRd= dvarR(:)'*dw(:);
     else 
		 dvarR = dw*varR;
		 dvarRw= dvarR(:)'*W0(:); 
		 dvarRd= dvarR(:)'*dw(:);
     end
	  dtdR=dvarRw;
	else
	  if ( opts.eigDecomp )
		 [U,s]  =eig(W0); V=U;  s=diag(s); 
	  else
		 [U,s,V]=svd(W0,'econ'); s=diag(s);      
	  end
	  si =nrms>=eta;
	  nrms(~si)= (nrms(~si).^2/eta+eta)/2;%update-loss: s' = s if |s|>h, ((s/h).^2+1)*h/2 =1/2*(s^2/h+h)
	  ds   = sign(s); ds(~si)=s(~si)/eta; %update-grad: ds = sign(s) if |s|>h, s/h otherwise
	  % true gradient
	  dtdR = sum((U'*dw).*V',2)'*ds; % Tr(W'*U*diag(s)*V') = Tr(V'*W'*U*diag(s)) = 
	end
   % initial values
   dtdJ  = -(2*C*dtdR - dX*Yerr - db*sum(Yerr));
   if ( opts.verb > 2 )
	  Ed   = -(log(max(p,eps))'*Yi(:,1)+log(max(1-p,eps))'*Yi(:,2)); 
	  if ( opts.quadApprox ) 			
       Ew   = 2*(reshape(varR*W0,[numel(W),1])'*W0(:)+tstep*2*dvarRw+tstep.^2*dvarRd); % P(w,b|R);
       %Wp   = W0+tstep*dw;Ew2  = 2*reshape(varR*Wp,[numel(W),1])'*Wp(:); % P(w,b|R);
	  else
		 Ew = sum(nrms);
	  end
     J    = Ed + C*Ew + opts.Jconst;              % J=neg log posterior
     fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ew,Ed); 
   end
   odtdJ=dtdJ;      % one step before is same as current
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      wX    = wX0+tstep*dX;%w'*X;
		dv    = wX+b+tstep*db;
		p     = 1./(1+exp(-dv(:))); % =Pr(x|y+)
      %g     = 1./(1+exp(-Y'.*(wX+(b+tstep*db))));
		Yerr  = Y1-p.*sY;
		if ( opts.quadApprox ) 
		  dtdR = (dvarRw+tstep*dvarRd);
		else % compute the current true gradient
		  W = W0 + tstep*dw; 
		  if ( opts.eigDecomp )
			 [U,s]  =eig(W); V=U;  s=diag(s); 
		  else
			 [U,s,V]=svd(W,'econ'); s=diag(s);      
		  end
		  nrms=abs(s);
		  % huberize near the orgin
		  % N.B. fix eta during the line search...
		  % 	  if ( opts.eta>0 ) eta=max(nrms)*abs(opts.eta); else; eta=abs(opts.eta); end
		  si =nrms>=eta;
		  nrms(~si)= ((nrms(~si).^2)/eta+eta)/2; % s' = s if |s|>h, ((s/h).^2+1)*h/2 = (s^2/h+h)/2
		  ds   =sign(s); ds(~si)=s(~si)/eta; % ds = sign(s) if |s|>h, s/h otherwise

		  % true gradient
		  dtdR = sum((U'*dw).*V',2)'*ds; % Tr(W'*U*diag(s)*V') = Tr(V'*W'*U*diag(s)) = 
		end
      dtdJ  = -(2*C*dtdR - dX*Yerr - db*sum(Yerr));
      %fprintf('.%d step=%g ddR=%g ddgdw=%g ddgdb=%g  sum=%g\n',j,tstep,2*(dRw+tstep*dRd),-dX*Yerr,-db*sum(Yerr),-dtdJ);
      
      if ( opts.verb > 2 )
		  % expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
		  Ed   = -(log(max(p,eps))'*Yi(:,1)+log(max(1-p,eps))'*Yi(:,2)); 
			if ( opts.quadApprox ) 
			  %N.B. this computation *does not* take acount the huberization around the origin...
			  %     hence it's very wrong if eta is large!
           Ew   = 2*(reshape(varR*W0,[numel(W),1])'*W(:)+tstep*2*dvarRw+tstep.^2*dvarRd); % P(w,b|R);
           %Wp   = W+tstep*dw;
           %Ew2  = 2*reshape(varR*Wp,[numel(W),1])'*Wp(:); % P(w,b|R);
			else
			  Ew = sum(nrms);
			end
         J    = Ed + C*Ew + opts.Jconst;              % J=neg log posterior
         fprintf('.%d %g=%g @ %g (%g+%g)\n',j,tstep,dtdJ,J,Ew,Ed); 
         %fprintf('.%d %g / %g\n',j,Ew,Ew2);
         if ( opts.verb > 3 ) 
            plot(tstep,dtdJ,'*'); text(double(tstep),double(dtdJ),num2str(j));
         end
      end;

      % convergence test, and numerical res test
      if(iter>1||j>2) % Ensure we do decent line search for 1st step size!
         if ( abs(dtdJ) < opts.lstol0*abs(dtdJ0) || ... % Wolfe 2, gradient enough smaller
              abs(dtdJ*step) <= opts.tol(2) )              % numerical resolution
            break;
         end
      end
      
      % now compute the new step size
      % backeting check, so it always decreases
      if ( oodtdJ*odtdJ < 0 && odtdJ*dtdJ > 0 ...      % oodtdJ still brackets
           && abs(step*dtdJ) > abs(odtdJ-dtdJ)*(abs(ostep+step)) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         %but prev points gradient, this is necessary stop very steep orginal gradient preventing decent step sizes
         odtdJ = -sign(odtdJ)*sqrt(abs(odtdJ))*sqrt(abs(oodtdJ)); % geometric mean
      end
      ostep = step;
      % *RELATIVE* secant step size
      ddtdJ = odtdJ-dtdJ; 
      if ( ddtdJ~=0 ) nstep = dtdJ/ddtdJ; else nstep=1; end; % secant step size, guard div by 0
      nstep = sign(nstep)*max(opts.minStep,min(abs(nstep),opts.maxStep)); % bound growth/min-step size
      step  = step * nstep ;           % absolute step
      tstep = tstep + step;            % total step size      
   end
   if ( opts.verb > 2 ) fprintf('\n'); end;
   
   % Update the parameter values!
   % N.B. this should *only* happen here!
   W  = W0 + tstep*dw; 
   if( opts.eigDecomp ) W=.5*(W+W'); end; % enforce symetry of the matrix
   b  = b + tstep*db;

   % update the information for the variational approx to the regularisor
   os=s; 
   % compute the new basis set and scaling
	% update the scaling using the previous basis
	UW=U'*W;
   s=tprod(UW,[1 -1],V,[-1 1]); nrms=abs(s);
	decomperr = abs(1-sqrt((s(:)'*s(:)) ./ (W(:)'*W(:)))); % compute the validity of the basis
   if ( decomperr>1e-3 || mod(iter,opts.decompInterval)==0 ) % recompute the decomposition every XXX iteration
	  %if ( opts.decompInterval>1 ) opts.decompInterval=opts.decompInterval*1.05; end; % exp-back-off
     if(opts.verb>0 && opts.decompInterval>1)fprintf('\n%d) svd\n',iter);end;
     if ( opts.eigDecomp )
       [U,s]  =eig(W); V=U;  s=diag(s); 
     else
       [U,s,V]=svd(W,'econ'); s=diag(s);      
     end   
   else % re-compute correctly
	  s = sqrt(tprod(UW,[1 -2],[],[1 -2]));
	end
   nrms=abs(s);
	% update the smoothing factor
	if(opts.eta>0) 
	  maxnrms=max(nrms);if(maxnrms==0)maxnrms=1;end; % guard against eta==0
	  eta=(.95)*eta+(1-.95)*(10.^(ceil(log(maxnrms)/log(10))))*abs(opts.eta); 
	end
	si     =nrms>=eta; % huber-approx near the origin
   nrmseta=max(eta,nrms); 
	% update the loss to refect the huberization around the origin in both methods
	nrms(~si)= (nrms(~si).^2/eta+eta)/2;%update-loss: s'=s if |s|>h, ((s/h).^2+1)*h/2 =1/2*(s^2/h+h)
	if ( opts.quadApprox ) 	
	  % quad var approx:  1/2* (x^2/|x|+x) for x>eta, 1/2(x^2/eta+eta) otherwise
     if (szX(1)<=szX(2)) % N.B. check this is right way round!!!
		 varR=U*diag(1./nrmseta)*U'./2; % update the variational l1 approx
		 dR  =2*varR*W;
     else
		 varR=V*diag(1./nrmseta)*V'./2; % leading dim
		 dR  =2*W*varR;
     end

	else % compute the current true gradient
	  ds   = sign(s); ds(~si)=s(~si)/eta; %update-grad: ds = sign(s) if |s|>h, s/h otherwise
	  dR   = U*diag(ds)*V';
	end
	condest = max(abs(nrms))./min(nrms);

   % compute the other bits needed for CG iteration
   odJw= dJw; % keep so can update pre-conditioner later if wanted...
   dLw = -X*Yerr;                dLb = -sum(Yerr);
   dJw = C*dR(:)+dLw;            dJb = dLb;
	% pre-conditioned gradient
	if ( size(wPC,2)==1 )          MdJw = reshape(wPC.*dJw,szX(1:2));% diag PC
	elseif ( size(wPC,1)==szX(1))  MdJw = wPC*reshape(dJw,szX(1:2)); % pre-mult PC
	elseif ( size(wPC,1)==szX(2))  MdJw = reshape(dJw,szX(1:2))*wPC; % post-mult PC
	end
	MdJb = bPC*dJb;

	% apply the constraints......
	if ( opts.actConsPC>0 ) 
	  if ( ~isempty(adaptPC) ) % include the adaptive pre-conditioner
		 % compute the updated condition estimate
		 pcNrm    = diag(U'*adaptPC*U)./nrms;
		 condest  = max(pcNrm(:))/min(pcNrm(:));
	  end
	  % if sufficiently bad the introduce a new pre-conditioner term
	  if ( condest > 10 ) %opts.actConsPC )
		 fprintf('\npc-update: ',iter); 
		 % udpate the pre-conditioner for the current situation
		 adaptPC      = U*diag(nrmseta)*U';
	  end
	  % lastly apply the pre-conditioner
	  if ( ~isempty(adaptPC) ) % include the adaptive pre-conditioner
		 MdJw = reshape(adaptPC*MdJw,[],1);
	  end
	end

	% project the active constraints to prevent matrix condition issues
	if ( 0 && condest > opts.actConsPC )
	  % udpate the pre-conditioner for the current situation
	  adaptPC      = U*diag(nrmseta)*U'./2;

	  % compute the gradient of the singular values by projecting weight gradient on singular-vectors
	  % ds      = sum((U'*reshape(dJ(1:end-1),szX(1:2))).*V',2); % projected gradient
	  % ds = sum((U'*(dLw-C*dR)).*V',2); % % projected gradient from reg is stronger than from loss
	  % active constraint, i.e. moving towards 0, if small value, small gradient, and grad makes smaller
	  actCons = abs(s)<eta;% & (s.*ds)>0;% & abs(ds)<eta;
	  if ( any(actCons) )
		 oUact = Uact; 
		 Uact  = U(:,actCons);
		 if ( isempty(oUact) ) oUact=zeros(size(Uact)); end;
		 sim = Uact'*oUact; sim=sum(sim.^2,2);
		 dact= sum((Uact'*reshape(dw,szX(1:2))).^2,2);
		 fprintf('%d) cond=%5f |act|=%d  <act,oact>=%6f  <d,act>=%6f\n',iter,condest,sum(actCons),sum(sqrt(sim)),sum(sqrt(dact)));
		  % shrink gradient (deflate) in the active directions
		 %CMdJ(1:end-1)= CMdJ(1:end-1) ...
		 %					 - opts.actConsPC*vec(U(:,actCons)*(U(:,actCons)'*reshape(CMdJ(1:end-1),szX(1:2)))); 
	  end
	end

   %------------------------------------------------
   % pre-conditioner update
   %% %condest=nrmseta(:)./wPCr(:); condest=max(condest)./min(condest);%N.B. ignores effects due to rotation of the basis!
   %% if ( iter>1 ) % cond est as non-diag nature of PC*Hess
   %%   %condest=sum(abs(sum(wPC.*(2*C*varR))-1)); 
   %%   %condest=sum(abs(sum(wPC.*(2*C*varR+reshape(ddLw,size(W))))-1)); 
   %% end;
   %% if ( opts.verb>1 ) 
   %%   fprintf('%d) pc*varR=[%g,%g]\n',iter,max(condest),min(condest));
   %% end;
	% TODO: Intelligent method to decide when to update the pre-conditioner?
   if ( opts.adaptPC>=1 && mod(iter,round(opts.adaptPC))==0 )
	  % && (condest > opts.PCcondtol || mod(iter,ceil(nf/2))==0) )   
     if ( opts.verb>0 ) fprintf('\npc* ',iter);  end;
	  opts.adaptPC=opts.adaptPC*1.2;
     if ( opts.verb>=2 ) 
       if ( opts.verb<2 ) fprintf('%d) pc*varR=[%g,%g] ',iter,max(condest),min(condest)); end;
     end;
	  switch ( opts.PCmethod{1} ) 
		 %case {'wb','adapt'};wght=g.*(1-g); wPCL(:) = X2*wght(:);
	  end	  
	  if ( szX(1)<szX(2) ) minD='1'; else minD='2'; end;
	  switch ( opts.PCmethod{min(end,2)} )
		 case 'd1';   
			diagddRdw1=(U.*U)*(1./nrmseta)/2; 
			wPCR = C*reshape(repmat(diagddRdw1,[1,szX(2)]),[],1);
		 case 'd2';
			diagddRdw2=(V.*V)*(1./nrmseta)/2;
			wPCR = C*reshape(repmat(diagddRdw2',[szX(1),1]),[],1);
		 case 'd12';
			diagddRdw1=(U.*U)*(1./nrmseta)/2; 
			diagddRdw2=(V.*V)*(1./nrmseta)/2;
			wPCR = C/2*(reshape(repmat(diagddRdw1,[1,szX(2)]),[],1)+...
							reshape(repmat(diagddRdw2',[szX(1),1]),[],1));		 
		 case 'R1';
			if ( any(strcmp(opts.PCmethod{1},'wb0')) ) % include effect of the loss
			  dLds = sum((U'*covX1).*U',2)*.25/2; % diag squared norm in the basis for the regularizor
			  wPCR = U*diag(nrmseta./(1+nrmseta./2./C.*dLds))*U'/2/C;
			else % don't include the loss terms
			  wPCR = U*diag(nrmseta)*U'/2/C;
			end
		 case 'R2';			
			if ( any(strcmp(opts.PCmethod{1},'wb0')) ) % include effect of the loss
			  dLds = sum((V'*covX2).*V',2)*.25/2; % diag squared norm in the basis for the regularizor
			  wPCR = V*diag(nrmseta./(1+nrmseta./2./C.*dLds))*V'/2/C;
			else % don't include the loss terms
			  wPCR = V*diag(nrmseta)*V'/2/C;
			end
		 case 'R12';
		   wPCR = {U*diag(sqrt(nrmseta))*U'/2/C V*diag(sqrt(nrmseta))*V'/2/C};
	  end
	  if ( size(wPCR,2)==1 ) % diag PC -> invert the diag hessian estimate to get the pre-conditioner
		 wPC = wPCL + wPCR; % combine the two components
		 wPC(wPC<eps) = 1;   
		 wPC=1./wPC;
	  else
		 wPC = wPCR;
	  end
	  % pre-conditioned gradient
	  if ( size(wPC,1)==numel(dJw) ) MdJw = reshape(wPC.*dJw,szX(1:2));% diag PC
	  elseif ( size(wPC,1)==szX(1))  MdJw = wPC*reshape(dJw,szX(1:2)); % pre-mult PC
	  elseif ( size(wPC,1)==szX(2))  MdJw = reshape(dJw,szX(1:2))*wPC; % post-mult PC		
	  elseif ( iscell(wPC) ) 			MdJw = wPC{1}*reshape(dJw,szX(1:2))*wPC{2}; % pre and post..
	  end
	  MdJb = bPC*dJb;
	  
   end   
   
	% compute the solution quality
	r2  = abs(dw(:)'*MdJw(:)      +db*MdJb);
   % compute the function evaluation
	% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
	Ed   = -(log(max(p,eps))'*Yi(:,1)+log(max(1-p,eps))'*Yi(:,2)); 
   Ew   = sum(abs(nrms)); % report the *true*, i.e. not huberized loss % BODGE!
   J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior
	Js(iter) = J;
   onFeat= nFeat; nFeat= sum(abs(s)>max(abs(s)).*abs(opts.eta));
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] |x|=%2d J=%5f (%5f+%5f) |dJ|=%8g cond=%5f' lend],...
              iter,neval,s(1),s(2),nFeat,J,Ew,Ed,madJ,decomperr);
   end   

   if ( ~newC && iter>2 && (J > oJ*(1.001) || isnan(J)) ) % check for stuckness
		nStuck=nStuck+1;
      if ( opts.verb>=1 ) warning(sprintf('%d) Line-search Non-reduction',iter)); end;
		if ( nStuck > 1 ) % two non-decrease in a row means we're stuck
		  J=oJ; W=oW; b=ob; 
        wX   = W(:)'*X;
		  fprintf('-aborted\n'); 
		  break; 
		end;
   else
	  nStuck=max(0,nStuck-1);
	end;
   
   %------------------------------------------------
   % convergence test
   if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(r02,r2);
   elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,r2); % conv if smaller than best single step
   end
   madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
   if ( r2<=opts.tol(1) || ... % small gradient + numerical precision
        r2< r02*opts.tol0(1) || ... % Wolfe condn 2, gradient enough smaller
        neval > opts.maxEval || ... % abs(odtdJ-dtdJ) < eps || ... % numerical resolution
        madJ <= opts.objTol(1) || madJ < opts.objTol0(1)*dJ0 ) % objective function change
      break;
   end;    
               
   %------------------------------------------------
   % conjugate direction selection
   % N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
   %      PR is better when have adaptive pre-conditioner so more robust in non-linear optimisation
	Mrw  = -MdJw;           Mrb  =-MdJb;
	r2   = dJw(:)'*MdJw(:)       +dJb*MdJb;
	beta=0; theta=0; 
   switch (opts.CGmethod) 
	  case 'PR'; beta = max(((Mrw(:)-oMrw(:))'*(-dJw)+(Mrb-oMrb)*-dJb)/or2,0); % Polak-Ribier
	  case 'MPRP'; beta = max(((Mrw(:)-oMrw(:))'*(-dJw)+(Mrb-oMrb)*-dJb)/or2,0); % Polak-Ribier
			 theta = ( dJw(:)'*dw(:) + dJb'*db(:) ) ./ or2;%modification to reduce line-search dependence
	  case 'FR'; beta = max(r2/or2,0); % Fletcher-Reeves % TODO: broken?
	  case 'GD'; beta = 0; % gradient descent
   end
	% conj grad direction
   dw    = Mrw+beta*dw;   db    = Mrb+beta*db;
	if ( theta~=0 ) dw = dw-theta*(Mrw-oMrw); db=db-theta*(Mrb-oMrb); end;
   dtdJ  = -dw(:)'*dJw(:) -db'*dJb;  % new search projected onto the true gradient
   if( dtdJ <= 0 )         % non-descent dir switch to steepest
     if ( opts.verb >= 1 ) fprintf('%d) non-descent dir\n',iter); end;      
     dw=Mrw;             db=Mrb; 
	  dtdJ=-dw(:)'*dJw(:)-db'*dJb; 
   end;      
  
   %-----------------------------------------------------
   % C search for correct number of features
   if ( ~isempty(opts.nFeat) ) 
     % est current number of active features
     newC  = false;
     if ( nFeat~=opts.nFeat && nFeat==onFeat && mod(iter,3)==0 ) % only when stable...
       wght=g(:).*(1-g(:));
       %dLw = -X*Yerr;
       ddLw= X2*max(wght,.01); % add a ridge to the hessian estimate
       dLw0= dLw-W(:).*ddLw(:);
       %dLw0= abs(dLw)+abs(ddLw).*abs(W(:)); % est loss gradient if this weight was set to 0
       %N.B. for a l1l2 regularisor, if the L2 norm of the loss gradient is greater than the 
       %     norm of the weight change in the same direction then the component will grow.
       %     Thus we only need to see if the loss-gradient is bigger then 1
       %N.B. only if all elm in group have same weight in structMx
       dLwRg= sqrt(sum((reshape(dLw0(:),size(W))'*U).^2));
       [sdLwRg]=sort(dLwRg,'descend');         
       estC=sdLwRg(opts.nFeat);
       if ( nFeat>opts.nFeat && estC>C*(1+5e-2))     C=estC; newC=true;
       elseif ( nFeat<opts.nFeat && estC<C*(1-5e-2)) C=estC; newC=true;
       end
       if ( newC ) 
         dJw = C*dR(:) - X*Yerr;    dJb = -sum(Yerr);
         MdJw= PC.*dJw;             MdJb= bPC*dJb;
         Mrw =-MdJw;                Mrb =-MdJb;
         dw  = Mr;                  db  = Mrb;
			dtdJ=-dw'*dJw             -db*dJb;
         fprintf('%d) nF=%d C=%g estC=%g\n',iter,nFeat,C,estC); 
       end;
     end
   end 
 end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   W=W0; b=b0;
end;

% compute the final performance with untransformed input and solutions
dv   = wX+b;
p    = 1./(1+exp(-dv(:))); % =Pr(x|y+)
% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
Ed   = -(log(max(p,eps))'*Yi(:,1)+log(max(1-p,eps))'*Yi(:,2)); 
Ew   = sum(abs(s));
J    = Ed + C*Ew + opts.Jconst;       % J=neg log posterior
nFeat= sum(abs(s)>max(abs(s)).*abs(opts.eta)*10); % slightly stricter
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] |x|=%2d J=%5f (%5f+%5f) |dJ|=%8g' '\n'],...
          iter,neval,s(1),s(2),nFeat,J,Ew,Ed,madJ);
end

% compute final decision values.
if ( all(size(X)==size(oX)) ) f=dv; else f   = W(:)'*oX + b; end;
f = reshape(f,size(oY));
obj = [J Ew Ed];
wb=[W(:);b];
return;

%-----------------------------------------------------------------------
function [opts,varargin]=parseOpts(opts,varargin)
% refined and simplified option parser with structure flatten
i=1;
while i<=numel(varargin);  
   if ( iscell(varargin{i}) ) % flatten cells
      varargin={varargin{1:i} varargin{i}{:} varargin{i+1:end}};
   elseif ( isstruct(varargin{i}) )% flatten structures
      cellver=[fieldnames(varargin{i})'; struct2cell(varargin{i})'];
      varargin={varargin{1:i} cellver{:} varargin{i+1:end} };
   elseif( isfield(opts,varargin{i}) ) % assign fields
      opts.(varargin{i})=varargin{i+1}; i=i+1;
   else
      error('Unrecognised option');
   end
   i=i+1;
end
return;

%-----------------------------------------------------------------------------
function []=testCase()
%Make a Gaussian balls + outliers test case
nch=8; nT=12; N=1600;
nd=nch*nT; padD=zeros(3,nd/2-1);
[X,Y]=mkMultiClassTst([padD [-1 0; 1 0; .2 .5] padD],[N*3/8 N*3/8 N*2/8],.3,[],[-1 1 1]);

% non-sym pos def matrices
wtrue=randn(nch,nT); [utrue,strue,vtrue]=svd(wtrue,'econ'); strue=diag(strue);
% sym-pd matrices
wtrue=randn(nch,nT); wtrue=wtrue*wtrue'; [utrue,strue]=eig(wtrue); strue=diag(strue); vtrue=utrue;
% re-scale components and make a dataset from it
strue=sort(randn(numel(strue),1).^2,'descend');
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
noise=randn([size(wtrue),size(Xtrue,3)])*sum(strue)/10; 
if(size(Xtrue,1)==size(Xtrue,2))noise=tprod(noise,[1 -2 3],[],[2 -2 3]); end;
X2d  =Xtrue + noise;
wb0  =randn(size(X2d,1),size(X2d,2));

%Make a chxtime test case
z=jf_mksfToy(); X2d=z.X; Y=z.Y;
% sym-pd
X2d=tprod(z.X,[1 -2 3],[],[2 -2 3]);

if ( ~exist('X') || isempty(X) )    X    =reshape(X,[],size(X,ndims(X))); end;
if ( ~exist('X2d') || isempty(X2d)) X2d  =reshape(X,[nch size(X,1)/nch size(X,2)]); end;

C=100; % N.B. this should create a rank=1 solution
% direct run with default parameters
tic,[wbl2,f,J]=lr_cg(X2d,Y,C,'verb',1);toc
tic,[wbl2,f,J]=lsigmalr_cg(X2d,Y,C,'verb',1);toc
tic,[wbl2,f,J]=lsigmalr_cg(permute(X2d,[2 1 3]),Y,C,'verb',1);toc


tols=10.^([-3 -4 -5 -7]);
% simple l2, + test convergence rate (approx linear)
for ti=1:numel(tols);
  tic,[wbl2,f,J]=lr_cg(X2d,Y,C,'verb',1,'objTol0',tols(ti));toc
end
% lsigmalr approx to l2
for ti=1:numel(tols);
  tic,[wbl2,f,J]=lsigmalr_cg(X2d,Y,C,'verb',1,'quadApprox',1,'eta',-1,'objTol0',tols(ti));toc
end
% lsigmalr with more l1 like situation
for ti=1:numel(tols);
  tic,[wbl2,f,J]=lsigmalr_cg(X2d,Y,C,'verb',1,'quadApprox',1,'eta',-.001,'objTol0',tols(ti));toc
end
% lsigmalr with even stricter l1 approx - 10x worse condition = 10x slower....
for ti=1:numel(tols);
  tic,[wbl2,f,J]=lsigmalr_cg(X2d,Y,C,'verb',1,'quadApprox',1,'eta',-.0001,'objTol0',tols(ti));toc
end

% scaling with PC
for ti=1:numel(tols);
  tic,[wbl2,f,J]=lsigmalr_cg(X2d,Y,C,'verb',1,'quadApprox',1,'eta',-.0001,'PCmethod','wb+Rmin','objTol0',tols(ti));toc
end

% evaluate different parameter settings effect on convergence performance
maxIter=300; adaptPC=10; eta=1e-4;
%optvals={'none' 'wb' 'wb+d12' 'none+dmin' 'none+Rmin' 'wb0+Rmin' 'wb+Rmin' 'none+R12'};
optname='PCmethod'; optvals={'none' 'wb' 'none+Rmin' 'wb0+Rmin'};
optname='decompInterval'; optvals={1 10 20 100};
optname='adaptPC'; optvals={1 5 10 20};
optname='eta'; optvals={1e-2 1e-3 1e-4 1e-5};
clf;hold on;
for mi=1:numel(optvals);
  optstr=optvals{mi};if(isnumeric(optvals{mi})) optstr=num2str(optvals{mi}); end;
  fprintf('\n----  %s = %s -----',optname,optstr);
  tic,[wb,f,J,ans,Js]=lsigmalr_cg(X2d,Y,C,'verb',1,'quadApprox',1,'eta',eta,'objTol0',1e-7,'maxIter',maxIter,optname,optvals{mi});t=toc;
  perf.J(mi)=J; perf.t(mi)=t;
  plot(Js,linecol(mi),'linewidth',1,'displayName',optstr);legend;drawnow
end
W=reshape(wb(1:end-1),size(X2d,1),size(X2d,2));svd(W)
tic,[wb,f,J]=lsigmalr_prox3(X2d,Y,C,'verb',0,'objTol0',1e-4);toc, 

%low rank -- different algorithms
szX=size(X);
tic,[wb,f,J]=lsigmalr_cg(X2d,Y,C,'verb',0,'eta',-1e-4,'quadApprox',1,'objTol0',1e-4);toc, 
W=reshape(wb(1:end-1),szX(1:2));[J dJ obj]=lsigmalr(X,Y,C,wb,0,0);
cgq=struct('wb',wb,'J',J,'dJ',dJ,'W',W,'R',obj(1),'L',obj(2),'s',obj(3:end));

tic,[wb,f,J]=lsigmalr_cg(X2d,Y,C,'verb',0,'eta',-1e-4,'quadApprox',0,'objTol0',1e-4);toc, 
W=reshape(wb(1:end-1),szX(1:2));[J dJ obj]=lsigmalr(X,Y,C,wb,0,0);
cg=struct('wb',wb,'J',J,'dJ',dJ,'W',W,'R',obj(1),'L',obj(2),'s',obj(3:end));

tic,[wb,f,J]=lsigmalr_prox3(X2d,Y,C,'verb',0,'objTol0',1e-4);toc, 
W=reshape(wb(1:end-1),szX(1:2));[J dJ obj]=lsigmalr(X2d,Y,C,wb,0,0);
prox=struct('wb',wb,'J',J,'dJ',dJ,'W',W,'R',obj(1),'L',obj(2),'s',obj(3:end));

fprintf('cgq : J=%g\t(%g+%g)\t  |s|=%d\n',cgq.J,cgq.R,cgq.L,sum(abs(cgq.s)>1e-6));
fprintf('cg  : J=%g\t(%g+%g)\t  |s|=%d\n',cg.J, cg.R, cg.L, sum(abs(cg.s)>1e-6));
fprintf('prox: J=%g\t(%g+%g)\t  |s|=%d\n',prox.J,prox.R,prox.L,sum(abs(prox.s)>1e-6));
clf;mimage(cgq.W,cg.W,prox.W)



% test with re-seeding solutions
Cscale=.1*sqrt(CscaleEst(X2d,2));
[wb10,f,J] =lsigmalr_cg(X2d,Y,Cscale*2.^4,'verb',1,'wb',[]);  
[wb,f,J]=lsigmalr_cg(X2d,Y,Cscale*2.^3,'verb',1);  
[wb102,f,J]=lsigmalr_cg(X2d,Y,Cscale*2.^3,'verb',1,'wb',wb10);  

% test the symetric version
tic,[wb,f,J] =lsigmalr_cg(X2d,Y,10,'verb',1);toc
tic,[wbs,f,J]=lsigmalr_cg(X2d,Y,10,'verb',1,'eigDecomp',1);toc

% test with automatic reg-strength determination
tic,[wb,f,J] =lsigmalr_cg(X2d,Y,100,'verb',1);toc
tic,[wb,f,J] =lsigmalr_cg(X2d,Y,-10,'verb',1);toc

% compare on real dataset
z=jf_cov(jf_fftfilter(stdpreproc(jf_load('own_experiments/motor_imagery/hand_tapping/continuous/online','S4','im_train','20090804'),'addClassInfo',1,'nFold',10),'bands',[7 8 26 28]));
jf_cvtrain(z,'objFn','lsigmalr_cg','Cs',5.^(-3:3),'Cscale','l1')

X2d=z.X; Y=z.Y; C=500;
lsigmalr_cg(X2d,Y,C);


% test the numerics of the variational approx to the gradient
Ut=orth(randn(10,10)); st=(10.^(rand(10,1)*10)).*sign(randn(10,1));st=st./sum(abs(st)); % high variance in eigenvalues
W=Ut*diag(st)*Ut';

if ( size(W,1)==size(W,2) ) 
  [U,s]=eig(W);s=diag(s); V=U;
else
  [U,s,V]=svd(W,'econ'); s=diag(s);
end
varR=(U*diag(1./abs(s))*U');
trace(W*(U*diag(1./abs(s))*U')*W')
trace(W'*(U*diag(1./abs(s))*U')*W)

% N.B. for symetric W these are all equivalent!
reshape(W'*varR,[numel(W),1])'*W(:)
W(:)'*reshape(varR*W,[numel(W),1])  % true in all cases!
reshape((W'*varR)',[numel(W),1])'*W(:) % true in all cases!
reshape(W*varR,[numel(W),1])'*W(:)
reshape(varR*W',[numel(W),1])'*W(:)

% for non-symetric W only this is true...
D=randn(size(W));
mad(D(:)'*reshape(varR*W,[numel(W),1]),trace(D'*(varR*W)))
mad(reshape((D'*varR)',[numel(W),1])'*W(:),trace(D'*varR*W))


% test with real data
fn={'bpcovLSigmaLRcg' 'cvtrain' 'eegonly' 'filt8-27' 'cov' 'verb' 1 'Cscale' 'l1' 'lsigmalr_cg'};
