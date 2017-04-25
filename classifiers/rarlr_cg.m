function [wb,f,J,obj,tstep]=rarlr_cg(X,Y,R,varargin);
% Regularised linear Logistic Regression Classifier
%
% [wb,f,J,obj]=lr_cg(X,Y,C,varargin)
% Regularised Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% Assuming: Pr(x,y+) = exp(x*w+b) and Pr(x,y-) = exp(-(x*w+b))
%           Pr(x|y) = exp(x*w+b)/(exp(x*w+b)+exp(-(x*w+b)))
%
% J = w' R w + w' mu + sum_i sum_y Pr(y_i=y) log( Pr(x_i|y) )
%
% if Pr(y_i=y) is 0/1 variable
%   = w' R w + w' mu + sum_i log(Pr(x_i|y_i))
%   = w' R w + w' mu + sum_i log(exp(y_i*(x*w+b))/(exp(x*w+b)+exp(-(x*w+b))))
%   = w' R w + w' mu + sum_i log (1 + exp( - y_i ( w'*X_i + b ) ) )^-1 ) 
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
%            OR
%            [Nx2] matrix of weighting that this is the true class
%  R       - quadratic regularisation matrix                                   (0)
%     [1x1]       -- simple regularisation constant             R(w)=w'*R*w
%     [d1xd2x...xd1xd2x...] -- full matrix                      R(w)=w'*R*w
%     [d1xd2x...] -- simple weighting of each component matrix, R(w)=w'*diag(R)*w
%     [d1 x d1]   -- 2D input, each col has same full regMx     R(w)=trace(W'*R*W)=sum_c W(:,c)'*R*W(:,c)
%     [d2 x d2]   -- 2D input, each row has same full regMX     R(w)=trace(W*R*W')=sum_r W(r,:) *R*W(r,:)'
%     N.B. if R is scalar then it corrospends to roughly max allowed length of the weight vector
%          good default is: .1*var(data)
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
%
% Options:
%  dim     - [int] dimension of X which contains the trials               (ndims(X))
%  rdim    - [1x1] dimensions along which the regularisor applies.        ([])
%  mu      - [d1xd2x...] vector containing linear term                    ([])
%  wb      - [(N+1)x1] initial guess at the weights parameters, [W;b]     ([])
%  maxEval - [int] max number for function evaluations                    (N*5)
%  maxIter - [int] max number of CG steps to do                           (inf)
%  maxLineSrch - [int] max number of line search iterations to perform    (50)
%  objTol0 - [float] relative objective gradient tolerance                (1e-4)
%  objTol  - [float] absolute objective gradient tolerance                (1e-5)
%  tol0    - [float] relative gradient tolerance, w.r.t. initial value    (0)
%  lstol0  - [float] line-search relative gradient tolerance, w.r.t. initial value   (1e-2)
%  tol     - [float] absolute gradient tolerance                          (0)
%  verb    - [int] verbosity                                              (0)
%  step    - initial step size guess                                      (1)
%  wght    - point weights [Nx1] vector of label accuracy probabilities   ([])
%            [2x1] for per class weightings (neg class,pos class)
%            [1x1] relative weight of the positive class
%  CGmethod - [str] type of conjugate gradients solver to use:
%             one-of: PR, HS, GD=gradient-descent, FR, MPRP-modified-PR
%  PCmethod - [str] type of pre-conditioner to use.                       ('adaptDiag')
%             one-of: none, adaptDiag=adaptive-diagonal-PC
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

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
  opts=struct('wb',[],'alphab',[],'dim',[],'rdim',[],'mu',0,'Jconst',0,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-3,'objTol',1e-4,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,...
				  'PCmethod','adaptDiag','PCalpha',exp(-log(2)/14),'PClambda',.25,'PCminiter',[10 20],...
				  'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0,'ar',[],'arb',[]);
  opts=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
% Ensure all inputs have a consistent precision
if(islogical(Y))Y=single(Y); end;
if(isa(X,'double') && ~isa(Y,'double') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence

if ( ~isempty(opts.dim) && opts.dim<ndims(X) ) % permute X to make dim the last dimension
   persistent warned
   if (isempty(warned) ) 
      warning('X has trials in other than the last dimension, permuting to make it so..');
      warned=true;
   end
  X=permute(X,[1:opts.dim-1 opts.dim+1:ndims(X) opts.dim]);
  if ( ~isempty(opts.rdim) && opts.rdim>opts.dim ) opts.rdim=opts.rdim-1; end; % shift other dim info
end
szX=size(X); nd=numel(szX); N=szX(end); nf=prod(szX(1:end-1));
if(size(Y,1)==1) Y=Y'; end; % ensure Y is col vector
Y(isnan(Y))=0; % convert NaN's to 0 so are ignored

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);
mu=opts.mu; if ( numel(mu)>1 ) mu=reshape(mu,[nf 1]); else mu=0; end;

% check for degenerate inputs
if ( (size(Y,2)==1 && (all(Y(:)>=0) || all(Y(:)<=0))) ||...
	  (size(Y,2)==2 && any(all(Y==0,1))) )
  warning('Degnerate inputs, 1 class problem');
end

% N.B. this form of loss weighting has no true probabilistic interpertation!
wght=opts.wght;
if ( ~isempty(opts.wght) ) % point weighting -- only needed in wghtY
   if ( numel(wght)==1 ) % weight ratio between classes
     wght=zeros(size(Y));
     wght(Y<0)=1./sum(Y<0)*opts.wght; wght(Y>0)=(1./sum(Y>0))*opts.wght;
     wght = wght*sum(abs(Y))./sum(abs(wght)); % ensure total weighting is unchanged
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
ar=opts.ar;   arb=opts.arb;
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  w=zeros(nf,1);b=0;
  a=[]; if ( ~isempty(arb) ) a=zeros(size(arb,2),1); ar=zeros(size(arb,1),1); end;
  % prototype classifier seed
  alpha=Yi(:,1)./sum(Yi(:,1))/2-Yi(:,2)./sum(Yi(:,2))/2;
  % prototype classifier for each time-shift
  for taui=1:(size(ar,1)+1); % N.B. take acount of the fixe ar=1 for tau=0
	 W(:,taui) = X*[zeros(taui-1,1);alpha(1:end-(taui-1))];
  end										  % decompose into a rank-1 solution
  [U,S,V]=svd(W); S=diag(S); [ans,si]=sort(abs(S));
  w=S(1).*U(:,1).*V(1,1);
  ar=V(:,1)./V(1,1); ar=ar(2:end); % ensure 1st ar element is fixed at unity...
  if ( ~isempty(a) ) a=(ar'*arb)'; end; %project ar onto the ar-basis
  %clf;mimage(W,w*[1;arb*a]','diff',1); % plot the approx vs. full-rank soln
  
  Rw    = R(1).*w;
  wRw   = w'*Rw(:);
  wX    = w'*oX; wX=wX(incInd); % only included points in seed
  % re-scale to sensible range, i.e. 0-mean, unit-std-dev
  if ( ~isempty(ar) ) wX    = filter([1;ar],1,wX,[],1); end % apply the AR
  b     = -mean(wX); sd=max(1,sqrt(wX*wX'/numel(wX)-b*b));
  w     = w/sd; b=b/sd;
  wX=wX/sd; wRw = wRw/sd;
  % find least squares optimal scaling and bias
  % N.B. this can cause numerical problems some time....
  %sb = pinv([wRw+wX*wX' sum(wX); sum(wX) sum(incInd)])*[wX*oY(incInd); sum(wghtY)];
  %w=w*sb(1); b=sb(2);
else
  w=wb(1:nf);        % weight vector
  b=wb(nf+1);        % bias
  a=wb(nf+2:end);    % ar-coefficients
  if ( ~isempty(a) )
	 if ( isempty(arb) ) ar=a; arb=eye(numel(a)); else ar=arb*a; end;
  end;
end 
ar=[1;ar(:)];      % include the constant 1st term

									  % extract the bits of the regularizor if needed
R_w=R; R_a=R;
if ( numel(R)>1 )
  if ( numel(R)==2 )
	 R_w = R(1); R_a=R(2);
  elseif ( numel(R)==numel(w)+numel(a) )
	 R_w = R(1:numel(w));  R_a=R(numel(w)+1:end);
  else
	 error('dont know how to handle this regularizor');
  end
end

Rw   = R_w.*w;
wX   = w'*X;
f0   = (wX+b)';
if ( isempty(ar) )
  f  = f0;                % [ N x L ]
else
  f  = filter(ar,1,f0,[],1); % convolve with the ar component
end
p    = 1./(1+exp(-f(:))); % =Pr(x|y+) = p1 % [N x L]
% dL_i = gradient of the loss = (1-p_y) = 1-prob of the true class = { py if y=y, 1-py otherwise i.e. 
dL = Y1-p.*sY; % dL = Yerr % [N x L]
if ( ~isempty(ar) )% convolve the loss gradient with the ar-model
  dL0 = dL;
  dL  = filter(ar(end:-1:1),1,[dL;zeros(numel(ar)-1,size(dL,2))],[],1); % pad to include end points
  dL  = dL(numel(ar):end,:); % remove padding
end
dLdw = -(X*dL);
dLdb = -sum(dL,1);
if ( ~isempty(a) )
  Ra  = R_a.*a;
  dLdar=zeros(numel(ar)-1,1);
  for taui=1:numel(dLdar); dLdar(taui,1) = -(dL0(taui+1:end,:)'*f0(1:end-taui)); end;
  if ( isempty(arb) ) dLda=dLdar; else dLda=arb'*dLdar; end;
else
  dLda = zeros(size(a));
  Ra   = zeros(size(a));
end
dJ  = [2*Rw + mu + dLdw; ...
                   dLdb;...
		 2*Ra      + dLda];
% precond'd gradient:
%  [H  0  ]^-1 [ Rw+mu-X'((1-g).Y))] 
%  [0  bPC]    [      -1'((1-g).Y))] 
PC   = 1;
MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d'*dJ);
r2   = dtdJ;

% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
Ed   = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); 
Ew   = w'*Rw(:);     % -ln P(w,b|R);
if( ~isempty(a) )    Ew =Ew+a'*Ra(:); end;
if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
J    = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,w(1),w(2),J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
% summary information for the adaptive diag-hessian estimation
% seed with prior for identity PC => zero-mean, unit-variance
N=0; Hstats = zeros(size(dJ,1),5); %[ sw sdw sw2 sdw2 swdw]
w0=w; b0=b;
nStuck=0;iter=1;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

  restart=false;
  oJ= J; oMr  = Mr; or2=r2; ow=w; ob=b; oa=a;% record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 2 )
      fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ed,Ew); 
      if ( opts.verb>3 ) 
         hold off;plot(0,dtdJ,'r*');hold on;text(0,double(dtdJ),num2str(0)); 
         grid on;
      end
   end;
   ostep=inf;otstep=tstep;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   odtdJ=dtdJ; % one step before is same as current
   % pre-compute for speed later
	w0  = w;       b0=b;       a0=a;
   dw  = d(1:nf); db=d(nf+1); da=d(nf+2:end);
   wX0 = wX;
   dwX = dw'*X;
   % TODO: FAST: pre-convolve the dv with the ar component, N.B. for all be the bias term
   % but then need to be careful about tracking convolved vs. non-convolved versions of all
	% the parameters
	% if ( ~isempty(ar) ) 
	%   wX = filter(ar,1,wX,[],1);
	%   dwX = filter(ar,1,dwX,[],1);	  
	% end
   if( ~isequal(mu,0) ) dmu = dw'*mu; else dmu=0; end;
	% pre-compute regularized weight contribution to the line-search
   Rw =R_w.*w;      dwRw =dw'*Rw;  % scalar or full matrix
   Rdw=R_w.*dw;     dwRdw=dw'*Rdw;	
	% pre-compute regularized ar-components contribution to the gradient
	Ra =R_a.*a;      daRa =da'*Ra;
	Rda=R_a.*da;     daRda=da'*Rda;
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,J,Ew,Ed); 
   end
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values      
		wX = wX0+tstep*dwX;
		b  = b0 +tstep*db;
		if ( ~isempty(a) ) % udpate the AR
         a  = a0 +tstep*da;
			if ( isempty(arb) ) ar(2:end)=a; else ar(2:end)=arb*a; end;
      end; 
		f0 = (wX + b)'; % [NxL]
		f  = f0;
		% SLOW: do the convolution in the line search loop
		if ( ~isempty(ar) ) f=filter(ar,1,f0,[],1); end; 
		p  = 1./(1+exp(-f)); % =Pr(x|y+) % [NxL]
		dL = Y1-p.*sY; % [NxL]
		dLda=zeros(size(a));
		if ( ~isempty(ar) )% convolve the loss gradient with the ar-model
		  dL0 = dL;
		  dL  = filter(ar(end:-1:1),1,[dL;zeros(numel(ar)-1,size(dL,2))],[],1); % pad to inc end points
		  dL  = dL(numel(ar):end,:); % remove padding
		  if ( ~isempty(a) )
			 dLdar=zeros(size(ar)-1);
			 for taui=1:size(dLdar,1); dLdar(taui,1) = -dL0(taui+1:end,:)'*f0(1:end-taui); end;
			 if ( isempty(arb) ) dLda=dLdar; else dLda=arb'*dLdar; end;			 
		  end
		end
		if ( isempty(a) )
        dtdJ = -(2*(dwRw+tstep*dwRdw) + dmu - dwX*dL - db*sum(dL));
		else
        dtdJ = -(2*(dwRw+tstep*dwRdw) + dmu - dwX*dL - db*sum(dL) + 2*(daRa+tstep*daRda) + da'*dLda);
		end


      if ( 0 || opts.verb>2 ) % debug code to validate if the incremental gradient computation is valid
		  sw  = (w0+tstep*dw);
		  sb  = (b0+tstep*db);
        sRw = R_w.*sw;%Rw+ tstep*Rd;
        swX  = sw'*X;
		  sf  = (swX + sb)'; % [NxL]
		  sf0 = sf;
		  if ( ~isempty(a) )
			 sa  = a0+tstep*da;
			 if ( isempty(arb) ) sar= [1;sa]; else sar= [1;arb*sa]; end;% update the ar-coeff
		  end;
		  if ( ~isempty(ar) ) sf=filter(sar,1,sf,[],1); end; 
		  sp  = 1./(1+exp(-sf)); % =Pr(x|y+) % [NxL]
		  sdL = Y1-sp.*sY; % [NxL]
		  sdLda = [];
		  if ( ~isempty(ar) )% convolve the loss gradient with the ar-model
			 sdL0 = sdL;
			 sdL  = filter(ar(end:-1:1),1,[sdL;zeros(numel(ar)-1,size(sdL,2))],[],1);
			 sdL  = sdL(numel(ar):end,:); % remove padding
			 if ( ~isempty(a) )
				sRa = R_a.*sa;
				sdLda=zeros(numel(ar)-1,1);
				for taui=1:size(sdLda,1);
				  sdLdar(taui,1) = -sdL0(taui+1:end,:)'*sf0(1:end-taui);
				end;
				if ( isempty(arb) ) sdLda=sdLdar; else sdLda=arb'*sdLdar; end;
			 end;
		  end
		  sdLdw = -(X*sdL);
		  sdLdb = -sum(sdL);
		  sdJ=[ 2*sRw + mu + sdLdw; ... % w
				  sdLdb;...               % b
				  2*sRa + sdLda];         % a
        sdtdJ= -d'*sdJ;  % gradient along the line @ new position
		  %mad(sdtdJ,dtdJ)
		  sp(sp==0)=eps; sp(sp==1)=1-eps; % guard for log of 0
		  Ed   = -(log(sp)'*Yi(:,1)+log(1-sp)'*Yi(:,2)); % P(D|w,b,fp)
		  Ew   = sw(:)'*sRw(:) + sa(:)'*sRa(:);
        J    = Ed + Ew + opts.Jconst;              % J=neg log posterior
		  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',j,tstep,dtdJ,sdtdJ,J,Ew,Ed); 
      end
		
      % convergence test, and numerical res test
      if(iter>1||j>2) % Ensure we do decent line search for 1st step size!
         if ( abs(dtdJ) < opts.lstol0*abs(dtdJ0) || ... % Wolfe 2, gradient enough smaller
              abs(dtdJ*step) <= opts.tol )              % numerical resolution
            break;
         end
      end
      
      % now compute the new step size
      % backeting check, so it always decreases
      if ( oodtdJ*odtdJ < 0 && odtdJ*dtdJ > 0 ...      % oodtdJ still brackets
           && abs(step*dtdJ) > abs(odtdJ-dtdJ)*(abs(ostep+step)) ) % would jump outside 
         step = ostep + step; % make as if we jumped here directly.
         odtdJ = -sign(odtdJ)*sqrt(abs(odtdJ))*sqrt(abs(oodtdJ)); % geometric mean
      end
      ostep = step;
      % *RELATIVE* secant step size
      ddtdJ = odtdJ-dtdJ; 
      if ( ddtdJ~=0 ) nstep = dtdJ/ddtdJ; else nstep=1; end; % secant step size, guard div by 0
      nstep = sign(nstep)*max(opts.minStep,min(abs(nstep),opts.maxStep)); % bound growth/min-step size
      step  = step * nstep;            % absolute step
      tstep = tstep + step;            % total step size      
   end
   if ( opts.verb > 2 ) fprintf('\n'); end;
   
   % Update the parameter values!
   % N.B. this should *only* happen here!
   w  = w0 + tstep*dw; 
   b  = b0 + tstep*db;
   Rw = Rw + tstep*Rdw;
   % compute the other bits needed for CG iteration
	dLdw = -(X*dL); % N.B. negate *after* multiply to avoid making a copy of X
	dLdb = -sum(dL,1);
	if ( ~isempty(a) ) % update the AR coefficients
	  a  = a0 + tstep*da;
	  Ra = Ra + tstep*Rda;
	  if ( isempty(arb) ) ar(2:end) = a; else ar(2:end) = arb*a; end;% update the ar-coeff
	  dLdar=zeros(numel(ar)-1,1);
	  for taui=1:numel(ar)-1; dLdar(taui,1) = -(dL0(taui+1:end,:)'*f0(1:end-taui)); end;
	  if ( isempty(arb) ) dLda=dLdar; else dLda=arb'*dLdar; end;
	end
	odJ = dJ;
	dJ  = [2*Rw + mu + dLdw; ...
			           + dLdb; ...
		    2*Ra      + dLda];

										 										  % update the pre-conditioner
	if ( strcmp(opts.PCmethod,'adaptDiag') && iter>opts.PCminiter(1) )
	  % update the diag-hessian estimation
	  wb     = [w;b;a];
										  % update the moment information
	  % half-life = half-life = log(.5)/log(alpha) => alpha = exp(log(.5)/half-life)
	  alpha  = opts.PCalpha;
	  % TODO: This is very sensitive to outliers.... hence it doens't forget the startup noise very rapidly....
		 N      =      N*alpha + (1-alpha)*1; % N.B. forall alpha, lim_t->inf \sum N*alpha+(1-alpha)*1 = 1
		 Hstats = Hstats*alpha + (1-alpha)*[wb dJ wb.^2 dJ.^2 wb.*dJ];
										  % update the diag-hessian estimate
		 wdJvar = (Hstats(:,5) - Hstats(:,1).*Hstats(:,2)./N)./N;
		 wvar   = (Hstats(:,3) - Hstats(:,1).*Hstats(:,1)./N)./N;
		 dJvar  = (Hstats(:,4) - Hstats(:,2).*Hstats(:,2)./N)./N;
		 expvar =  wdJvar ./ sqrt(abs(wvar)) ./ sqrt(abs(dJvar)); % explained variance = goodness-of-fit
		 goodIdx= expvar>.5; % these are the estimates we trust (so far)
									  % unreg estimate
		 H      = wdJvar ./ wvar;
		 muH    = H;
		 medH   = median(muH(goodIdx));
										  % reg-estimate, towards a spherical hessian
		 H      =  (wdJvar+opts.PClambda.*medH.*wvar) ./ (wvar*(1+opts.PClambda)); 
										  % test if we should update PC
	  Hrec.wb(:,iter)=wb; Hrec.dJ(:,iter)=dJ;
	  
     % hessian is trustworthy, i.e. sufficient fraction of the half-life achieved
	  % DEBUG: test comparsion between est and true diag-hessian info
	  % [d,dy,dh,ddy,ddh]=checkgrad(@(w) arlrFn(w,X,Y,R,[],arb),wb,1e-5,1,0);
% npt=60;h=[];for di=1:numel(wb); tmp=[Hrec.wb(di,end-npt+1:end);ones(1,npt)]'\Hrec.dJ(di,end-npt+1:end)'; h(di,1)=tmp(1);end;
% clf;semilogy([abs(H./ddh) abs(muH./ddh)],'linewidth',2); [ans,si]=sort(abs(log(H./ddh)),'descend');
% clf;di=si(1);plot(Hrec.wb(di,:)',Hrec.dJ(di,:)','-*','linewidth',1);title(sprintf('H=%g Htrue=%g',H(di),ddh(di)));
% clf; plot([ddh H muH],'linewidth',2);legend('Htrue_fd','H_RLS_reg','H_RLS');%,sprintf('H_ls%d',npt));
		 
	  condest=H.*PC./medH; % deviation in spherisity between the old/new estimates
	  if ( iter>opts.PCminiter(2) &&... % Hest is trustworthy?
			 (max(abs(condest(goodIdx)))>10 || min(abs(condest(goodIdx))) < .1) ) % order mag wrong
		 if ( opts.verb>=1 ) fprintf('%d) PC update\n',iter); end;
%clf;plot([H expvar goodIdx condest],'linewidth',2);legend({'H','expvar','goodIdx','condest'});set(gca,'ylim',[-5 5]);
		 restart = true;
		 if ( numel(PC)==1 ) PC=PC*ones(size(dJ)); end;
		 goodIdx = goodIdx & (abs(condest)>5 | abs(condest)<.2); % only update the big changers
         %PC = median(H)./abs(H); % N.B. PC is inverse Hessian est.. with guard for divide by 0
		 PC(goodIdx,1) = medH./(H(goodIdx)); % N.B. PC is inverse Hessian est.. with guard for divide by 0
	  end
	end

   MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   dJ2= dJ'*dJ;
   
   % compute the function evaluation
	p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
	Ed   = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); % P(D|w,b,fp)
   Ew   = w'*Rw(:);% P(w,b|R);
	if ( ~isempty(a) ) Ew = Ew + a'*Ra(:); end;
   J    = Ed + Ew + opts.Jconst;       % J=neg log posterior
   if( ~isequal(mu,0) ) Emu=w'*mu; J=J+Emu; end;
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,w(1),w(2),J,Ew,Ed,dJ2);
   end   
   
   %------------------------------------------------
   % convergence test
   if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(r02,dJ2);
   elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,dJ2); % conv if smaller than best single step
   end
   madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
   if ( dJ2<=opts.tol || ... % small gradient + numerical precision
        dJ2< r02*opts.tol0 || ... % Wolfe condn 2, gradient enough smaller
        neval > opts.maxEval || ... % abs(odtdJ-dtdJ) < eps || ... % numerical resolution
        madJ <= opts.objTol || madJ < opts.objTol0*dJ0 ) % objective function change
      break;
   end;    

	% TODO : fix the backup correctly when the line-search fails!!
   if ( J > oJ*(1.001) || isnan(J) ) % check for stuckness
     if ( opts.verb>=1 )
		 warning(sprintf('%d) Line-search Non-reduction - aborted\n',iter));
	  end;
     J=oJ; w=ow; b=ob; a=oa; Mr=oMr; r2=or2; tstep=otstep*.01;
      wX   = w'*X;
		nStuck =nStuck+1;
		restart=true;
		if ( nStuck > 1 ) break; end;
	end;

	
   %------------------------------------------------
   % conjugate direction selection
   % N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
   %     PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
	beta=0;theta=0;
	switch opts.CGmethod;
	  case 'PR';		 
		 beta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
	  case 'MPRP';
		 beta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
		 theta = Mr'*dJ / or2; % modification which makes independent of the quality of the line-search
	  case 'FR';
		 beta = max(r2/or2,0); % Fletcher-Reeves
	  case 'GD'; beta=0; % Gradient Descent
	  case 'HS'; beta=Mr(:)'*(Mr(:)-oMr(:))/(-d(:)'*(Mr(:)-oMr(:)));  % use Hestenes-Stiefel update
	  otherwise; error('unrecog CG method -- using GD')
	end
   d     = Mr+beta*d;     % conj grad direction
	if ( theta~=0 ) d = d - theta*(Mr-oMr); end; % include the modification factor
   dtdJ  = -d'*dJ;         % new search dir grad.
   if( dtdJ <= 0 || restart || ...  % non-descent dir switch to steepest
       (opts.restartInterval>0 && mod(iter,opts.restartInterval)==0))         
     if ( dtdJ<=0 && opts.verb >= 2 ) fprintf('non-descent dir\n'); end;
     d=Mr; dtdJ=-d'*dJ;
   end; 
   
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

% compute the final performance with untransformed input and solutions
Rw  = R_w.*w;                  % scalar or full matrix
Ra  = R_a.*a;
f   = (wX + b)'; % [N x L]
% Convolve with the ar coefficients
if ( ~isempty(ar) ) f=filter(ar,1,f,[],1); end; 
p   = 1./(1+exp(-f));     % [L x N] =Pr(x|y_+) = exp(w_ix+b)./sum_y(exp(w_yx+b));
p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
Ed  = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); % expected loss
Ew  = w'*Rw(:);     % -ln P(w,b|R);
if ( ~isempty(a) ) Ew = Ew + a'*Ra(:); end;
J   = Ed + Ew + opts.Jconst;       % J=neg log posterior
if( ~isequal(mu,0) ) Emu=w'*mu; J=J+Emu; end;
if ( opts.verb >= 0 ) 
   fprintf(['\n%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,w(1),w(2),J,Ew,Ed,dJ2);
end

% compute final decision values.
if ( ~all(size(X)==size(oX)) )
  f   = (w'*oX + b)';
  % Convolve with the ar coefficients
  if ( ~isempty(ar) ) f=filter(ar,1,f,[],1); end;   
end;
f = reshape(f,[size(oY,1) 1]);
obj = [J Ew Ed];
wb=[w(:);b;a(:)];
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


%-----------------------------------------------------------------------
function [f,f0]=fwdPass(X,W,b,ar)
% compute the forward pass to get the activations for each example
f0   = (W'*X + b)';
if ( isempty(ar) )
  f  = f0;                % [ N x L ]
else
  f  = filter(ar,1,f0,[],1); % convolve with the ar component
end
return;

%-----------------------------------------------------------------------
function [dJ]=bwdPass(X,dLdf,Rw,W,b,ar,Ra,a,arb);
% compute the backward pass to get the error gradients w.r.t. the loss
%
% Inputs:
%   X    - [nf x N]
%   dLdf - [N x L]
%   dRdw - [nf x L]
%   W    - [nf x L]
%   b    - [1 x L]
%   ar   - [ntau x 1]
%   a    - [ntau-1 x 1]
%   arb  - [ntau-1 x NTAU]
dLdftau=dLdf; % include the effect of the ar and time-reverse to make contribution per t
if ( ~isempty(ar) )% convolve the loss gradient with the ar-model
  %pad to include end points
  dLdftau= filter(ar(end:-1:1),1,[dLdftau;zeros(numel(ar)-1,size(dLdftau,2))],[],1);
  dLdftau= dLdftau(numel(ar):end,:); % remove padding
end
dLdw = -(X*dLdftau);
dLdb = -sum(dLdftau,1);
Rw   = R_w.*w;
if ( ~isempty(a) ) % do we learn a as well?
  Ra  = R_a.*a;
  dLdar=zeros(numel(ar)-1,1);
  for taui=1:numel(dLdar); dLdar(taui,1) = -(dLdf0(taui+1:end,:)'*f0(1:end-taui)); end;
  if ( isempty(arb) ) dLda=dLdar; else dLda=arb'*dLdar; end;
else
  dLda = zeros(size(a));
  Ra   = zeros(size(a));
end
dJ  = [2*Rw + dLdw; ...
              dLdb;...
		 2*Ra + dLda];
return;


%-----------------------------------------------------------------------------
function []=testCase()
%Make a Gaussian balls + outliers test case
nd=100; nClass=800;
[X,Y]=mkMultiClassTst([zeros(1,nd/2-1) -1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) 1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) .2 .5 zeros(1,nd/2-1)],[nClass nClass 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
wb0=randn(size(X,1)+1,1);

tic,lr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0);toc
tic,[wb,f,J]=rarlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0);toc
tic,[wb,f,J]=rarlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'ar',[1]);toc
tic,[wb,f,J]=rarlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'ar',[1 -1]);toc


% add ar to the optimization
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(1,1)]);toc
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(10,1)]);toc
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(100,1)]);toc
										  % stiff system
tic,[wb,f,J]=rarlr_cg(X,Y,[1e5;zeros(numel(wb0)-1+100-1,1)],'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(100,1)]);toc


										  % try the adaptive pre-conditioner
tic,[wb,f,J]=rarlr_cg(X,Y,[1e5;zeros(numel(wb0)-1+100-1,1)],'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(100,1)],'PCmethod','adaptDiag','PCalpha',exp(-log(2)/14));toc


										  % plot the result in bits;
clf;plot(wb(1:size(X,1)),'r','displayname','w','linewidth',1);hold on;plot(wb(size(X,1)+2:end),'g','displayname','a','linewidth',1);legend('show');

										% with a baises matrix for the ar coefficients
arb=ones(10,1); arb=arb./sqrt(sum(arb.^2,1));
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',[wb0;zeros(1,1)],'arb',arb);toc

nb=5;
arb=zeros(2^nb-1,nb);t=0;for i=1:size(arb,2); l=2^(i-1); arb(t+(1:l),i)=1; t=t+l; end;
arb=repop(arb,'/',sqrt(sum(arb.^2))); %normalize reg
wb0=rarlr_cg(X,Y,1,'maxIter',0,'arb',arb); % get seed solution
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',wb0,'arb',arb);toc %=483) 1993 J=45
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'wb',[wb0(1:end-nb);zeros(nb,1)],'arb',arb);toc %=609) J=45.7
tic,[wb,f,J]=rarlr_cg(X,Y,1,'verb',1,'objTol0',1e-10,'arb',arb);toc % with auto-seed %483) J=45
clf;plot(arb*wb(size(X,1)+2:end),'k','linewidth',2);

% test the solution
[J,dJ]=arlrFn(wb,X,Y,0);


										  % toy non-stationary tests
Y=sign(randn(1,1000));
Xt=cat(1,Y,-Y);
Xs=Xt + randn(size(Xt))*1e-3;
wb0=randn(size(Xs,1)+1,1);
% add non-stationarity
tauNS=100; % period of the non-stationarity
X=Xs;
X(1,:) = Xs(1,:)+ sin((0:size(Xs,2)-1)./tauNS*2*pi)*5; %(0:size(X,2)-1)./tauNS; % 
X(2,:) = Xs(2,:)- cos((0:size(Xs,2)-1)./tauNS*2*pi)*5; % (0:size(X,2)-1)./tauNS; % 
clf;labScatPlot(X,Y)

cvtrainLinearClassifier(Xs,Y,[],[],'objFn','rarlr_cg','arb',ones(10,1))
cvtrainLinearClassifier(X,Y)
%(ave)	0.61/0.60*	0.61/0.60 	0.61/0.60 	0.61/0.60 	0.61/0.60 	0.61/0.60 	0.61/0.60

% simple local average feature
cvtrainLinearClassifier(X,Y,[],[],'objFn','rarlr_cg','arb',ones(5,1)./sqrt(5)); 
%(ave)	0.87/0.85 	0.93/0.93 	0.94/0.93 	0.94/0.94 	0.94/0.94 	0.94/0.94 	0.94/0.94*	l=5
%(ave)	0.79/0.79 	0.93/0.93 	0.93/0.93 	0.93/0.93 	0.93/0.94 	0.94/0.94*	0.93/0.94 	l=3
%(ave)	0.66/0.66 	0.89/0.89*	0.89/0.89 	0.89/0.89 	0.89/0.89 	0.89/0.89 	0.89/0.89 	l=2

% full ar model
cvtrainLinearClassifier(X,Y,[],[],'objFn','rarlr_cg','arb',eye(5));
%(ave)	0.87/0.86 	0.95/0.94 	0.95/0.94*	0.94/0.94 	0.95/0.94 	0.94/0.94 	0.94/0.94 	l=5
%(ave)	0.79/0.79 	0.93/0.93 	0.93/0.93 	0.94/0.93 	0.93/0.94*	0.94/0.93 	0.94/0.93 	l=3
%(ave)	0.66/0.66 	0.89/0.89 	0.89/0.89 	0.89/0.89 	0.89/0.89 	0.89/0.89*	0.89/0.89 	l=2

% exp weighted back-wards, longer history with few parameters
nb=2;
arb=zeros(2^nb-1,nb);t=0;for i=1:size(arb,2); l=2^(i-1); arb(t+(1:l),i)=1; t=t+l; end;
arb=repop(arb,'/',sqrt(sum(arb.^2))); %normalize reg
tic,cvtrainLinearClassifier(X,Y,[],[],'objFn','rarlr_cg','arb',arb,'PCmethod','adaGrad'),toc
%(ave)	0.99/0.99 	0.99/0.99 	0.99/0.99 	0.99/0.99*	1.00/0.99 	1.00/0.99 	1.00/0.99 	nb=5
%(ave)	0.87/0.85 	0.98/0.97 	0.98/0.97 	0.98/0.97*	0.98/0.97 	0.98/0.97 	0.98/0.97 	nb=4
%(ave)	0.87/0.86 	0.95/0.94*	0.94/0.94 	0.94/0.94 	0.94/0.94 	0.94/0.94 	0.94/0.94 	nb=3
%(ave)	0.79/0.79 	0.94/0.94 	0.94/0.94 	0.95/0.94*	0.94/0.94 	0.94/0.94 	0.94/0.94 	nb=2

										  % overlapping exp weight
nb=5;
arb=zeros(2^(nb-1),nb);for i=1:size(arb,2); arb(1:2^(i-1),i)=1; end;
arb=repop(arb,'/',sqrt(sum(arb.^2))); %normalize reg
cvtrainLinearClassifier(X,Y,[],[],'objFn','rarlr_cg','arb',arb)
%(ave)	0.93/0.93 	0.98/0.97 	0.98/0.98 	0.98/0.98*	0.98/0.98 	0.98/0.98 	0.98/0.98 	nb=5



% real data:
expt='external_data/physiobank/eegmmidb';
subj='S042';
label='mm';
session=[];
z=struct('expt',expt,'subj',subj,'label',label,'session',session);
commonpreproc={'load' {'runfn' 'physiolabels'} {'stdpreproc' 'eegonly',1} {'retain' 'dim' 'time' 'range' 'between' 'vals' [0 3000]}};
fn={'greg.0whtbpcovfMA2LR'    'cvtrain' 'eegonly' {'wht' 'reg',[-.0,2],'blockIdx',[]} 'filt8-27' 'cov' 'rarlr_cg'}
fn={'greg.0whtbpcovfMA2LR'    'cvtrain' 'eegonly' {'wht' 'reg',[-.0,2],'blockIdx',[]} 'filt8-27' 'cov' 'arb',[ones(2,1)./sqrt(2)] 'rarlr_cg'}
runsubFn(fn{2},z,commonpreproc{:},fn{3:end-1},fn{end})
