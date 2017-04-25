function [wb,f,J,obj,tstep]=embedlr_cg(X,Y,R,varargin);
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
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',5e-2,'objTol',1e-4,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,...
				  'PCmethod','adaptDiag','PCalpha',exp(-log(2)/14),'PClambda',.25,'PCminiter',[10 20],...
				  'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0,'taus',[],'breaks',[],'rescaledv',0);
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
Y(isnan(Y))=0; % convert NaN's to 0 so are ignored

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);
if ( size(Y,2)==N ) Y=Y'; end; L=size(Y,2); % ensure Y=[N x L]
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


% check if it's more efficient to sub-set the data, because of lots of ignored points
oX=X; oY=Y;
incInd=any(Y~=0,2);
% TODO: fix this so it works with a buffer for the convolution!
if ( 0 && sum(incInd)./size(Y,1) < opts.incThresh ) % if enough ignored to be worth it
   if ( sum(incInd)==0 ) error('Empty training set!'); end;
   X=X(:,incInd); Y=Y(incInd,:);
end
% pre-compute stuff needed for gradient computation
Yi=Y;
if(max(size(Yi))==numel(Yi)) % convert to indicator
  Yi=cat(2,Yi(:)>0,Y(:)<0); % [ N x L ]
end
if( isa(X,'single') ) Yi=single(Yi); else Yi=double(Yi); end; % ensure is right data type
Yi(Yi<0)=0; Y1=Yi(:,1); sY=sum(Yi,2);
if ( ~isempty(wght) )  Yi=repop(Yi,'*',wght); end % apply example weighting

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( ~isempty(wb) )
  ntau=(numel(wb)/L-1)/nf;
  W  = reshape(wb(1:nf*ntau*L),[nf,ntau,L]);       % weight vector [dim x ntau x L]
  b  = wb(nf*ntau*L+(1:L)');   % bias [L x 1]  
else
  % prototype classifier seed
  alpha= Yi;
  alpha= repop(alpha,'./',sum(alpha,1));
  Xmu  = X*alpha; % centroid of each class
  W    = repop(Xmu,'-',mean(Xmu,2)); % subtract the data center
  W    = repop(W,'/',sum(W.*W,1));   % scale to unit magnitude output
  b    = -(W'*mean(Xmu,2))';         % offset to average 0 for global mean point
  W    = reshape(W,[size(W,1),1,size(W,2)]); % [ nf x 1 x L ]
  b    = b(:); % [L x 1]
end 
							  % limit to the number of time points wanted if needed
taus=opts.taus;
if ( isempty(taus) )      taus=0:size(W,2)-1;
elseif ( numel(taus)==1 ) taus=0:taus-1;
end;
if (    size(taus,2)>size(W,2) )  W=cat(2,W,zeros(size(W,1),size(taus,2)-size(W,2),size(W,3))); 
elseif( size(taus,2)<size(W,2) )  W=W(:,1:size(taus,2),:); % remove
end
breaks=opts.breaks;
if ( ~isempty(breaks) )
  if( numel(breaks)==1 ) if ( breaks>0 ) breaks=1:N/breaks:N-1; else breaks=1:-breaks:N-1; end; end;
  if( size(breaks,1)==1 ) breaks=breaks'; end; % ensure is col vec
  if( breaks(1)~=1 ) breaks=[1;breaks(:)]; end; % add the 1st element if not already there
  if ( size(taus,1)>1 && ~isequal(breaks,[1:floor(N/numel(breaks)):N-1]') )
	 error('Breaks only supported with ar models if all equal sized blocks');		 
  end
end

Rw   = R*W;
f    = fwdPass(X,W,b,taus,breaks); % get the example activations, [N x L]
if ( size(Y,1)>1 && opts.rescaledv )
  f  = repop(f,'-',max(f,[],2)); % re-scale for numerical stability
end
p    = exp(f); % =Pr(x|y+) = p1 % [N x L]
% dL_i = gradient of the loss = (1-p_y) = 1-prob of the true class = { py if y=y, 1-py otherwise i.e. 
if ( size(p,2)==1 ) % binary problem
  p  = p/(1+p);
  %g   = 1./(1+exp(-(Y.*f)));    % Pr(Y_true|x,w,b,fp) % [N x L], numerically more accurate
  %dLdf= Y.*(1-g); % comp more efficient given prob_true_class, and Y=+/-1
  dLdf= Y1 - p.*sY; % dLdf = Yerr % [N x L]
else % multi-class
  p   = repop(p,'/',sum(p,2)); % [N x L] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b)); % softmax
  dLdf= Yi - repop(p,'*',sY); % [N x L]
end
dJ   = bwdPass(X,dLdf,2*Rw,W,b,taus,breaks);
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
if ( size(p,2)==1 ) % binary problem
  Ed  = -(sum(log(max(p(Yi(:,1)>0),1e-120)))+sum(log(max(1-p(Yi(:,2)>0),1e-120)))); 
else
  Ed  = -log(max(p(:),1e-120))'*Yi(:); % Ed = sum(-log(Pr(Y_true)))
end
Ew   = W(:)'*Rw(:);     % -ln P(w,b|R);
J    = Ed + Ew + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
			  0,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
% summary information for the adaptive diag-hessian estimation
% seed with prior for identity PC => zero-mean, unit-variance
N=0; Hstats = zeros(size(dJ,1),5); %[ sw sdw sw2 sdw2 swdw]
W0=W; b0=b;
nStuck=0;iter=1;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

  restart=false;
  oJ=J; oMr=Mr; or2=r2; oW=W; ob=b; % record info about prev result we need

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
	W0  = W;       b0=b;     
	f0  = f;
   dw  = reshape(d(1:numel(W)),size(W)); db=d(numel(W)+1); 
	df  = fwdPass(X,dw,db,taus,breaks); % get the forward activations in the search direction
	
	% pre-compute regularized weight contribution to the line-search
   Rw =R*W;      dRw=dw(:)'*Rw(:);  % scalar or full matrix
   Rd =R*dw;     dRd=dw(:)'*Rd(:);	
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,J,Ew,Ed); 
   end
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
		f    = f0 + tstep*df;
		if ( size(f,2)>1 && opts.rescaledv )
		  f  = repop(f,'-',max(f,[],2)); % re-scale for numerical stability
		end
		p    = exp(f); % =Pr(x|y+) = p1 % [N x L]
		if ( size(p,2)==1 ) % binary problem
		  p   = p/(1+p);
		  dLdf= Y1 - p.*sY; % dLdf = Yerr % [N x L]
		else % multi-class
		  p   = repop(p,'/',sum(p,2)); % [N x L] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b)); % softmax
		  dLdf= Yi - repop(p,'*',sY); % [N x L]
		end
		dtdJ = -(2*(dRw+tstep*dRd) - df(:)'*dLdf(:));

		if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		  fprintf('%d) Numerical issues falling back on re-scaled dv\n',iter);
		  oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		  opts.rescaledv=true; continue;
		end;
		
		if ( opts.verb>2 )
		  if ( size(Y,2)==1 ) % binary problem
			 Ed  = -(sum(log(max(p(Yi(:,1)>0),1e-120)))+sum(log(max(1-p(Yi(:,2)>0),1e-120)))); 
		  else
			 Ed  = -log(max(p(:),1e-120))'*Yi(:); % Ed = sum(-log(Pr(Y_true)))
		  end
		  Ew   = W(:)'*Rw(:) + 2*tstep*dRw + tstep*tstep*dRd;     % -ln P(w,b|R);
		  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,j,dtdJ,0,Ed+Ew,Ew,Ed); 
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
   W  = W0 + tstep*dw; 
   b  = b0 + tstep*db;
   Rw = Rw + tstep*Rd;
   % compute the other bits needed for CG iteration, i.e. a full gradient computation
	dJ = bwdPass(X,dLdf,2*Rw,W,b,taus,breaks);
   MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   
   % compute the function evaluation
	if ( size(Y,2)==1 ) % binary problem
	  Ed  = -(sum(log(max(p(Yi(:,1)>0),1e-120)))+sum(log(max(1-p(Yi(:,2)>0),1e-120)))); 
	else
	  Ed  = -log(max(p(:),1e-120))'*Yi(:); % Ed = sum(-log(Pr(Y_true)))
	end
   Ew   = W(:)'*Rw(:);% P(w,b|R);
   J    = Ed + Ew + opts.Jconst;       % J=neg log posterior
   if( ~isequal(mu,0) ) Emu=w'*mu; J=J+Emu; end;
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
   end   
   
   %------------------------------------------------
   % convergence test
   if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(r02,r2);
   elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,r2); % conv if smaller than best single step
   end
   madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
   if ( r2<=opts.tol || ... % small gradient + numerical precision
        r2< r02*opts.tol0 || ... % Wolfe condn 2, gradient enough smaller
        neval > opts.maxEval || ... % abs(odtdJ-dtdJ) < eps || ... % numerical resolution
        madJ <= opts.objTol || madJ < opts.objTol0*dJ0 ) % objective function change
      break;
   end;    

	% TODO : fix the backup correctly when the line-search fails!!
   if ( J > oJ*(1.001) || isnan(J) ) % check for stuckness
     if ( opts.verb>=1 )
		 warning(sprintf('%d) Line-search Non-reduction - aborted\n',iter));
	  end;
     J=oJ; W=oW; b=ob; Mr=oMr; r2=or2; tstep=otstep*.01;
	  opts.rescaledv=true;fprintf('%d) Numerical issues falling back on re-scaled dv\n',iter);
     f = fwdPass(X,W,b,taus,breaks);
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
   W=W0; b=b0;
end;

% compute the final performance with untransformed input and solutions
Rw  = R*W;       % scalar or full matrix
p   = exp(f);    % [L x N] =Pr(x|y_+) = exp(w_ix+b)./sum_y(exp(w_yx+b));
if ( size(p,2)==1 ) % binary problem
  p   = p/(1+p);
  Ed  = -(sum(log(max(p(Yi(:,1)>0),1e-120)))+sum(log(max(1-p(Yi(:,2)>0),1e-120)))); 
else
  p   = repop(p,'/',sum(p,2)); % soft-max!
  Ed  = -log(max(p(:),1e-120))'*Yi(:); % Ed = sum(-log(Pr(Y_true)))
end
Ew  = W(:)'*Rw(:);     % -ln P(w,b|R);
J   = Ed + Ew + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
          iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
end

% compute final decision values.
if ( ~all(size(X)==size(oX)) )
  f = fwdPass(oX,W,b,taus,breaks);  
end;
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

%-----------------------------------------------------------------------
function [f,ftau]=fwdPass(X,W,b,taus,breaks)
% compute the forward pass to get the activations for each example
if ( size(W,3)==1 )
  f0  = W'*X; % [tau x N]
else
  f0  = tprod(W,[-1 1 3],X,[-1 2]); % [tau x N x L]
end
f   = zeros([size(f0,2),size(f0,3)],class(X));
for ti=1:size(taus,2); % TODO: can we vectorize this?
  tau=taus(:,ti);
  ftau=zeros(size(f));
  if ( numel(tau)==1 ) % tau is simply a time shift
	 if ( tau>=0 )
		ftau(tau+1:end,:) = f0(ti,1:end-tau,:);
	 else
		ftau(1:end+tau,:) = f0(ti,-tau+1:end,:);
	 end
	 if ( ~isempty(breaks) )
		if (tau>0) 
		  ftau(breaks(2:end) + (0:tau-1),:) =0;
		elseif ( tau<0 )
		  ftau(breaks(2:end) + (tau:-1),:)  =0;
		end
	 end
  else % tau is an ar set to apply to the data
	 if ( isempty(breaks) )
		ftau(:,:) = filter(tau,1,f0(ti,:,:),[],2);
	 else % split along breaks to make it work
		fsz=[size(f,1)/numel(breaks) numel(breaks) size(f,2)];
		ftau(:,:) = reshape(filter(tau,1,reshape(f0(ti,:,:),fsz),[],2),size(f));
	 end
  end
  f = f + ftau;
end
% include the (class specific) bias
if ( size(f,2)>1 ) 
  f  = repop(f,'+',b(:)');
else
  f  = f+b;
end
return;

%-----------------------------------------------------------------------
function [dJ]=bwdPass(X,dLdf,dRdw,W,b,taus,breaks);
% compute the backward pass to get the error gradients w.r.t. the loss
%
% Inputs:
%   X    - [nf x N]
%   dLdf - [N x L]
%   dRdw - [nf x ntau x L]
%   W    - [nf x ntau x L]
%   b    - [1 x L]
%   taus - [ntau x 1]
dLdw=zeros(size(W));
for ti=1:size(taus,2); %TODO: can we vectorize this?
  tau=taus(:,ti);
  if ( numel(tau)==1 ) % tau is simply a time shift
	 if ( tau>=0 ) % backward shift
		dLdftau =                           [dLdf((tau+1):end,:);zeros(tau,size(dLdf,2))]; % [ N x L]
	 else % forward shift
		dLdftau = [zeros(-tau,size(dLdf,2)); dLdf(1:end+tau,:)]; % [N x L]
	 end
	 if ( ~isempty(breaks) )
		if (tau>0) 
		  dLdftau(breaks(2:end) + (-tau:-1),:) =0;
		elseif ( tau<0 )
		  dLdftau(breaks(2:end) + (0:-tau-1),:) =0;
		end	 
	 end
  else % tau is an ar filter to apply
% apply an inverse filter to get back to per-input-time-point & zero-pad to include end points
	 dLdftau = dLdf; % [ N x L ]
	 if ( ~isempty(breaks) ) %dLdftau -> [NxbreaksxL]
		dLdftau=reshape(dLdf,[size(dLdf,1)/numel(breaks),numel(breaks),size(dLdf,2)]); 
	 end
	 dLdftau = filter(tau(end:-1:1),1,...  
							cat(1,dLdftau,zeros(numel(tau)-1,size(dLdftau,2),size(dLdftau,3))),[],1);
	 dLdftau = reshape(dLdftau(numel(tau):end,:,:),[],size(dLdftau,3)); % [N x L]
  end
  
  dLdw(:,ti,:) = -(X*dLdftau); % [ nf x 1 x L ]
end
dLdb = -sum(dLdf,1); % [ 1 x 1 ]
dJ=[ dRdw(:) + dLdw(:); ...   % w
     dLdb(:)];       % b
return;


%-----------------------------------------------------------------------------
function []=testCase()
%Make a Gaussian balls + outliers test case
nd=100; nClass=800;
[X,Y]=mkMultiClassTst([zeros(1,nd/2-1) -1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) 1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) .2 .5 zeros(1,nd/2-1)],[nClass nClass 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
wb0=randn(size(X,1)*3+1,1);

tic,lr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0);toc
tic,[wb,f,J]=embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0);toc

[J,dJ]=embedLRFn(wb,X,Y,0)
tic,[wb2,f,J]=embedlr_cg(X,Y,0,'verb',1,'wb',wb,'maxIter',0);toc

tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'taus',[0 1]);toc % with taus, neg
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'breaks',41);toc % with breaks
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'breaks',-20);toc % with breaks
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'taus',[1 0;0 1]');toc % with ar-basis
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'taus',[1 0;0 1]','breaks',41);toc % with ar-basis+breaks
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'taus',[1 0 0;0 .5 .5]');toc

										  % multi-class version
[X,Yl]=mkMultiClassTst([-1 0;1 0;0 1;0 -1],[400 400 400 400],[.2 .2]);[dim,N]=size(X);
Y=lab2ind(Yl)';
wb0=randn(size(X,1)*2+1,min(size(Y)));

tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1);toc
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'wb',wb0);toc
tic,[wb,f,J] =embedlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'taus',[0 1]);toc % with taus
