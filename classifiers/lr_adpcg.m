function [wb,f,J,obj,tstep]=lr_adpcg(X,Y,R,varargin);
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
%  R       - quadratic regularisation strength                                   (0)
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
  opts=struct('wb',[],'alphab',[],'dim',[],...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-3,'objTol',1e-4,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,...
				  'PCmethod','adaptDiag','PCalpha',.95,...
				  'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0);
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
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  w=zeros(nf,1);b=0;
  % prototype classifier seed
  alpha=Yi(:,1)./sum(Yi(:,1))/2-Yi(:,2)./sum(Yi(:,2))/2;
  % prototype classifier for each time-shift
  w = X*alpha;
  %clf;mimage(W,w*[1;arb*a]','diff',1); % plot the approx vs. full-rank soln
  Rw    = R*w;
  wRw   = w'*Rw(:);
  wX    = w'*oX; wX=wX(incInd); % only included points in seed
  % re-scale to sensible range, i.e. 0-mean, unit-std-dev
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
end 

Rw   = R*w;
wX   = w'*X;
f    = (wX+b)'; % [ N x L ]
p    = 1./(1+exp(-f(:))); % =Pr(x|y+) = p1 % [N x L]
% dL_i = gradient of the loss = (1-p_y) = 1-prob of the true class = { py if y=y, 1-py otherwise i.e. 
dL   = Y1-p.*sY; % dL = Yerr % [N x L]
dLdw = -(X*dL);
dLdb = -sum(dL,1);
dJ   = [2*Rw + dLdw; ...
               dLdb];
% precond'd gradient:
%  [H  0  ]^-1 [ Rw-X'((1-g).Y))] 
%  [0  bPC]    [   -1'((1-g).Y))] 
PC   = 1; % initially no PC
MdJ  = PC.*dJ; % pre-conditioned gradient

Mr   =-MdJ;
d    = Mr;
dtdJ =-(d'*dJ);
r2   = dtdJ;

% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
Ed   = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); 
Ew   = w'*Rw(:);     % -ln P(w,b|R);
J    = Ed + Ew;      % J=neg log posterior

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
  oJ= J; oMr  = Mr; or2=r2; ow=w; ob=b; % record info about prev result we need

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
	w0  = w;       b0=b;      
   dw  = d(1:nf); db=d(nf+1);
   wX0 = wX;
   dwX = dw'*X;
	% pre-compute regularized weight contribution to the line-search
   Rw =R*w;      dwRw =dw'*Rw;  % scalar or full matrix
   Rdw=R*dw;     dwRdw=dw'*Rdw;	
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,J,Ew,Ed); 
   end
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values      
		wX  = wX0+tstep*dwX;%[1xN]
		b   = b0 +tstep*db;%[1x1]
		f   = (wX + b)'; % [NxL]
		p   = 1./(1+exp(-f)); % =Pr(x|y+) % [NxL]
		dL  = Y1-p.*sY; % [NxL]
      dtdJ= -(2*(dwRw+tstep*dwRdw) - dwX*dL - db*sum(dL));%[1x1]
		
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
      step  = step * nstep ;           % absolute step
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
	dJ  = [2*Rw + dLdw; ...
			        dLdb];

										  % update the pre-conditioner
	if ( strcmp(opts.PCmethod,'adaptDiag') )
	  % update the diag-hessian estimation
	  wb     = [w;b];
										  % update the moment information
	  % half-life = half-life = log(.5)/log(alpha) 
	  N      =      N*opts.PCalpha + (1-opts.PCalpha); % N.B. forall alpha, lim_t->inf \sum N*alpha+(1-alpha)*1 = 1
	  Hstats = Hstats*opts.PCalpha + (1-opts.PCalpha)*[wb dJ wb.^2 dJ.^2 wb.*dJ];
										  % update the diag-hessian estimate
	  wdJvar = (Hstats(:,5) - Hstats(:,1).*Hstats(:,2)./N);
	  wvar   = (Hstats(:,3) - Hstats(:,1).*Hstats(:,1)./N);
	  %dJvar  = (Hstats(:,4) - Hstats(:,2).*Hstats(:,2)./N);
	  H      =  wdJvar ./ (wvar+1e-8);
	  %expvar =  wdJvar ./ sqrt(abs(wvar)+1e-8) ./ sqrt(abs(dJvar)+1e-8); % explained variance = goodness-of-fit
										  % test if we should update PC

	  %Hrec.wb(:,iter)=wb; Hrec.dJ(:,iter)=dJ; % detail info recording to check quality of the hessian estimate

	  if ( N>.5 ) % hessian is trustworthy, i.e. sufficient fraction of the half-life achieved
				% DEBUG: test comparsion between est and true diag-hessian info
				% [d,dy,dh,ddy,ddh]=checkgrad(@(w) primalLRFn(w,X,Y,R),wb,1e-5,1,1);
		      % clf;semilogy(abs(H./ddh),'linewidth',1); [ans,si]=sort(abs(log(H./ddh)),'descend');
		      % clf;di=si(1);plot(Hrec.wb(di,:)',Hrec.dJ(di,:)','-*','linewidth',1);title(sprintf('H=%g Htrue=%g',H(di),ddh(di)));
            % clf; plot([H ddh expvar*mean(H)],'linewidth',2);legend('H_track','Htrue_fd','expvar');

		 H    = H./mean(H); % remove total scaling effects...
		 condest=H.*PC; % estimated improvement factor for the condition number
		 if ( max(abs(condest))./min(abs(condest)) > 10 )
			if ( opts.verb>1 ) fprintf('%d) PC update\n',iter); end;
			restart = true;
			PC = 1./(abs(H)+1e-8); % N.B. PC is inverse Hessian est.. with guard for divide by 0
		 end
	  end
	end

	MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   
   % compute the function evaluation
	p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
	Ed   = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); % P(D|w,b,fp)
   Ew   = w'*Rw(:);% P(w,b|R);
   J    = Ed + Ew; 
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,w(1),w(2),J,Ew,Ed,r2);
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
     J=oJ; w=ow; b=ob; Mr=oMr; r2=or2; tstep=otstep*.01;
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
	end
   d     = Mr+beta*d;     % conj grad direction
	if ( theta~=0 ) d = d - theta*(Mr-oMr); end; % include the modification factor
   dtdJ  = -d'*dJ;         % new search dir grad.
   if( dtdJ <= 0 || restart || ...  % non-descent dir switch to steepest
	  (opts.restartInterval>0 && mod(iter,opts.restartInterval)==0))         
     if ( dtdJ<=0 && opts.verb >= 2 ) fprintf('non-descent dir\n'); end;
     d=Mr; dtdJ=-d'*dJ;
	  if( tstep<=eps ) tstep=min(sqrt(abs(J/max(dtdJ,eps))),1); end;
   end; 
   
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

% compute the final performance with untransformed input and solutions
Rw  = R*w;                  % scalar or full matrix
f   = (wX + b)'; % [N x L]
% Convolve with the ar coefficients
p   = 1./(1+exp(-f));     % [L x N] =Pr(x|y_+) = exp(w_ix+b)./sum_y(exp(w_yx+b));
p(p==0)=eps; p(p==1)=1-eps; % guard for log of 0
Ed  = -(log(p)'*Yi(:,1)+log(1-p)'*Yi(:,2)); % expected loss
Ew  = w'*Rw(:);     % -ln P(w,b|R);
J   = Ed + Ew;       % J=neg log posterior
if ( opts.verb >= 0 ) 
   fprintf(['\n%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,w(1),w(2),J,Ew,Ed,r2);
end

% compute final decision values.
if ( ~all(size(X)==size(oX)) ) f   = w'*oX + b; end;
f = reshape(f,[size(oY,1) 1]);
obj = [J Ew Ed];
wb=[w(:);b];
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
nd=100; nClass=800;
[X,Y]=mkMultiClassTst([zeros(1,nd/2-1) -1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) 1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) .2 .5 zeros(1,nd/2-1)],[nClass nClass 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
wb0=randn(size(X,1)+1,1);

% normal problem -- stiff as is non-regularized
R=0;
% normal problem -- very smooth because of regularizer
R=1;
% make a super-'stiff' problem, i.e. make some dimensions much more costly to use than others
R=diag([1e5 zeros(1,nd-2) 1e5]);


tic,[wb,f,J]=simplr_cg(X,Y,R,'verb',1,'objTol0',1e-10,'wb',wb0);toc
tic,[wb,f,J]=lr_adpcg(X,Y,R,'verb',1,'objTol0',1e-10,'wb',wb0,'PCmethod','adaptDiag','PCalpha',.95);toc


