function [wb,f,J,obj,tstep]=dnn_cg(X,Y,R,varargin);
% Regularised linear Logistic Regression Classifier
%
% [wb,f,J,obj]=dnn_cg(X,Y,C,varargin)
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
%  PCmethod - [str] type of pre-conditioner to use.                       ('adaGrad')
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
  opts=struct('wb',[],'alphab',[],'dim',[],'wbsz',[],...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-1,'objTol',1e-4,'objTol0',1e-4,...
				  'objTgt',[],...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'maxStuck',3,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,'PCmethod','none',...
				  'incThresh',.66,'optBias',0,'maxTr',inf,...
				  'nonlinType','exp','h',0,...
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
if(size(Y,1)==N) Y=Y'; end; % ensure Y is col vector [ L x N ] vector
Y(isnan(Y))=0; % convert NaN's to 0 so are ignored

% reshape X to be 2d for simplicity
X=reshape(X,[nf N]);

Yi=Y;
if ( all(Y(:)>=0 & Y(:)==ceil(Y(:))) ) % class labels input, convert to indicator matrix
  Yl=Y;key=unique(Y);key(key==0)=[];
  Y=zeros(numel(key),size(X,2));for l=1:numel(key); Y(l,:)=Yl==key(l); end;	 
end
Y(Y(:)<0)=0; % remove negative indicators
% pre-compute stuff needed for gradient computation
sY=sum(Y,1);

% generate an initial seed solution if needed
wbsz=opts.wbsz;
wb  =opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )
  W0=ortho_init(wb0sz,1,1); % important when don't have sufficient width in the network...
  wb=[]; for di=1:numel(W0); wb=[wb0;W0{di}(:)]; end;
else
  w=wb(1:nf);        % weight vector
  b=wb(nf+1);        % bias
end 

				% build index expressions for each layer and unpack the weights
wbIdx={};sz=int32([wbsz(1:end-1)+1; wbsz(2:end)]);
for li=1:numel(wbsz)-1;
  wbIdx{li}=int32(reshape(sum(prod(sz(:,1:li-1)))+(1:prod(sz(:,li))),sz(:,li)));
  wIdx{li} =wbIdx{li}(1:end-1,:); bIdx{li}=wbIdx{li}(end,:);
end
wbIdx2p=int32(prod(sz(:,1))+1:(sum(prod(sz))));%params for all but the 1st layer

										  % set the pre-conditioner
PC=1; if ( ~isempty(opts.wPC) ) PC=opts.wPC; end

										  % compute the fwd-bwd info + gradients
[J,dJ,g,f,dLdf,p,Ed,Ew]=dnn(wb,X,Y,R,wbsz,opts.nonlinType,opts.h);

										  % initialize the CG info
										  % precond'd gradient:
										  %  [H  0  ]^-1 [ Rw-X'((1-g).Y))] 
										  %  [0  bPC]    [   -1'((1-g).Y))] 
MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
ddJdw=-(d(:)'*dJ(:));
r2   = ddJdw;

% Set the initial line-search step size
step=abs(opts.step); 
if( step<=0 ) step=min(sqrt(abs(J/max(sum(ddJdw),eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,w(1),w(2),J,Ew,Ed,sum(r2));
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
maPC=PC;
w0=w; b0=b;
nStuck=0;iter=1;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

  restart=false;
  oJ= J; oMr  = Mr; or2=r2; owb=wb; % record info about prev result we need

		%---------------------------------------------------------------------
		% Secant method for the root search.

												  % setup for the line-search
  ostep=inf;otstep=tstep;step=otstep; % prev step size is first guess!
  oddJdw=ddJdw; % one step before is same as current
					 % save the starting point
  wb0 =wb;
  ddJdw0 = ddJdw;
				  % pre-extract the later layers parameters and search direction
  wb2p0=wb(wbIdx2p);
  d2p  =d(wbIdx2p);

					  % pre-compute 1st layer activations to save some time later
  f10 =f{1};
  d1  =d(wbIdx{1});
  df1 =repop(d1(1:end-1,:)'*X,'+',d1(end,:)');
  Rdw1   = (R*d1(1:end-1,:));
  dw1Rw1 = reshape(wb(wbIdx{1}(1:end-1,:)),1,[])*Rdw1(:);
  dw1Rdw1= reshape(d1(1:end-1,:),1,[])          *Rdw1(:);
  if ( 0 || opts.verb>1 )
	 J_true=0; ddJdw_true=0;
	 if ( opts.verb>2 ) % full computation for true ddJdw & J
		[J_true,dJ_true,g_true,f_true]=dnn(wb,X,Y,R,wbsz,opts.nonlinType,opts.h);
		ddJdw_true = -d(:)'*dJ_true(:);
	 end
	 fprintf('\n%2da stp=%8f-> ddJdw=%8f/%8f @ J=%8f/%8f (%8f+%8f)\n',0,0,ddJdw,ddJdw_true,J,J_true,Ew,Ed); 
	 if ( opts.verb > 3 )
		clf;
		subplot(211); plot(0,ddJdw,'r*');hold on;text(0,double(ddJdw),num2str(0));
		grid on;title('ddJdw');
		subplot(212); plot(0,J,'r*');hold on;text(0,double(J),num2str(0)); 
		title('J');
	 end;
  end

								% loop over possible step sizes to find the best one
  for j=1:opts.maxLineSrch;
	 neval=neval+1;
	 ooddJdw=oddJdw; oddJdw=ddJdw; % prev and 1 before grad values      
											 % backup the gradient to compute the dw'dJ
	 f1   = f10  + tstep*df1;%[D_l x N]
	 wb2p = wb2p0+ tstep*d2p; % updated parameters for later layers

	 switch (opts.nonlinType)
		case 'srelu';
		  g1    = max(0,f1); % output map % [M x N]
		  qis   = -opts.h<f1 & f1<opts.h;										  
		  g1(qis(:))=(f1(qis(:)).^2)./(4*opts.h) + f1(qis(:))/2 + opts.h/4; %quad reg f(x)=x.^2/(4*h)+x/2+h/4
		  dg1df1= double(g1>0);
		  dg1df1(qis(:))=f1(qis(:))/opts.h/2+.5;
		case 'relu';  g1 = max(0,f1); dg1df1=g1>0; % relu
		case 'exp';   g1 = exp(f1);   dg1df1=g1;   % exp
		otherwise; 'unrecog activation function';
	 end
    % full forward-backward pass for later layers + get the dLdf at this point
	 [J,dJ2p,g(2:end),f(2:end),dLdf2]=dnn(wb2p,g1,Y,R,wbsz(2:end),opts.nonlinType,opts.h);
				 % backup dL/dg by one layer target dL/dg= dL/df_{+1} df_{+1}/dg
	 % dL/df = dL/df_{+1} df_{+1}/dg dg/df = dL/df_{+1} W_{+1} dg/df % [D_{l} x N]
	 dLdf1 =  (wb2p(wbIdx{2}(1:end-1,:)-wbIdx2p(1)+1)*dLdf2).*dg1df1; 
	 ddJdw = -(2*(dw1Rw1+tstep*dw1Rdw1) - df1(:)'*dLdf1(:)) ... % 1st layer
				-(d2p(:)'*dJ2p(:)); % later layers
	 
	 if ( opts.verb > 1 )		  
		[J_true,dJ_true,g_true,f_true]=dnn(wb0+tstep*d,X,Y,R,wbsz,opts.nonlinType,opts.h);
		ddJdw_true = -d(:)'*dJ_true(:);
		fprintf('%2da stp=%8f-> ddJdw=%8f/%8f @ J=%8f/%8f (%8f+%8f)\n',j,tstep,ddJdw,ddJdw_true,J,J_true,Ew,Ed); 
		if ( opts.verb > 3 )
		  subplot(211);plot(tstep,ddJdw,'*'); text(double(tstep),double(ddJdw),num2str(j));
		  subplot(212);plot(tstep,J,'*'); text(double(tstep),double(J),num2str(j));
		end
	 end;
	 
										  % convergence test, and numerical res test
	 if(iter>1||j>2) % Ensure we do decent line search for 1st step size!
		if ( j==opts.maxLineSrch || ... % max iterations
			  abs(ddJdw) < opts.lstol0*abs(ddJdw0) || ... % Wolfe 2, gradient enough smaller
			  abs(ddJdw*step) <= opts.tol )              % numerical resolution
		  break;
		end
	 end
	 
										  % now compute the new step size
										  % backeting check, so it always decreases
	 if ( ooddJdw*oddJdw < 0 && oddJdw*ddJdw > 0 ...      % ooddJdw still brackets
			&& abs(step*ddJdw) > abs(oddJdw-ddJdw)*(abs(ostep+step)) ) % would jump outside 
		step = ostep + step; % make as if we jumped here directly.
		oddJdw = -sign(oddJdw)*sqrt(abs(oddJdw))*sqrt(abs(ooddJdw)); % geometric mean
	 end
	 ostep = step;
										  % *RELATIVE* secant step size
	 dddJdw = oddJdw-ddJdw; 
	 if ( dddJdw~=0 ) nstep = ddJdw/dddJdw; else nstep=1; end; % secant step size, guard div by 0
	 nstep = sign(nstep)*max(opts.minStep,min(abs(nstep),opts.maxStep)); % bound growth/min-step size
	 step  = step * nstep ;           % absolute step
	 tstep = tstep + step;            % total step size      
  end % end of line search
  if ( opts.verb > 2 ) fprintf('\n'); end;
  
										  % Update the parameter values!
										  % N.B. this should *only* happen here!
  wb  = wb0 + tstep*d;
						% compute the full gradient = forward+backward pass
						% TODO: replace this with partial re-comp only for layer 1
  neval=neval+1;
  [J,dJ,g,f,dLdf,p,Ed,Ew]=dnn(wb,X,Y,R,wbsz,opts.nonlinType,opts.h);
  % update CG necessary parts
  MdJ= PC.*dJ; % pre-conditioned gradient
  Mr =-MdJ;
  r2 = abs(Mr(:)'*dJ(:));
  
  if(opts.verb>0)   % debug code      
	 fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
				iter,neval,w(1),w(2),J,Ew,Ed,sum(r2));
  end   
  
									%------------------------------------------------
									% convergence test
  if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=max(sum(r02),sum(r2));
  elseif( iter<5 )   dJ0=max(dJ0,abs(oJ-J)); r02=max(r02,sum(r2)); 
  end
  madJ=madJ*(1-opts.marate)+abs(oJ-J)*(opts.marate);%move-ave objective grad est
  if ( sum(r2)<=opts.tol || ... % small gradient + numerical precision
		 sum(r2)< r02*opts.tol0 || ... % Wolfe condn 2, gradient enough smaller
		 neval > opts.maxEval || ... % abs(oddJdw-ddJdw) < eps || ... % numerical resolution
		 madJ <= opts.objTol || madJ < opts.objTol0*dJ0 ) % objective function change
	 break;
  end;    
  if( ~isempty(opts.objTgt) && J<opts.objTgt)
	 if ( opts.verb>0) fprintf('Objective achieved') end;
	 break;
  end;

				  % TODO : fix the backup correctly when the line-search fails!!
  if ( J > oJ*(1.001) || isnan(J) ) % check for stuckness
	 if ( opts.verb>=1 )
		warning(sprintf('%d) Line-search Non-reduction - aborted\n',iter));
	 end;
	 J=oJ; wb=owb; Mr=oMr; r2=or2; tstep=otstep*.01;
	 nStuck =nStuck+1;
	 restart=true;
	 if ( nStuck > opts.maxStuck ) break; end;
  end   

  
					 %------------------------------------------------
					 % conjugate direction selection -- using the updated info
					 % compute updated info for the conjugate direction selection
% N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
%     PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
  beta=0;theta=0;
  switch opts.CGmethod;
	 case 'PR';		 
		beta = max((Mr(:)-oMr(:))'*(-dJ(:))/or2,0); % Polak-Ribier
	 case 'MPRP';
		beta = max((Mr(:)-oMr(:))'*(-dJ(:))/or2,0); % Polak-Ribier
		theta = Mr(:)'*dJ(:) / or2;% modification which makes independent of the quality of the line-search
	 case 'FR';
		beta = max(r2/or2,0); % Fletcher-Reeves
	 case 'GD'; beta=0; % Gradient Descent				
	 case 'HS'; % use Hestenes-Stiefel update
		beta=Mr(:)'*(Mr(:)-oMr(:))/(-d(:)'*(Mr(:)-oMr(:)));  
  end
  d = Mr+beta*d;     % conj grad direction
  if ( theta~=0 ) d = d - theta*(Mr-oMr); end; % include the modification factor
  ddJdw  = -d(:)'*dJ(:);         % new search dir grad.
  if( ddJdw <= 0 || restart || ...  % non-descent dir switch to steepest
		(opts.restartInterval>0 && mod(iter,opts.restartInterval)==0))         
	 if ( ddJdw<=0 && opts.verb >= 1 ) fprintf('non-descent dir\n'); end;
	 d=Mr; ddJdw=-d(:)'*dJ(:);
  end; 

end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

		% compute the final performance with untransformed input and solutions
[J,dJ,g,f]=dnn(wb,X,Y,R,wbsz,opts.nonlinType,opts.h);
if ( opts.verb >= 0 ) 
   fprintf(['\n%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,w(1),w(2),J,sum(Ew),Ed,sum(r2));
end

% compute final decision values.
obj = [J Ew Ed];
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
[X,Yl]=mkMultiClassTst([-1 0;1 0;0 1;0 -1],[400 400 400 400],[.2 .2]);[dim,N]=size(X);
Y=lab2ind(Yl)';

										  % different numbers of layers for test
% N.B. double input-dim in 1st layer output to compensate for relu discarding have the input space
wb0sz=[size(X,1) size(X,1)*2 size(Y,1)];       % 2 - layers
wb0sz=[size(X,1),size(X,1)*2,4,size(Y,1)];     % 3 - layers
wb0sz=[size(X,1),size(X,1)*2,4,4,4,size(Y,1)]; % 5 - layers
wb0sz=[size(X,1),size(X,1)*2,4,4,4,4,4,4,4,4,size(Y,1)]; % 10 - layers

% N.B. don't forget scaling to decent size
wb0  =randn(sum([(wb0sz(1:end-1)+1).*wb0sz(2:end)]),1)*1e-2; 
% more intelligent initialization?
W0=randn_init(wb0sz,1,1);
W0=ortho_init(wb0sz,1,1); % important when don't have sufficient width in the network...
wb0=[]; for di=1:numel(W0); wb0=[wb0;W0{di}(:)]; end;

								% check the correctness of the objective computation
[J,dJ,dvout,dvall,obj]=dnn_exp(wb0,X,Y,1,wb0sz);J
[wb,dvoutcg,Jcg,objcg]=dnn_cg(X,Y,1,'wb',wb0,'wbsz',wb0sz,'verb',1,'maxIter',0);
[J,dJ,dvout,dvall]=dnn_relu(wb0,X,Y,1,wb0sz);J
[wb,dvoutcg,Jcg,objcg]=dnn_cg(X,Y,1,'wb',wb0,'wbsz',wb0sz,'verb',1,'maxIter',0,'nonlinType','relu');

										  % test within a generic cg framework
nonlin='exp';  objTgt=6;
nonlin='relu'; objTgt=56;
nonlin='srelu'; h=.5; objTgt=56;
tic,[wb nfs]=nonLinConjGrad(@(w) dnn(w,X,Y,1,wb0sz,nonlin,h),wb0,'plot',0,'verb',1,'objTgt',objTgt);toc
% test special case layer-wise optimization
tic,[wb,f,J]=dnn_cg(X,Y,1,'wb',wb0,'wbsz',wb0sz,'verb',1,'objTgt',objTgt,'nonlinType',nonlin,'h',h,'objTol0',1e-10);toc
