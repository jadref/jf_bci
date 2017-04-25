function [wb,f,J,obj,tstep]=linfnlr_cg(X,Y,R,varargin);
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
%            OR
%            [LxN] matrix of per-example class information
%  R       - quadratic regularisation matrix                                   (0)
%     [1x1]       -- simple regularisation constant             R(w)=w'*R*w
%     N.B. if R is scalar then it corrospends to roughly max allowed length of the weight vector
%          good default is: .1*var(data)
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
%
% Options:
%  fwdFn   - {func_name fwd_opts} or [fn-handle] function which applys the forward computation                   ('lin_fwd')
%             This should have the signature:
%              [f]=fwdFn(X,W,b,fwd_opts,varargin{:});
%  bwdFn   - {func_name bwd_opts} or [fn-handle] function which applys the backward computation to get gradients ('lin_bwd')
%             This should have the signature:
%              [dJdw,dJdb]=bwdFn(X,dLdf,dRdw,W,b,bwd_opts,varargin{:});
%  seedFn  - {func_name seed_opts} or [fn-handle] function which generates a sensible initial seed
%             solution when none is given
%             This should have the signature:
%              [W,b]=seedFn(X,Y,R,szX,seed_opts,varargin{:})
%  dim     - [int] dimension of X which contains the trials               (ndims(X))
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
%  CGmethod - [str] type of conjugate gradients solver to use:            ('PR')
%             one-of: PR, HS, GD=gradient-descent, FR, MPRP-modified-PR
%  PCmethod - [str] type of pre-conditioner to use.                       ('wbg')
%             one-of: none, wb, wb0, wbg, adaptDiag=adaptive-diagonal-PC
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
  opts=struct('wb',[],'alphab',[],'dim',[],'rdim',[],'ydim',[],'N2w',[],...
				  'fwdFn','lin_fwd','bwdFn','lin_bwd','seedFn','lin_seed','fwdbwdOpts',{{}},...
				  'mu',0,'Jconst',0,...
              'maxIter',1000,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-2,'objTol',0,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,...
				  'PCmethod','wbg',...
				  'incThresh',.66,'maxTr',inf,...
              'getOpts',0,'rescaledv',0);
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( ~isempty(opts.N2w) ) warning('example data-set info in N2w is ignored'); end;
if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
% Ensure all inputs have a consistent precision
if(islogical(Y))Y=single(Y); end;
if(isa(X,'double') && ~isa(Y,'double') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; minp=eps; maxf=single(-log(minp)); else eps=1e-16; minp=1e-15; maxf=-log(minp); end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence

fwdbwdOpts =opts.fwdbwdOpts;   if ( ~iscell(fwdbwdOpts) ) fwdbwdOpts={fwdbwdOpts}; end;
fwdFn =opts.fwdFn; fwdOpts={}; if ( iscell(fwdFn) ) fwdOpts=fwdFn(2:end);  fwdFn =fwdFn{1}; end;
bwdFn =opts.bwdFn; bwdOpts={}; if ( iscell(bwdFn) ) bwdOpts=bwdFn(2:end);  bwdFn =bwdFn{1}; end;
seedFn=opts.seedFn;seedOpts={};if ( iscell(seedFn)) seedOpts=seedFn(2:end);seedFn=seedFn{1}; end;

dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
if ( ~isempty(dim) && max(dim)<ndims(X) ) % permute X to make dim the last dimension
   persistent warned
   if (isempty(warned) ) 
      warning('X has trials in other than the last dimension, permuting to make it so..');
      warned=true;
   end
	X=permute(X,[1:min(dim)-1 max(dim)+1:ndims(X) dim(:)']); dim=ndims(X)-numel(dim)+1:ndims(X);
  if ( ~isempty(opts.rdim) && opts.rdim>dim ) opts.rdim=opts.rdim-1; end; % shift other dim info
end
szX=size(X); szX(end+1:max(dim))=1; % input size (padded with 1 for extra unity dimensions)
nd=numel(szX); N=prod(szX(dim)); nf=prod(szX(setdiff(1:end,dim)));

% ensure Y has examples in last dims
szY=size(Y); szY(end+1:numel(dim)+1)=1;
if( szY(1)==N ) Y=Y'; 
elseif ( all(szY(1:numel(dim))==szX(dim)) ) %n-d Y
   Y=permute(Y,[numel(dim)+1:numel(szY) 1:numel(dim)]); % n-d Y with examples in last numel(dim) dimensions
end;
szY=size(Y); szY(end+1:numel(dim)+1)=1; 
if ( prod(szY(2:end))~=N ) error('Y should be [LxN]'); end;
binp=false;
if( size(Y,1)==1 )
  if( all(Y(:)==-1 | Y(:)==0 | Y(:)==1 | isnan(Y(:))) ) % binary problem
    binp=true;
    Y=[Y; -Y]; % convert to per-class labeling
  elseif ( all((Y(:)>=0 | isnan(Y(:))) & Y(:)==ceil(Y(:))) ) % class labels input, convert to indicator matrix
	 Yl=Y;key=unique(Y);key(key==0)=[];
	 Y=zeros(numel(key),N);for l=1:numel(key); Y(l,:)=Yl==key(l); end;	 
  end
elseif (size(Y,1)==2 && all((Y(:)>=0 & Y(:)<=1) | Y(:)==-1)) % binary indicator input
  binp=true;
end
Y(:,any(isnan(Y),1))=0; % convert NaN's to 0 so are ignored
L=size(Y,1);

% reshape X to be 2d for simplicity, though provide the orginal size to the seed function if needed
X=reshape(X,[nf N]);
Y=reshape(Y,[L N]); % same for Y

% check for degenerate inputs
if ( all(diff(Y,1)==0) ) warning('Degnerate inputs, 1 class problem'); end

% compute an example weighting if wanted
wght=opts.wght;
if ( ~isempty(opts.wght) ) % compute an example weighting
  % weight ratio between classes
  cwght=opts.wght; if ( strcmp(cwght,'bal') ) cwght=ones(size(Y,1),1); end; 
  wght=zeros(size(Y));
  for li=1:size(Y,1); wght(li,Y(li,:)>0)=cwght(li)./sum(Y(li,:)>0); end;
  wght=sum(wght,1).*size(Y,2); % example weight = sum class+example weight, preserve total weight
  Y = repop(Y,'*',wght); % example weight = sum class+example weight
end

% check if it's more efficient to sub-set the data, because of lots of ignored points
incInd=any(Y~=0,2); 
incInner=false; if( sum(incInd)/N < opts.incThresh ) incInner=true; end;

% pre-compute stuff needed for gradient computation
if( isa(X,'single') ) Y=single(Y); else Y=double(Y); end; % ensure is right data type
Y(Y<0)=0;
sY=double(sum(Y,1));
Y1=Y(1,:);
%BODGE/TODO : This is silly as if onetrue the only need to learn L-1 weight vectors.......
onetrue=all((sY==1 & sum(Y>0,1)==1) | (sY==0 & sum(Y>0,1)==0));
exInd  =find(sY==0);


% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( ~isempty(wb) )
  if ( binp )
	 W  = reshape(wb(1:end-1),nf,[]);   % weight vector [dim x L]
	 b  = wb(end-1+1:end);              % bias [L x 1]
  else
	 W  = reshape(wb(1:end-L),nf,[]);   % weight vector [dim x L]
	 b  = wb(end-L+1:end);              % bias [L x 1]
  end
else
  if ( ~binp )
	 [W,b]=feval(seedFn,X,Y,R,szX,seedOpts{:},fwdbwdOpts{:},varargin{:});
  else
    Yb = (Y1-.5)*2; Yb(:,exInd)=0;
	 [W,b]=feval(seedFn,X,(Y1-.5)*2,R,szX,seedOpts{:},fwdbwdOpts{:},varargin{:});
  end
end 

Rw   = R*W;
f    = feval(fwdFn,X,W,b,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get the example activations, [L x N]
%TODO: speed-up by only computing the loss on the non-excluded time-points?
if ( size(f,1)>1 )
  if ( 1 || opts.rescaledv ) % default to on for the seed solution
	 f  = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
  end
else
   if ( 1 || opts.rescaledv ) % default to on
      f  = max(min(f,maxf),-maxf); % clip the range of f for stability
   end
end
p    = exp(f); % =Pr(x|y+) = p1 % [L x N]
% dL_i = gradient of the loss = (1-p_y) = 1-prob of the true class = { py if y=y, 1-py otherwise i.e. 
if ( size(p,1)==1 ) % binary problem
  p  = p./(1+p);
  if ( onetrue ) % fast-path common case of single true class with equal example weights
	 dLdf = -(Y1 - p);
	 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
  else
	 dLdf = -(Y1 - p.*sY); % dLdf = Yerr % [L x N]
  end
else % multi-class
  p   = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b)); % softmax
  if ( onetrue ) % fast-path common case of single true class with equal example weights
	 dLdf = -(Y-p);  % [LxN] = dp_y, i.e. gradient of the log true class probability
	 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
  else
	 % [LxN] = dln(p_y), i.e. gradient of the log probability of the true class
	 dLdf = -(Y-repop(p,'*',sY));  	 
  end
end
[dJdw,dJdb]   = feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
dJ=[dJdw(:);dJdb(:)];

% setup the pre-condintor
wPC=opts.wPC; bPC=opts.bPC;
if ( isempty(opts.PCmethod) || any(strcmp(opts.PCmethod,{'none','zero'})) ) wPC=1; bPC=1; end;
if ( isempty(wPC) ) 
  switch lower(opts.PCmethod)
	 case {'wb','adapt'}; % use an estimate of the loss hessian around the current point
      ddLdf = p.*(1-p); 
	 case 'wb0'; % use an estimate of loss gradient very very very far away from the goal
      if ( size(p,1)==1 ) ddLdf = Y1; else ddLdf=Y; end; % get the points which contribute to loss
      ddLdf(:,any(Y,1)) = 1./size(Y,1); % set equal probability over all classes
      ddLdf = ddLdf.*(1-ddLdf); 
      if(~isempty(wght)) ddLdf =repop(ddLdf,'*',wght); end;%include pt-weight in pre-conditioner
	 case 'wbg'; % use an estimate of loss gradient near the goal
      if ( size(p,1)==1 ) ddLdf = Y1; else ddLdf=Y; end; % get the points which contribute to loss
      peps =min(.25,1./size(Y,1));
      ddLdf(ddLdf(:)>0) =(1-peps);           % true class gets most of the mass
      ddLdf(ddLdf(:)==0)=peps/(size(Y,1)-1); % other classes get what's left over equally
      ddLdf = ddLdf.*(1-ddLdf); 
      if(~isempty(wght)) ddLdf =repop(ddLdf,'*',wght); end;
    otherwise; error('unrecognised preconditioner : %s',opts.PCmethod);
  end
  ddLdf(:,exInd)=0; % ensure excluded points are ignored
  % do the backward computation to get the diag-hessian estimate
  % N.B. internally the bwd-fn does a gaus-newton approx hessian estimate so we give the
  %      sqrt of the diag-hessian for the loss to get an exact output
  [ans,ans,ddLdw,ddLdb]=feval(bwdFn,X,ddLdf,2*Rw,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
  % include the effect of the quadratic regularisor -- this cannot be done in the bwdFn
  ddLdw = ddLdw + R;

  % normalize the pre-conditioner, to equivalent norm as a identity matrix
  rsf  = sqrt(numel(ddLdw))./sqrt(ddLdw(:)'*ddLdw(:));
  ddLdw=ddLdw.*rsf; ddLdb=ddLdb.*rsf;% N.B. scale of PC doens't matter
  ddLdw(ddLdw<eps) = 1; % check for to small values
  fprintf('Hess cond: %g\n',max(ddLdw(:))./min(ddLdw(:)));
  if(max(ddLdw(:))./min(ddLdw(:)) < 10) ddLdw=1; end; % don't PC if already clustered diag values
  wPC=1./ddLdw;
  bPC=1./ddLdb;
end;
if( numel(wPC)==1 )    PC = wPC;
elseif( numel(wPC)>1 ) PC = [wPC(:);bPC];
end

MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d(:)'*dJ(:));
r2   = dtdJ;

% Ed = entropy loss = expected log-prob of the true class
% expected loss = -P(+)*ln P(D|w,b,+) -P(-)*ln(P(D|w,b,-)
if ( onetrue ) % fast-path for common case
  if ( size(p,1)==1 ) % binary problem
	 Ed  = -(sum(log(p(Y(1,:)>0)+minp))+sum(log(1-p(Y(2,:)>0)+minp))); 
  else
	 Ed  = -sum(log(p(Y(:)>0)+minp)); % Ed = sum(-log(Pr(Y_true)))
  end
else
  if ( size(p,1)==1 ) % binary problem
	 Ed  = -(log(p(Y(1,:)>0)+minp)*Y(1,Y(1,:)>0)'+log(1-p(Y(2,:)>0)+minp)*Y(2,Y(2,:)>0)'); 
  else % multi-class
	 Ed  = 0;
	 for li=1:size(p,1); % accumulate over the different possible classes
	 	Ed = Ed + -log(p(li,Y(li,:)>0)+minp)*Y(li,Y(li,:)>0)';
	 end
  end  
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
  oJ=J; oEd=Ed; oEw=Ew; oMr=Mr; or2=r2; oW=W; ob=b; % record info about prev result we need

   %---------------------------------------------------------------------
   % Secant method for the root search.
   if ( opts.verb > 2 )
      fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ed,Ew); 
      if ( opts.verb>3 ) 
         hold off;plot(0,dtdJ,'r*');hold on;text(0,double(dtdJ),num2str(0)); 
         grid on;
      end
   end;
   ostep=inf;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   odtdJ=dtdJ; % one step before is same as current
   % pre-compute for speed later
	W0  = W;       b0=b;     
	f0  = f;
   dw  = reshape(d(1:numel(W)),size(W));
	db  = reshape(d(numel(W)+(1:numel(b))),size(b));
	df  = feval(fwdFn,X,dw,db,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get the forward activations in the search direction
	
	% pre-compute regularized weight contribution to the line-search
   Rw =R*W;      dRw=dw(:)'*Rw(:);  % scalar or full matrix
   Rd =R*dw;     dRd=dw(:)'*Rd(:);	
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,Ew+Ed,Ew,Ed); 
   end
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
		f    = f0 + tstep*df;
		if ( opts.rescaledv )
         if ( size(f,1)>1 )
            f  = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
         else
            f  = min(f,maxf); % clip the range of f for stability
         end
		end
		p    = exp(f); % =Pr(x|y+) = p1 % [L x N]
		if ( size(p,1)==1 ) % binary problem
		  p  = p./(1+p);
		  if ( onetrue ) % fast-path common case
			 dLdf = -(Y1-p); % dont-forget neg sign, as loss is *negative* log entropy
			 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
		  else
			 dLdf = -(Y1-p.*sY);
		  end
		else % multi-class
		  %[L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b)); % softmax
		  p   = repop(p,'/',sum(p,1));
		  if ( onetrue ) % fast-path common case
			 dLdf = -(Y-p);  % [LxN] = dp_y, i.e. gradient of the log true class probability
			 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
		  else
          % [LxN] = dln(p_y), i.e. gradient of the log probability of the true class
			 dLdf = -(Y-repop(p,'*',sY));  	 
		  end
		end
		dtdJ = -(2*(dRw+tstep*dRd) + df(:)'*dLdf(:));

		if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		  fprintf('%d) linfnlr_cg(@345)::Numerical issues falling back on re-scaled dv\n',iter);
		  oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		  opts.rescaledv=true; continue;
		end;
		
		if ( opts.verb>2 )
		  if ( onetrue ) % fast-path for common case
			 if ( size(p,1)==1 ) % binary problem
				Ed  = -(sum(log(p(Y(1,:)>0)+minp))+sum(log(1-p(Y(2,:)>0)+minp))); 
			 else % multi-class
				Ed  = -sum(log(p(Y(:)>0)+minp)); % Ed = sum(-log(Pr(Y_true)))
			 end
		  else % example weighting
			 if ( size(p,1)==1 ) % binary problem
				Ed  = -(log(p(Y(1,:)>0)+minp)*Y(1,Y(1,:)>0)'+log(1-p(Y(2,:)>0)+minp)*Y(2,Y(2,:)>0)'); 
			 else % multi-class
				Ed  = 0;
				for li=1:size(Y,1); % accumulate over the different possible classes
	 			  Ed = Ed + -log(p(li,Y(li,:)>0)+minp)*Y(li,Y(li,:)>0)';
				end
			 end
		  end
		  Ew   = W(:)'*Rw(:) + 2*tstep*dRw + tstep*tstep*dRd;     % -ln P(w,b|R);
		  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',j,tstep,dtdJ,0,Ed+Ew,Ew,Ed); 
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
										  % Argh! this causes breakage.... why?
	%muf= mean(f(:)); if(abs(muf)>1) b=b-muf; f=f-muf; end; % help to stabilize f?

   % compute the other bits needed for CG iteration, i.e. a full gradient computation
	[dJdw,dJdb] = feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
	dJ=[dJdw(:);dJdb(:)];
   MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr(:)'*dJ(:)); 
   
   % compute the function evaluation
	if ( onetrue ) % fast-path for the common case - 1 true class , equal weight
	  if ( size(p,1)==1 ) % binary problem
		 Ed  = -(sum(log(p(Y(1,:)>0)+minp))+sum(log(1-p(Y(2,:)>0)+minp))); 
	  else % multi-class
		 Ed  = -sum(log(p(Y(:)>0)+minp)); % Ed = sum(-log(Pr(Y_true)))
	  end
	else % example weighting
	  if ( size(p,1)==1 ) % binary problem
		 Ed  = -(log(p(Y(1,:)>0)+minp)*Y(1,Y(1,:)>0)'+log(1-p(Y(2,:)>0)+minp)*Y(2,Y(2,:)>0)'); 
	  else % multi-class
		 Ed  = 0;
		 for li=1:size(Y,1); % accumulate over the different possible classes
	 		Ed = Ed + -log(p(li,Y(li,:)>0)+minp)*Y(li,Y(li,:)>0)';
		 end
	  end
	end
   Ew   = W(:)'*Rw(:);% P(w,b|R);
   J    = Ed + Ew + opts.Jconst;       % J=neg log posterior
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
   end   
   if(opts.verb>3)   % debug code      
	  [J2,dJ2,f2,obj2]=fwdbwdmlrFn([W(:);b(:)],X,Y,R,fwdFn,bwdFn,varargin{:});
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J2,obj2,r2);
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
     if ( opts.verb>=0 )
		 warning(sprintf('%d) linfnlr_cg(@490)::Line-search Non-reduction: %g > %g\n',iter,J,oJ));
	  end;
     J=oJ; Ed=oEd; Ew=oEw; W=oW; b=ob; Mr=oMr; r2=or2; %tstep=otstep*.01;
     if( ~opts.rescaledv ) 
        fprintf('%d) linfnlr_cg(@494)::Numerical issues falling back on re-scaled dv\n',iter);
        opts.rescaledv=true;
     else
        nStuck =nStuck+1; % only counts as stuck if already with re-scale turned on
     end
     f = feval(fwdFn,X,W,b,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:});
     % update the performance info for convergence testing
     if ( opts.rescaledv )
        if ( size(f,1)>1 )
           f  = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
        else
           f  = min(f,maxf); % clip the range of f for stability
        end
     end
		p    = exp(f); % =Pr(x|y+) = p1 % [L x N]
		if ( size(p,1)==1 ) % binary problem
		  p  = p./(1+p);
		else % multi-class
		  p   = repop(p,'/',sum(p,1));
		end
      if ( onetrue ) % fast-path for the common case - 1 true class , equal weight
         if ( size(p,1)==1 ) % binary problem
            Ed  = -(sum(log(p(Y(1,:)>0)+minp))+sum(log(1-p(Y(2,:)>0)+minp))); 
         else % multi-class
            Ed  = -sum(log(p(Y(:)>0)+minp)); % Ed = sum(-log(Pr(Y_true)))
         end
      else % example weighting
         if ( size(p,1)==1 ) % binary problem
            Ed  = -(log(p(Y(1,:)>0)+minp)*Y(1,Y(1,:)>0)'+log(1-p(Y(2,:)>0)+minp)*Y(2,Y(2,:)>0)'); 
         else % multi-class
            Ed  = 0;
            for li=1:size(Y,1); % accumulate over the different possible classes
               Ed = Ed + -log(p(li,Y(li,:)>0)+minp)*Y(li,Y(li,:)>0)';
            end
         end
      end
	  restart=true;
	  if ( nStuck > 1 ) 
        warning(sprintf('%d) linfnlr_cg(@501)::Line-search Non-reduction - aborted\n',iter));
        break; 
     end;
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
   dtdJ  = -d(:)'*dJ(:);         % new search dir grad.
   if( dtdJ <= 0 || restart || ...  % non-descent dir switch to steepest
	  (opts.restartInterval>0 && mod(iter,opts.restartInterval)==0))         
     if ( dtdJ<=0 && opts.verb >= 2 ) fprintf('non-descent dir\n'); end;
     d=Mr; dtdJ=-d(:)'*dJ(:);
   end; 
   
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   W=W0; b=b0;
end;

% compute the final performance with untransformed input and solutions
Rw  = R*W;       % scalar or full matrix
p   = exp(f);    % [L x N] =Pr(x|y_+) = exp(w_ix+b)./sum_y(exp(w_yx+b));
if ( size(p,1)==1 ) % binary problem
  p  = p./(1+p);
  if ( onetrue ) 
	 Ed  = -(sum(log(p(Y(1,:)>0)+minp))+sum(log(1-p(Y(2,:)>0)+minp)));
  else
	 Ed  = -(log(p(Y(1,:)>0)+minp)*Y(1,Y(1,:)>0)'+log(1-p(Y(2,:)>0)+minp)*Y(2,Y(2,:)>0)'); 
  end
else % multi-class
  p    = repop(p,'/',sum(p,1));       % [1xN] = Pr(y_true)
  if ( onetrue ) 
	 Ed  = -sum(log(p(Y(:)>0)+minp)); % Ed = sum(-log(Pr(Y_true)))
  else
	 Ed  = 0;
	 for li=1:size(Y,1); % accumulate over the different possible classes
	 	Ed = Ed + -log(p(li,Y(li,:)>0)+minp)*Y(li,Y(li,:)>0)';
	 end	 
  end
end
Ew  = W(:)'*Rw(:);     % -ln P(w,b|R);
J   = Ed + Ew + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
          iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
end

% compute final decision values.
obj = [J Ew Ed];
wb=[W(:);b(:)];
										  % check the eval
% massignin('base','X',X,'Y',Y,'R',R,'fwdFn',fwdFn,'bwdFn',bwdFn,'va',varargin,'W',W,'b',b);
%[J,dJ,f,obj]=fwdbwdmlrFn([W(:);b(:)],X,Y,R,fwdFn,bwdFn,varargin{:});
%checkgrad(@(wb) fwdbwdmlrFn(wb,X,Y,R,fwdFn,bwdFn,varargin{:}),[W(:);b(:)],1e-5,0,1);
return;

%-----------------------------------------------------------------------------
function []=testCase()
% low dimensions
[X,Yl]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 20],[.3 .3; .3 .3; .2 .2],[],[1 -1 -1]);
Y=lab2ind(Yl)';[dim,N]=size(X);L=size(Y,1)
% high dimensions
nd=100; nClass=800;
[X,Yl]=mkMultiClassTst([zeros(1,nd/2-1) -1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) 1 0 zeros(1,nd/2-1); zeros(1,nd/2-1) .2 .5 zeros(1,nd/2-1)],[nClass nClass 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);
Y=lab2ind(Yl)';[dim,N]=size(X);
% multi-class
[X,Yl]=mkMultiClassTst([-1 0;1 0;0 1;0 -1],[400 400 400 400],[.2 .2]);[dim,N]=size(X);
Y=lab2ind(Yl)';[dim,N]=size(X);

wb0=randn(size(X,1)+1,size(Y,1));
Xtrn=X; Ytrn=Y;

tic,lr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'dim',2);toc
tic,mlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'dim',2);toc
tic,[wb,f,J]=linfnlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'dim',2,'PCmethod',[]);toc
% various pre-conditioners
tic,[wb,f,J]=linfnlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'PCmethod','wb0');toc
tic,[wb,f,J]=linfnlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'PCmethod','wb');toc
tic,[wb,f,J]=linfnlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'PCmethod','wbg');toc

% with skewed feature norms so pre-conditioner is useful
Xtrn=repop(X,'*',[1e2;ones(size(X,1)-1,1)]);

										  % multi-class version
[X,Yl]=mkMultiClassTst([-1 0;1 0;0 1;0 -1],[400 400 400 400],[.2 .2]);[dim,N]=size(X);
Y=lab2ind(Yl)';
wb0=randn(size(X,1)+1,min(size(Y)));

tic,[wb,f,J] =mlr_cg(X,Y,0,'verb',1,'wb',wb0);toc
tic,[wb,f,J] =linfnlr_cg(X,Y,0,'verb',1,'wb',wb0);toc

										  % with unbalanced data + example weighting
[X,Yl]=mkMultiClassTst([-1 0;1 0],[1200 400],[.2 .2]);[dim,N]=size(X);
Y=lab2ind(Yl)';
wb0=randn(size(X,1)+1,size(Y,1));
tic,[wb,f,J]=linfnlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0);toc
tic,[wb,f,J]=linfnlr_cg(X,Y,0,'verb',1,'objTol0',1e-10,'wb',wb0,'wght','bal');toc

clf;plotLinDecisFn(X,Y,reshape(wb(1:end-min(size(Y))),size(X,1),min(size(Y))),wb(end-min(size(Y))+1:end))


% with a large 2-d structure
nd=23; nSamp=100; nClass=300;
X=randn(nd,nSamp,nClass*2);
Y=sign(randn([1,size(X,2),size(X,3)]));

taus=0:3;
W0    = randn([size(X,1),size(taus,2),size(Y,1)]);% [ d x tau x L]
W01   = permute(W0,[1 3 2]); % [ d x L x tau ]
b0    = randn(1,size(Y,1)); % [ 1 x L]

[wb,f,J]=linfnlr_cg(X,Y,1,'verb',1,'dim',[2 3],'fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd','seedFn','embedWLR_seed',taus);
