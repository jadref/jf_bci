function [wb,f,J,obj,tstep]=linfn_cg(X,Y,R,varargin);
% generic linear fwd/bwd + loss cg based solver
%
% [wb,f,J,obj]=linfn_cg(X,Y,C,varargin)
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
%  lossFn- {func_name loss_opts} or [fn-handle] function which does loss computation to get gradients ('ls_loss')
%             This should have the signature:
%              [L,dLdf]=lossFn(Y,f,loss_opts,varargin{:});
%  objLossFn - pseudoname for lossFn
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
  opts=struct('wb',[],'alphab',[],'dim',[],'ydim',[],...
              'eta',1,'h',.5,...
				  'lossFn','','objLossFn','ls_loss',...
              'fwdFn','lin_fwd','bwdFn','lin_bwd','seedFn','lin_seed','fwdbwdOpts',{{}},...
				  'mu',0,'Jconst',0,...
              ...%'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',5e-2,'objTol',1e-4,'objTol0',1e-4,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',5e-2,'objTol',0,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,...
				  'CGmethod','PR','bPC',[],'wPC',[],'restartInterval',0,...
				  'PCmethod','adaptDiag','PCalpha',exp(-log(2)/14),'PClambda',.25,'PCminiter',[10 20],...
				  'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0,'rescaledv',0);
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
% Ensure all inputs have a consistent precision
if(islogical(Y))Y=single(Y); end;
if(isa(X,'double') && ~isa(Y,'double') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; minp=1e-40; else eps=1e-16; minp=1e-120; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence
if( isempty(opts.lossFn) && ~isempty(opts.objLossFn) ) opts.lossFn=opts.objLossFn; end;

lossFn=opts.lossFn;lossOpts={};if ( iscell(lossFn)) lossOpts=lossFn(2:end); lossFn=lossFn{1}; end
fwdFn =opts.fwdFn; fwdOpts={}; if ( iscell(fwdFn) ) fwdOpts=fwdFn(2:end);   fwdFn =fwdFn{1}; end;
bwdFn =opts.bwdFn; bwdOpts={}; if ( iscell(bwdFn) ) bwdOpts=bwdFn(2:end);   bwdFn =bwdFn{1}; end;
seedFn=opts.seedFn;seedOpts={};if ( iscell(seedFn)) seedOpts=seedFn(2:end); seedFn=seedFn{1}; end;

dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
if ( ~isempty(dim) && max(dim)<ndims(X) ) % permute X to make dim the last dimension
   persistent warned
   if (isempty(warned) ) 
      warning('X has trials in other than the last dimension, permuting to make it so..');
      warned=true;
   end
	X=permute(X,[1:min(dim)-1 max(dim)+1:ndims(X) dim(:)']); dim=ndims(X)-numel(dim)+1:ndims(X);
end
szX=size(X); szX(end+1:max(dim))=1; % input size (padded with 1 for extra unity dimensions)
nd=numel(szX); N=prod(szX(dim)); nf=prod(szX(setdiff(1:end,dim)));

% ensure Y has examples in last dims
szY=size(Y); szY(end+1:numel(dim)+1)=1;
oszY=szY;
if( szY(1)==N ) Y=Y'; 
elseif ( all(szY(1:numel(dim))==szX(dim)) ) %n-d Y
   Y=permute(Y,[numel(dim)+1:numel(szY) 1:numel(dim)]);%n-d Y with examples in last numel(dim) dimension
end;
szY=size(Y); szY(end+1:numel(dim)+1)=1; 
if ( prod(szY(2:end))~=N ) error('Y should be [LxN]'); end;
L=size(Y,1);

% reshape X to be 2d for simplicity, though provide the orginal size to the seed function if needed
X=reshape(X,[nf N]);
Y=reshape(Y,[L N]); % same for Y

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
oX=X; oY=Y;

% pre-compute stuff needed for gradient computation
if( isa(X,'single') ) Y=single(Y); else Y=double(Y); end; % ensure is right data type
exInd  =find(all(Y==0,2));

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( ~isempty(wb) )
  W  = reshape(wb(1:end-L),nf,[]);   % weight vector [dim x L]
  b  = wb(end-L+1:end);              % bias [L x 1]
else
  [W,b]=feval(seedFn,X,Y,R,szX,seedOpts{:},opts.fwdbwdOpts{:},varargin{:});
end 

Rw   = R*W;
f          = feval(fwdFn,X,W,b,szX,fwdOpts{:},opts.fwdbwdOpts{:},varargin{:}); % get activations, [L x N]
[L,dLdf]   = feval(lossFn,Y,f,lossOpts{:}); % get output loss & gradient
[dJdw,dJdb]= feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},opts.fwdbwdOpts{:},varargin{:}); % backup
dJ=[dJdw(:);dJdb(:)];
PC   = 1;%./(sqrt(dHest));
MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d(:)'*dJ(:));
r2   = dtdJ;

Ed   = L;
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
   ostep=inf;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   odtdJ=dtdJ; % one step before is same as current
   % pre-compute for speed later
	W0  = W;       b0=b;     
	f0  = f;
   dw  = reshape(d(1:numel(W)),size(W));
	db  = reshape(d(numel(W)+(1:numel(b))),size(b));
	df  = feval(fwdFn,X,dw,db,szX,fwdOpts{:},opts.fwdbwdOpts{:},varargin{:}); % get the forward activations in the search direction
	
	% pre-compute regularized weight contribution to the line-search
   Rw =R*W;      dRw=dw(:)'*Rw(:);  % scalar or full matrix
   Rd =R*dw;     dRd=dw(:)'*Rd(:);	
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,J,Ew,Ed); 
 end
 % get state out for gradient checking: massignin('base','wb',[W(:);b(:)],'X',X,'Y',Y,'szX',szX,'R',R,'lossFn',lossFn,'fwdFn',fwdFn,'bwdFn',bwdFn,'opts',varargin)
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
		f    = f0 + tstep*df;
      [L,dLdf]=feval(lossFn,Y,f,lossOpts{:}); % get output loss & gradient
		dtdJ = -(2*(dRw+tstep*dRd) - df(:)'*dLdf(:));
      % gradient quality check: checkgrad(@(wb) fwdbwdlossFn(wb,X,Y,R,lossFn,fwdFn,bwdFn,varargin{:}),wb,1e-5,0,1)
      
		if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		  fprintf('%d) Numerical issues falling back on re-scaled dv\n',iter);
		  oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		  opts.rescaledv=true; continue;
		end;
		
		if ( opts.verb>2 )
		  Ed   = L;
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

   % compute the other bits needed for CG iteration, i.e. a full gradient computation
	[dJdw,dJdb] = feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},opts.fwdbwdOpts{:},varargin{:});
	dJ=[dJdw(:);dJdb(:)];
   MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr(:)'*dJ(:)); 
   
										  % compute the function evaluation
	Ed   = L;
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
		 warning(sprintf('308::%d) Line-search Non-reduction - aborted\n',iter));
	  end;
     J=oJ; W=oW; b=ob; Mr=oMr; r2=or2; %tstep=otstep*.01;
	  if( ~opts.rescaledv) 
        opts.rescaledv=true;fprintf('311::%d) Numerical issues falling back on re-scaled dv\n',iter);
     end
     f = feval(fwdFn,X,W,b,szX,fwdOpts{:},opts.fwdbwdOpts{:},varargin{:});
	  nStuck =nStuck+1;
	  restart=true;
	  if ( nStuck > 1 ) 
        warning('316::Non-reduction too many times... giving up!');
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
Ed  = feval(lossFn,Y,f,lossOpts{:}); % get output loss & gradient
Ew  = W(:)'*Rw(:);     % -ln P(w,b|R);
J   = Ed + Ew + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
          iter,neval,norm(W(:,1)),norm(W(:,min(end,2))),J,Ew,Ed,r2);
end

% compute final decision values.
if ( ~all(size(X)==size(oX)) )
  f = feval(fwdFn,oX,W,b,szX,fwdOpts{:},opts.fwdbwdOpts{:},varargin{:});
end;
obj = [J Ew Ed];
wb=[W(:);b(:)];
										  % check the eval
% massignin('base','X',X,'Y',Y,'R',R,'fwdFn',fwdFn,'bwdFn',bwdFn,'va',varargin,'W',W,'b',b);
%[J,dJ,f,obj]=fwdbwdmlrFn([W(:);b(:)],X,Y,R,fwdFn,bwdFn,varargin{:});
%checkgrad(@(wb) fwdbwdmlrFn(wb,X,Y,R,fwdFn,bwdFn,varargin{:}),[W(:);b(:)],1e-5,0,1);
return;

%-----------------------------------------------------------------------------
function []=testCase()
										  % simple regression problem
nd=4; nEp=1000; nSamp=100;
M = randn(nd,1);
Y = repmat(1:nSamp,size(M,2),1,nEp);
Xt= reshape(M*Y(:,:),[size(M,1),size(Y,2),size(Y,3)]);
noise = randn(size(Xt))*(4.^2);%((randn(size(Xt))*4).^2).*sign(randn(size(Xt)));%
X = Xt + noise; % regression + heavytailed noise
clf;plot([Y(:,:,1);Xt(:,:,1)]');


% normal LS
w=X(:,:)'\Y(:,:)';
Yest=reshape(w'*X(:,:),size(Y));
est2corr(Y(:,:)',Yest)

wb0=randn(size(X,1)+1,size(Y,1));

[wb,f,J]=linfn_cg(X(:,:),Y(:,:),1);
est2corr(Y(:,:)',f)
clf;plot(Y,f,'*','markersize',10,'linewidth',2);hold on;plot([min(Y) max(Y)],[min(Y) max(Y)],'k-','linewidth',4);

% gradient check
[J,dJ,f]=fwdbwdlossFn(wb,X,Y,1,'ls_loss','embedWLR_fwd','embedWLR_bwd',taus);
checkgrad(@(wb) fwdbwdlossFn(wb,X,Y,0,'ls_loss','embedWLR_fwd','embedWLR_bwd',taus),wb,1e-5,0,1);


										  % with complex fwd/bwd functions
taus=0:3; % 0:3;%0;
[wb,f,J]=linfn_cg(X,Y,1,'verb',1,'dim',[2 3],'lossFn','ls_loss','fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd','seedFn','embedWLR_seed',taus);
W=reshape(wb(1:end-size(Y,1)),size(X,1),numel(taus));
est2corr(Y(:,:)',f)
clf;plot(reshape(wb(1:end-1),size(X,1),numel(taus))','linewidth',2)

% with multiple targets to fit at once
ndy= 3;
nY = repmat(Y,[ndy,1,1]);
[wb,f,J]=linfn_cg(X,nY,1,'verb',1,'dim',[2 3],'lossFn','ls_loss','fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd','seedFn','embedWLR_seed',taus);
est2corr(nY(:,:)',f)

% with some invalid points
fIdx=-ones(size(Y,3),1); fIdx(1:10)=1;
Ytrn=Y; Ytrn(:,:,fIdx<0)=NaN; 
Ytst=Y; Ytst(:,:,fIdx>0)=NaN;
[wb,f,J]=linfn_cg(X,Ytrn,1,'verb',1,'dim',[2 3],'lossFn','ls_loss','fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd','seedFn','embedWLR_seed',taus);
est2corr(Ytrn,f)

										  % time shifted problem
nd=4; nEp=1000; nSamp=100;
M = randn(nd,1);
irf= [0 0 1];% randn(4,1);%
Y  = randn(1,nSamp,nEp);
Xt = filter(irf,1,Y,[],2);
Xt= reshape(M*Xt(:,:),[size(M,1),size(Y,2),size(Y,3)]);
noise = randn(size(Xt))*1e-0;%((randn(size(Xt))*4).^2).*sign(randn(size(Xt)));%
X  = Xt + noise; % regression + heavytailed noise
clf;plot([Y(:,:,1);X(:,:,1)]');
