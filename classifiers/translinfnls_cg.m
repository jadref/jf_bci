function [wb,f,J,obj,tstep]=translinfnls_cg(X,Y,R,varargin);
% Regularised quadratic classifier
%
% [wb,f,J,obj]=lr_cg(X,Y,C,varargin)
%
% J = w' R w + w' mu + sum_i sum_y Pr(y_i=y) log( Pr(x_i|y) )
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
%
% TODO: 
%   [X] Local + Global regularisors.  
%   [X] Correct pre-conditioner computation
%   [X] Correct seed computaton
%   [] Don't reshape X or Y
%   [] Fix effective over-scale of R when cut into smaller sub-pieces?

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
if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
% Ensure all inputs have a consistent precision
if(islogical(Y))Y=single(Y); end;
if(isa(X,'double') && ~isa(Y,'double') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; minp=1e-40; else eps=1e-16; minp=1e-120; end;
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

% get the sub-problem indicators, and hence the number of data-sets (nDs) and sets of weight vectors (nDsw)
% TODO: pre-cut X,Y into cell-arrays of features?
N2w=opts.N2w; nDs=1; nDsw=1; if ( ~isempty(N2w) ) nDs=max(N2w); nDsw=nDs+1; end

% pre-compute stuff needed for gradient computation
if( isa(X,'single') ) Y=single(Y); else Y=double(Y); end; % ensure is right data type

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( ~isempty(wb) )
   W  = reshape(wb(1:end-(L*nDsw)),nf,[]);   % weight vector [dim x L x nDs]
   b  = wb(end-(L*nDsw)+1:end);              % bias [1 x L x nDs], N.B. bias per-class-per-dataset
else
   % TODO: Make this more efficient? pre-allocate and insert?
   W={}; b={}; Xi=X; Yi=Y;
   for dsi=1:nDs;      
      if ( ~isempty(N2w) ) 
         idx   = (N2w==dsi);
         Xi    = reshape(X,[],szX(end)); Xi=Xi(:,idx); Xi=reshape(Xi,size(X,1),[]);
         Yi    = reshape(Y,[],szY(end)); Yi=Yi(:,idx); Yi=reshape(Yi,size(Y,1),[]);
      end
      [Wi,bi]=feval(seedFn,Xi,Yi,R,szX,seedOpts{:},fwdbwdOpts{:},varargin{:});
      W{dsi}=Wi; b{dsi}=bi;
  end
  W=cat(ndims(W{1})+1,W{:}); b=cat(3,b{:});
  if( nDs>1 ) % final grand average seed
     % grand-average seed is average of all, correct individual such that sum remains constant
     Wga = sum(W,ndims(W))./nDs; W = repop(W,'-',Wga);  W=cat(ndims(W),W,Wga);
     bga = sum(b,3)./nDs;        b = repop(b,'-',bga);  b=cat(3,       b,bga);
     if ( numel(R)>1 ) % re-scale individual and grand-average solutions by the reg-strength
        sf = (R(end)+1e-3)./(R+1e-3);  % N.B. divide by 0 safe
        if( numel(R)==2 ) sf=[repmat(sf(1),1,nDs) sf(2)];end;
        sf=sf./norm(sf);
        sf(isnan(sf(:))|isinf(sf(:)))=1; % guard extreem values
        W = repop(W,'*',shiftdim(sf(:),-(ndims(W)-1)));
        b = repop(b,'*',shiftdim(sf(:),-2));
     end
  end
end 
if ( ndims(W)<3 ) % ensure W has the data-sets in the last dim
   W  =reshape(W,[nf size(W,2)/(nDsw) nDsw]);
   b  =reshape(b,[1      szY(1)     size(W,3)]);
elseif ( ndims(W)>3 ) % ensure W is 3d...
   szW=size(W);
   W  =reshape(W,[nf prod(szW(2:end-1)) szW(end)]); % [ d x (tauxL) x nDs]
   b  =reshape(b,[1  szY(1) szW(end)]);
end

if(numel(R)==1) 
   Rw   = R*W;
elseif( numel(R)==2 )
   Rw   = cat(3,R(1)*W(:,:,1:end-1),R(2)*W(:,:,end));
elseif( numel(R)==nDsw )
   Rw   = repop(shiftdim(R(:),-2),'*',W);
end
% forward computation for each data-set independently
if( isempty(N2w) ) 
   f    = feval(fwdFn,X,W,b,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get example activations, [L x N]
else
   f    = zeros([szY(1),prod(szY(2:end-1)),szY(end)]); % [ L x other-bits x blockDim ] - easy to index in block bits
   for dsi=1:nDs;
      idx = (N2w==dsi);
      Wi  = W(:,:,dsi)+W(:,:,end); % include the grand-average classifier
      bi  = b(:,:,dsi)+b(:,:,end);
      Xi  = reshape(X,[],szX(end)); Xi=Xi(:,idx); Xi=reshape(Xi,size(X,1),[]); szXi=szX; szXi(end)=sum(idx);
      fi = feval(fwdFn,Xi,Wi,bi,szXi,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get example activations, [L x N]
      f(:,:,idx)=reshape(fi,[szY(1),prod(szY(2:end-1)),sum(idx)]);
   end
   f    = f(:,:); % map back to 2-d
end

err  = Y-f; 
err(isnan(Y))=0; % ensure ignored points don't count
dLdf = -2*err;

% backward function computation
if( isempty(N2w) ) % single-dataset
   [dJdw,dJdb]= feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
else % multiple-datasets version
   dJdw = zeros(size(W)); dJdb=zeros(size(b));
   for dsi=1:nDs;
      idx = (N2w==dsi);
      Wi  = W(:,:,dsi);
      bi  = b(:,:,dsi);
      Xi  = reshape(X,[],szX(end));    Xi=Xi(:,idx);      Xi=reshape(Xi,size(X,1),[]);       szXi=szX; szXi(end)=sum(idx);
      dLdfi=reshape(dLdf,[],szY(end)); dLdfi=dLdfi(:,idx); dLdfi=reshape(dLdfi,size(dLdf,1),[]);
      [dJdwdsi,dJdbdsi]= feval(bwdFn,Xi,dLdfi,0,Wi,bi,szXi,bwdOpts{:},fwdbwdOpts{:},varargin{:});
      dJdw(:,:,dsi) = reshape(dJdwdsi,nf,[]);
      dJdb(:,:,dsi) = reshape(dJdbdsi,1,[]);
   end
   if( nDs>1 ) % now for the final all-datasets weights
      dJdw(:,:,nDsw) = sum(dJdw,3);
      dJdb(:,:,nDsw) = sum(dJdb,3);
   end
   dJdw  = 2*Rw + dJdw; % include the effect of the regularisor
end
dJ=[dJdw(:);dJdb(:)];

% setup the pre-condintor
wPC=opts.wPC; bPC=opts.bPC;
if ( isempty(opts.PCmethod) || any(strcmp(opts.PCmethod,{'none','zero'})) ) wPC=1; bPC=1; end;
if ( isempty(wPC) ) 
  ddLdf=2*ones(size(dLdf));
  ddLdf(isnan(Y))=0; % ensure excluded points are ignored
  if(~isempty(wght)) ddLdf =repop(ddLdf,'*',wght); end;%include pt-weight in pre-conditioner
  % do the backward computation to get the diag-hessian estimate
  % N.B. internally the bwd-fn does a gaus-newton approx hessian estimate so we give the
  %      sqrt of the diag-hessian for the loss to get an exact output
  if( isempty(N2w) ) % single-dataset
     [ans,ans,ddLdw,ddLdb]=feval(bwdFn,X,ddLdf,0,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
     ddLdw = ddLdw + R(min(end,2)); % include effect of quad-reg, can't be done in bwdfn
  else % multiple-datasets version
     ddLdw = zeros(size(W)); ddLdb=zeros(size(b));
     for dsi=1:nDs;
        idx = (N2w==dsi);
        Wi  = W(:,:,dsi);
        bi  = b(:,:,dsi);
        Xi  = reshape(X,[],szX(end));    Xi=Xi(:,idx);      Xi=reshape(Xi,size(X,1),[]);       szXi=szX; szXi(end)=sum(idx);
        ddLdfi=reshape(ddLdf,[],szY(end)); ddLdfi=ddLdfi(:,idx); ddLdfi=reshape(ddLdfi,size(ddLdf,1),[]);
        [ans,ans,ddLdw(:,:,dsi),ddLdb(:,:,dsi)]= feval(bwdFn,Xi,ddLdfi,0,Wi,bi,szXi,bwdOpts{:},fwdbwdOpts{:},varargin{:});
     end
     if( nDs>1 ) % now for the final all-datasets weights
        ddLdw(:,:,nDsw) = sum(ddLdw,3);
        ddLdb(:,:,nDsw) = sum(ddLdb,3);
     end
     % include the effect of the quadratic regularisor -- this cannot be done in the bwdFn
     if(numel(R)==1) 
        ddLdw              = R + dJdw; 
     elseif( numel(R)==2 )
        ddLdw(:,:,1:end-1) = ddLdw(:,:,1:end-1)+R(1); 
        ddLdw(:,:,end)     = ddLdw(:,:,end)    +R(2);
     elseif( numel(R)==nDsw )
        ddLdw              = repop(ddLdw,'+',shiftdim(R(:),-2));
     end
  end

  % normalize the pre-conditioner, to equivalent norm as a identity matrix
  rsf  = sqrt(numel(ddLdw))./sqrt(ddLdw(:)'*ddLdw(:));
  ddLdw=ddLdw.*rsf; ddLdb=ddLdb.*rsf;% N.B. scale of PC doens't matter
  ddLdw(ddLdw<eps) = 1; % check for to small values
  ddLdb(ddLdb<eps) = 1;
  fprintf('Hess cond: %g\n',max(ddLdw(:))./min(ddLdw(:)));
  if(max(ddLdw(:))./min(ddLdw(:)) < 10) ddLdw=1; end; % don't PC if already clustered diag values
  wPC=1./ddLdw;
  bPC=1./ddLdb;
end;
if( numel(wPC)==1 )    PC = wPC;
elseif( numel(wPC)>1 ) PC = [wPC(:);bPC(:)];
end

MdJ  = PC.*dJ; % pre-conditioned gradient
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d(:)'*dJ(:));
r2   = dtdJ;

Ed   = err(:)'*err(:);
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
   if ( isempty(N2w) ) 
      df = feval(fwdFn,X,dw,db,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get the activations, [L x N]
   else
      df = zeros([szY(1),prod(szY(2:end-1)) szY(end)]);
      for dsi=1:nDs;
         idx = (N2w==dsi);
         dwi = dw(:,:,dsi)+dw(:,:,end);
         dbi = db(:,:,dsi)+db(:,:,end);
         Xi  = reshape(X,[],szX(end)); Xi=Xi(:,idx); Xi=reshape(Xi,size(X,1),[]); szXi=szX; szXi(end)=sum(idx);
         dfi = feval(fwdFn,Xi,dwi,dbi,szXi,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get activations, [L x N]
         df(:,:,idx)=reshape(dfi,[szY(1),prod(szY(2:end-1)),sum(idx)]);
      end
      df = df(:,:); % map back to 2d
	end
   df(isnan(Y))=0; % ensure ignored points don't count      

	% pre-compute regularized weight contribution to the line-search
   if(numel(R)==1) 
      Rw = R*W;
      Rd = R*dw;     
   elseif( numel(R)==2 )
      Rw = cat(3,R(1)*W(:,:,1:end-1),R(2)*W(:,:,end));
      Rd = cat(3,R(1)*dw(:,:,1:end-1),R(2)*dw(:,:,end));     
   elseif( numel(R)==nDsw )
      Rw = repop(shiftdim(R(:),-2),'*',W);
      Rd = repop(shiftdim(R(:),-2),'*',dw);     
   end

   dRw=dw(:)'*Rw(:);  % scalar or full matrix
   dRd=dw(:)'*Rd(:);	
   % pre-compute and cache the linear/quadratic terms
   % N.B. L(Y-f) -> dLdf = -2*(y-f) -> dLdf|(f+t*df) = -2*(y-(f+t*df)) = -2*(y-f) + 2*t*df
   %      thus: df'*dLdf = df'*(-2*err + 2*t*df) = -2*df'*err + 2*t*df'*df = -2*derr0 + 2*t*dfdf
   err0 = err;
   derr0= df(:)'*err0(:);
   dfdf = df(:)'*df(:);

   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	if ( 0 || opts.verb>2 )
	  fprintf('.%2da stp=%8f-> dtdJ=%8f/%8f @ J=%8f (%8f+%8f)\n',0,0,dtdJ,0,Ew+Ed,Ew,Ed); 
   end
   for j=1:opts.maxLineSrch;
      neval =neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      dfdLdf= -2*(derr0 - tstep*dfdf);
		dtdJ  = -(2*(dRw+tstep*dRd) + dfdLdf);

		if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		  fprintf('%d) linfnlr_cg(@345)::Numerical issues falling back on re-scaled dv\n',iter);
		  oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		  opts.rescaledv=true; continue;
		end;
		
		if ( opts.verb>2 )
		  Ed   = err(:)'*err(:);
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
   % update error and it's weighted version
   err = err0 - tstep*df; % L0=(y-f), L'=Y-(f+t*df)=Y-f-t*df = L0-t*df
   err(isnan(Y))=0; % ensure ignored points don't count  % TODO: probably don't need this...
   dLdf= -2*err;

   % compute the other bits needed for CG iteration, i.e. a full gradient computation
   % backward function computation
   if( isempty(N2w) ) % single-dataset
      [dJdw,dJdb]= feval(bwdFn,X,dLdf,2*Rw,W,b,szX,bwdOpts{:},fwdbwdOpts{:},varargin{:});
   else % multiple-datasets version
      dJdw = zeros(size(W)); dJdb=zeros(size(b));
      for dsi=1:nDs;
         idx   = (N2w==dsi);
         Wi  = W(:,:,dsi);
         bi  = b(:,:,dsi);
         Xi  = reshape(X,[],szX(end));    Xi=Xi(:,idx);       Xi=reshape(Xi,size(X,1),[]);       szXi=szX; szXi(end)=sum(idx);
         dLdfi=reshape(dLdf,[],szY(end)); dLdfi=dLdfi(:,idx); dLdfi=reshape(dLdfi,size(dLdf,1),[]);
         [dJdwdsi,dJdbdsi]= feval(bwdFn,Xi,dLdfi,0,Wi,bi,szXi,bwdOpts{:},fwdbwdOpts{:},varargin{:});
         dJdw(:,:,dsi) = reshape(dJdwdsi,nf,[]);
         dJdb(:,:,dsi) = reshape(dJdbdsi,1,[]);
      end
      if( nDs>1 ) % now for the final all-datasets weights
         dJdw(:,:,nDsw) = sum(dJdw,3);
         dJdb(:,:,nDsw) = sum(dJdb,3);
      end
      dJdw  = 2*Rw + dJdw; % include the effect of the regularisor
   end
	dJ=[dJdw(:);dJdb(:)];
   MdJ= PC.*dJ; % pre-conditioned gradient
   Mr =-MdJ;
   r2 =abs(Mr(:)'*dJ(:)); 
   
   % compute the function evaluation
	Ed   = err(:)'*err(:);
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
     nStuck =nStuck+1; % only counts as stuck if already with re-scale turned on
     f  = zeros(szY(1),prod(szY(2:end-1)),prod(szY(end)));
     for dsi=1:size(W,3)-1;
        idx = (N2w==dsi);
        Wi  = W(:,:,dsi)+W(:,:,end);
        bi  = b(:,:,dsi)+b(:,:,end);
        Xi  = reshape(X,[],szX(end)); Xi=Xi(:,idx); Xi=reshape(Xi,size(X,1),[]); szXi=szX; szXi(end)=sum(idx);
        fi  = feval(fwdFn,Xi,Wi,bi,szXi,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % example activations, [L x N]
        f(:,:,idx)=reshape(fi,[szY(1),prod(szY(2:end-1)),sum(idx)]);
     end
     f = f(:,:); % map back to 2d
     err= Y-f; err(isnan(Y(:)))=0; 
	  nStuck =nStuck+1;
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
if(numel(R)==1) 
   Rw   = R*W;
elseif( numel(R)==2 )
   Rw   = cat(3,R(1)*W(:,:,1:end-1),R(2)*W(:,:,end));
elseif( numel(R)==nDsw )
   Rw   = repop(shiftdim(R(:),-2),'*',W);
end
% forward computation for each data-set independently
if( isempty(N2w) ) 
   f    = feval(fwdFn,X,W,b,szX,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get example activations, [L x N]
else
   f    = zeros([szY(1),prod(szY(2:end-1)),szY(end)]); % [ L x other-bits x blockDim ] - easy to index in block bits
   for dsi=1:nDs;
      idx = (N2w==dsi);
      Wi  = W(:,:,dsi)+W(:,:,end); % include the grand-average classifier
      bi  = b(:,:,dsi)+b(:,:,end);
      Xi  = reshape(X,[],szX(end)); Xi=Xi(:,idx); Xi=reshape(Xi,size(X,1),[]); szXi=szX; szXi(end)=sum(idx);
      fi = feval(fwdFn,Xi,Wi,bi,szXi,fwdOpts{:},fwdbwdOpts{:},varargin{:}); % get example activations, [L x N]
      f(:,:,idx)=reshape(fi,[szY(1),prod(szY(2:end-1)),sum(idx)]);
   end
   f    = f(:,:); % map back to 2-d
end
err = Y-f;
err(isnan(Y))=0;
Ed  = err(:)'*err(:);
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

nDs=2;
N2w=ceil(rand(size(X,2),1)*nDs);
wb0=randn(size(X,1)+1,size(Y,1),nDs+1);
Xtrn=X; Ytrn=Y;

tic,lr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'dim',2);toc
tic,mlr_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'dim',2);toc
tic,[wb,f,J]=linfnls_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0(:,:,1),'dim',2,'PCmethod',[]);toc
% single dataset version
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0(:,:,1),'N2w',[],'dim',2,'PCmethod',[]);toc
% multiple datasets version
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,0,'verb',1,'objTol0',1e-10,'wb',wb0,'N2w',N2w,'dim',2,'PCmethod',[]);toc

% validate the solution optimality
checkgrad(@(wb) fwdbwdmlrFn(wb,Xtrn,Ytrn,0,'translin_fwd','translin_bwd',N2w),wb,1e-5,1,1);


% with a more complex forward/backward function
nd=23; nSamp=100; nClass=300;
X=randn(nd,nSamp,nClass*2);
Y=sign(randn([1,size(X,2),size(X,3)]));

nDs=2;
N2w=ceil(rand(size(X,3),1)*nDs);

taus=0:3;
W0    = randn([size(X,1),size(taus,2),size(Y,1),nDs+1]);% [ d x tau x L x nDs+1]
b0    = randn([1,size(Y,1),nDs+1]); % [ 1 x L x nDs+1]
Xtrn  = X;
Ytrn  = Y;

% normal
tic,[wb,f,J] =linfnlr_cg(Xtrn,Ytrn,0,'verb',1,'wb',[vec(W0(:,:,:,1));vec(b0(:,:,1))],'dim',[2 3],'PCmethod',[],'fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd',taus);toc
% single-dataset
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,0,'verb',1,'wb',[vec(W0(:,:,:,1));vec(b0(:,:,1))],'N2w',[],'dim',[2 3],'PCmethod',[],'fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd',taus);toc
% multi-datasets
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,0,'verb',1,'wb',[W0(:);b0(:)],'N2w',N2w,'dim',[2 3],'PCmethod',[],'fwdFn','embedWLR_fwd','bwdFn','embedWLR_bwd',taus);toc


% true-transfer learning examples % Make sub-problems (all identical) with bias shifts
nDs=10; 
bs=rand(nDs,1)*10; 
N2w=[];
for i=1:numel(bs);
  [Xs{i},Ys{i}]=mkMultiClassTst([-1+bs(i) 0;1 0;.2 .5],[40 40 10],[.3 .3;.3 .3;.2 .2],[],[1 -1 -1]);
  N2w = [N2w; repmat(i,size(Xs{i},2),1)];
end
X=cat(2,Xs{:});
Y=cat(1,Ys{:})';

wb0=randn(size(X,1)+1,size(Y,1),nDs+1);
Xtrn=X; Ytrn=Y;

% Add a rotation to the sub-probs
thetas=(1:nDs)*pi/2;
for i=1:nDs;
  M = [cos(thetas(i)) -sin(thetas(i));sin(thetas(i)) cos(thetas(i))];
  Xsr{i}=M*Xs{i};
end
Xtrn=cat(2,Xsr{:});

C=10;
% single-dataset normal call
tic,[wb,f,J]=linfnls_cg(Xtrn,Ytrn,C,'verb',1,'objTol0',1e-10,'wb',wb0(:,:,1),'dim',2);toc % global soln
% single dataset version
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,C,'verb',1,'objTol0',1e-10,'wb',wb0(:,:,1),'N2w',[],'dim',2);toc % global soln
% multiple datasets version
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,C,'verb',1,'objTol0',1e-10,'wb',wb0,'N2w',N2w,'dim',2);toc % global+local
% multiple datasets with diff global and local reg
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,[100000 C],'verb',1,'objTol0',1e-10,'wb',wb0,'N2w',N2w,'dim',2);toc% -> global +b-adapt
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,[C 100000],'verb',1,'objTol0',1e-10,'wb',wb0,'N2w',N2w,'dim',2);toc% -> local
% above with auto-seed
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,[100000 C],'verb',1,'objTol0',1e-10,'N2w',N2w,'dim',2);toc % -> global soln + bias-adapt
tic,[wb,f,J]=translinfnls_cg(Xtrn,Ytrn,[C 100000],'verb',1,'objTol0',1e-10,'N2w',N2w,'dim',2);toc % -> local soln


dv2loss(Ytrn,f)
W=reshape(wb(1:end-(size(Y,1)*(nDs+1))),size(X,1),[]); 
b=reshape(wb(end-(size(Y,1)*(nDs+1))+1:end),size(Y,1),nDs+1);
[bs (b(1:numel(bs))+b(end))'] % check if the learned b is any good