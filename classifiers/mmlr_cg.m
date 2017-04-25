function [wb,f,J,obj,tstep]=mmlr_cg(X,Y,R,varargin);
% Regularised max-margin multiple linear Logistic Regression Classifier
%
% [wb,f,J,obj]=lr_cg(X,Y,C,varargin)
% Regularised Multiple-Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J =   \sum_l ( w_l' R w_l ) + \sum_l ln ( Pr(y_i | x_i, w_1,w_2,...,w_L) )
%   =   \sum_l ( w_l' R w_l ) + \sum_l ( w_l'*mu_l ) + \sum_i ( softmax(y_i,w_1,w_2,...,w_L) )
%   =   w' R w  + w' mu + \sum_i log( exp( w_(y_i)'*X_i + b_(y_i) ) / sum_y (exp( w_(y_i)'*X_i + b_(y_i) )) ) 
%
% Inputs:
%  X       - [d1 x d2 x ... x L x N] data matrix with examples in the *last* dimension
%  Y       - [LxN] matrix of -1/0/+1 labels for L classes, (N.B. 0 label pts are implicitly ignored)
%  R       - quadratic regularisation matrix                                   (0)
%     [1x1]       -- simple regularisation constant             R(w)=w'*R*w
%     [d1xd2x...xd1xd2x...] -- full matrix                      R(w)=w'*R*w
%     [d1xd2x...] -- simple weighting of each component matrix, R(w)=w'*diag(R)*w
%     [d1 x d1]   -- 2D input, each col has same full regMx     R(w)=trace(W'*R*W)=sum_c W(:,c)'*R*W(:,c)
%     [d2 x d2]   -- 2D input, each row has same full regMX     R(w)=trace(W*R*W')=sum_r W(r,:) *R*W(r,:)'
%     N.B. if R is scalar then it corrospends to roughly max allowed length of the weight vector
%          good default is: .1*var(data)
% Outputs:
%  wb      - [size(X,1:end-2)+1 1] matrix of the feature weights and the bias {W;b}
%  f       - [NxL] vector of decision values
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
%  objTol0 - [float] relative objective gradient tolerance                (1e-5)
%  objTol  - [float] absolute objective gradient tolerance                (0)
%  tol0    - [float] relative gradient tolerance, w.r.t. initial value    (0)
%  lstol0  - [float] line-search relative gradient tolerance, w.r.t. initial value   (1e-2)
%  tol     - [float] absolute gradient tolerance                          (0)
%  verb    - [int] verbosity                                              (0)
%  step    - initial step size guess                                      (1)
%  wght    - point weights [Nx1] vector of label accuracy probabilities   ([])
%            [2x1] for per class weightings (neg class,pos class)
%            [1x1] relative weight of the positive class
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied.
if ( nargin < 3 ) R(1)=0; end;
if( numel(varargin)==1 && isstruct(varargin{1}) ) % shortcut eval option procesing
  opts=varargin{1};
else
  opts=struct('wb',[],'alphab',[],'dim',[],'labdim',[],'ydim',[],'mu',0,'Jconst',0,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-5,'objTol',1e-5,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'rescaledv',0,...
				  'bPC',[],'wPC',[],'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0);
  opts=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
% Ensure all inputs have a consistent precision
if(isa(X,'double') && isa(Y,'single') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence

dim=opts.dim; if ( isempty(dim) ) dim=ndims(X); end;
labdim=opts.labdim; if ( isempty(labdim) ) labdim=ndims(X)-1; end;
szX=size(X); nd=numel(szX); N=szX(dim); nf=prod(szX(setdiff(1:end,[labdim dim])));
if( size(Y,1)==N && size(Y,2)~=N ) Y=Y'; end; % ensure Y has examples in last dim
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
  if ( size(X,labdim)>L ) % have repeated stuff, convert to extra dimension
	  nStim = szX(labdim)/L;
	  X=reshape(X,[szX(1:labdim-1) L nStim szX(labdim+1:end)]);
	  szX=size(X);
	  nf =nf*nStim;
	  dim(dim>labdim)=dim(dim>labdim)+1;
  end
  X=permute(X,[setdiff(1:ndims(X),[labdim dim]) labdim dim]); 
end;

X=reshape(X,[nf L N]);
mu=opts.mu; if ( numel(mu)>1 ) mu=reshape(mu,nf,[]); else mu=0; end;

% check for degenerate inputs
if ( all(diff(Y,1)==0) ) warning('Degnerate inputs, 1 class problem'); end

% check if it's more efficient to sub-set the data, because of lots of ignored points
oX=X; oY=Y;
incInd=any(Y~=0,1);  exInd=find(~incInd);
if ( sum(incInd)./size(Y,2) < opts.incThresh ) % if enough ignored to be worth it
   if ( sum(incInd)==0 ) error('Empty training set!'); end;
   X=X(:,:,incInd); Y=Y(:,incInd);
	exInd=[];
end

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  % TODO: Develop a better seed solution, e.g. prototype classifier?
  b=zeros(1,1); w=zeros(nf,1);
else
  if ( size(wb,1)==numel(wb) ) wb=reshape(wb,nf+1,1); end;
  w=wb(1:end-1,:); b=wb(end,:);
end 

% build index expression so can quickly get the predictions on the true labels
wghtYnEx= ones(size(Y,2),1); wghtYnEx(exInd)=[];
Yind = Y>0;
Yidx = int32(find(Y>0))'; 
onetrue = all(sum(Yind)==1 | sum(Yind)==0);

Rw   = R*w;
wRw  = w(:)'*Rw(:);
f    = tprod(X,[-1 1 2],w,-1)+b; % [L x N]
dv   = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
p    = exp(dv);
p    = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
if ( onetrue ) 
  g    = p(Yidx); % [1xN] = Pr(y_true)
  dLdf = Yind-p;  % [LxN] = dp_y, i.e. gradient of the probability of the true class
else
  pt   = p.*Yind;  % [LxN] = Pr(all y_true), i.e. only the labels which could be true
  g    = sum(pt,1);  % [1xN] = Pr(y_true)
  dLdf = repop(pt,'./',g)-p;  %[LxN] = dln(p_y), i.e. gradient of the log probability of the true class
end
dLdf(:,exInd)=0; % ensure ignored points aren't included

% set the pre-conditioner
% N.B. the Hessian for this problem is:
%  H  =[X*diag(wght)*X'+2*C(1)*R  (X*wght');...
%       (X*wght')'                sum(wght)];
% where wght=p(y_i).*(1-p(y_j)) where 0<p<1
% So: diag(H) = [sum(X.*(wght.*X),2) + 2*diag(R);sum(wght)];
% Now, the max value of wght=.25 = .5*(1-.5) and min is 0 = 1*(1-1)
% So approx assuming average wght(:)=.25/2; (i.e. about 1/2 points are on the margin)
%     diag(H) = [sum(X.*X,2)*.25/2+2*diag(R);N*.25/2];
wPC=opts.wPC; bPC=opts.bPC;
if ( isempty(wPC) ) 
  wPC=tprod(X(:,:),[1 -2],[],[1 -2])*.25/2; % H=X'*diag(wght)*X -> diag(H) = wght.*sum(X.^2,2) ~= sum(X.^2,2)
  % include the effect of the regularisor
  wPC=wPC+2*diag(R);
  wPC(wPC<eps) = 1; 
  wPC=1./wPC;
end;
if ( isempty(bPC) ) % Q: Why this pre-conditioner for the bias term?
  %bPC=sqrt((X(:)'*X(:)))/numel(X);%1;%size(X,2);%%*max(.1,mean(wght))); 
  bPC=1./(size(X,2)*.25/2);
end % ==mean(diag(cov(X)))
if ( numel(wPC)==1 ) % scalar pre-condn
  if ( isequal(wPC,1) ) PC = []; 
  else                  PC = [ones(size(X,1),1)*wPC;bPC];
  end
elseif ( max(size(wPC))==numel(wPC) ) % vector pre-condn
  PC = [wPC(:);bPC];
else
  PC = [];
end

dJ = [2*Rw + mu - X(:,:)*dLdf(:); ...%[nf x L]
		-sum(dLdf(:))];                %[ 1 x L]
% precond'd gradient:
%  [H  0  ]^-1 [ Rw+mu-X'((1-g).Y))] 
%  [0  bPC]    [      -1'((1-g).Y))] 
MdJ = dJ;    if ( ~isempty(PC) ) MdJ  = repop(PC,'*',dJ); end; 
Mr  =-MdJ;
d   = Mr;
dtdJ=-(d(:)'*dJ(:));
r2  = dtdJ;

Ed  = -sum(log(g));    % -ln P(D|w,b,fp)
Ew  = w(:)'*Rw(:);     % -ln P(w,b|R);
if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
J   = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
%if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\n';
if(opts.verb>0)   % debug code      
   if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
   fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,w(1),w(min(end,2)),J,Ew,Ed,r2);
end

% pre-cond non-lin CG iteration
J0=J; r02=r2;
madJ=abs(J); % init-grad est is init val
w0=w; b0=b;
for iter=1:min(opts.maxIter,2e6);  % stop some matlab versions complaining about index too big

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
   ostep=inf;step=tstep/2;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   odtdJ=dtdJ; % one step before is same as current
   % pre-compute for speed later
   f0  = f;
   dw  = d(1:end-1,:); db=d(end,:);
   df  = tprod(X,[-1 1 2],dw,[-1])+db; % [L x N]
   if( ~isequal(mu,0) ) dmu = dw'*mu; else dmu=0; end;
   Rw=R*w;       dRw=dw(:)'*Rw(:);  % scalar or full matrix
   Rd=R*dw;      dRd=dw(:)'*Rd(:);
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
	tstep=step;
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      f    = f0 + tstep*df;
		dv   = f; if(opts.rescaledv) dv=repop(f,'-',max(f,[],1)); end;% re-scale for numerical stability
		p    = exp(dv);
		p    = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
		if ( onetrue ) 
		  % g    = p(Yidx); % [1xN] = Pr(y_true)
		  dLdf = Yind-p;  % [LxN] = dp_y, i.e. gradient of the log probability of the true class
		else
		  pt   = p.*Yind;  % [LxN] = Pr(all y_true), i.e. only the labels which could be true
		  %g    = sum(pt,1);  % [1xN] = Pr(y_true)
		  dLdf = repop(pt,'./',sum(pt,1))-p;%[LxN]= dln(p_y), i.e. grad log probability of the true class
		end
		dLdf(:,exInd)=0;% ensure excluded are ignored
      dtdJ  = -(2*(dRw+tstep*dRd) + dmu - df(:)'*dLdf(:));
      %fprintf('.%d step=%g ddR=%g ddgdw=%g ddgdb=%g  sum=%g\n',j,tstep,2*(dRw+tstep*dRd),-dX*dLdf',-db*sum(dLdf),-dtdJ);
      
		if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		  fprintf('Numerical issues falling back on re-scaled dv');
		  oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		  opts.rescaledv=true; continue;
		end;

      if ( opts.verb > 2 )
        sw  = w + tstep*dw; 
        sb  = b + tstep*db;
		  sf  = tprod(X,[-1 1 2],sw,-1)+sb; % [L x N]
		  dv  = repop(sf,'-',max(sf,[],1)); % re-scale for numerical stability
		  p   = exp(dv);
		  p   = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
		  if ( onetrue ) 
			 g    = p(Yidx);         % [1xN] = Pr(y_true)
		  else
			 g    = sum(p.*Yind,1);  % [1xN] = Pr(y_true)
		  end
		  Ed   = -log(g)*wghtYnEx;   % -ln P(D|w,b,fp)		  
		  Rsw  = R*sw(:);
		  Ew   = sw(:)'*Rsw;         % -ln P(w,b|R);
		  if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
		  J    = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior
        %N.B. don't bother to compute the real gradient... we don't actually use it in the line search
		  dLdf = Yind-p;  % [LxN] = dp_y, i.e. gradient of the probability of the true class
		  dLdf(:,exInd)=0; % ensure ignored points aren't included
        sdJ    = [2*Rsw + mu - X(:,:)*dLdf(:);...
                 -sum(dLdf(:))];
        sdtdJ   =-d(:)'*sdJ(:);  % gradient along the line @ new position
        fprintf('.%d %g=%g (%g) @ %g (%g+%g)\n',j,tstep,dtdJ,sdtdJ,J,Ew,Ed); 
        if ( opts.verb > 3 ) 
          plot(tstep,dtdJ,'*'); text(double(tstep),double(dtdJ),num2str(j));
        end
      end;

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
   w  = w + tstep*dw; 
   b  = b + tstep*db;
   Rw = Rw+ tstep*Rd;
   % compute the other bits needed for CG iteration
   dJ = [2*Rw + mu - X(:,:)*dLdf(:);...
         -sum(dLdf(:))];
	MdJ= dJ;    if ( ~isempty(PC) ) MdJ  = repop(PC,'*',dJ); end; 
   Mr = -MdJ;
   r2 = abs(Mr(:)'*dJ(:)); 
   
   % compute the function evaluation
	if ( onetrue ) 
	  g    = p(Yidx);         % [1xN] = Pr(y_true)
	else
	  g    = sum(p.*Yind,1);  % [1xN] = Pr(y_true)
	end
	Ed = -log(g)*wghtYnEx;    % -ln P(D|w,b,fp)
	Ew = w(:)'*Rw(:);         % -ln P(w,b|R);
	if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
	J  = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,w(1),w(min(end,2)),J,Ew,Ed,r2);
   end

   if ( J > oJ*(1.001) || isnan(J) ) % check for stuckness
      if ( opts.verb>=1 ) 
		  warning('Line-search Non-reduction - aborted'); 
		end;
      J=oJ; w=ow; b=ob; 
      f=tprod(X,[-1 1 2],w,[-1])+b; % [L x N]
      %break;
   end;
   
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
   
   %------------------------------------------------
   % conjugate direction selection
   % N.B. According to wikipedia <http://en.wikipedia.org/wiki/Conjugate_gradient_method>
   %     PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
   delta = max((Mr(:)-oMr(:))'*(-dJ(:))/or2,0); % Polak-Ribier
   %delta = max(r2/or2,0); % Fletcher-Reeves
   d     = Mr+delta*d;     % conj grad direction
   dtdJ  = -d(:)'*dJ(:);   % new search dir grad.
   if( dtdJ <= 0 )         % non-descent dir switch to steepest
      if ( opts.verb >= 2 ) fprintf('non-descent dir\n'); end;      
      d=Mr; dtdJ=-d(:)'*dJ(:); 
   end; 
   
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

% compute the final performance with untransformed input and solutions
Rw   = R*w;
dv   = repop(f,'-',max(f,[],1));  % re-scale for numerical stability
p    = exp(dv);
p    = repop(p,'/',sum(p,1));       % [1xN] = Pr(y_true)
if ( onetrue ) 
  g    = p(Yidx);         % [1xN] = Pr(y_true)
else
  g    = sum(p.*Yind,1);  % [1xN] = Pr(y_true)
end
Ed   = -log(g)*wghtYnEx;            % -ln P(D|w,b,fp)
Ew   = w(:)'*Rw(:);                 % -ln P(w,b|R);
if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
J    = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior
if ( opts.verb >= 0 ) 
   fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,w(1),w(min(end,2)),J,Ew,Ed,r2);
end

% compute final decision values.
if ( all(size(X)==size(oX)) ) f=dv; else f = tprod(oX,[-1 1 2],w,-1)+b; end;
f = reshape(f,size(oY));
obj = [J Ew Ed];
wb=[w;b];
return;

%-----------------------------------------------------------------------------
function []=testCase()
%simple 2d 4 class problem
cents=[-1 0;1 0;0 1;0 -1];
[X,Yl]=mkMultiClassTst(cents,[400 400 400 400],[.2 .2]);[dim,N]=size(X);
Y =lab2ind(Yl)';
clf;labScatPlot(X,Y,'linewidth',1)

% pre-transform X in a class specific way
S =randn(size(X,1),2,size(Y,1));
S =reshape(cents',size(X,1),1,size(Y,1));
XS=tprod(X,[-1 3],S,[-1 1 2]); % [d x L x N]
clf;imagesc('cdata',shiftdim(XS)); % visual how it looks

XSmu = tprod(XS,[1 2 -3],cat(3,single(Y>0),single(Y==0)),[2 -3 3]);clf;imagesc('cdata',shiftdim(XSmu));

[wb,f,Jlr]=mmlr_cg(XS,Y,0,'verb',1);

