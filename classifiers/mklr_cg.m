function [wb,f,J,obj]=mklr_cg(K,Y,C,varargin);
% Regularised Kernel Logistic Regression Classifier
%
% [alphab,f,J,obj]=klr_cg(K,Y,C,varargin)
% Regularised Kernel Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J = C(1) w' K w + C(2) w'*K mu + sum_i log( (1 + exp( - y_i ( w'*K_i + b ) ) )^-1 ) 
%
% Inputs:
%  K       - [NxN] kernel matrix
%  Y       - [Nx1] matrix of -1/0/+1 labels, (0 label pts are implicitly ignored)
%  C       - the regularisation parameter, roughly max allowed length of the weight vector
%            good default is: .1*var(data) = .1*(mean(diag(K))-mean(K(:))))
%
% Outputs:
%  alphab  - [(N+1)x1] matrix of the kernel weights and the bias [alpha;b]
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]
%  p       - [Nx1] vector of conditional probabilities, Pr(y|x)
%  mu      - [Nx1] vector containing mu
%
% Options:
%  alphab  - [(N+1)x1] initial guess at the kernel parameters, [alpha;b] ([])
%  ridge   - [float] ridge to add to the kernel to improve convergence.  
%             ridge<0 -- absolute ridge value
%             ridge>0 -- size relative to the mean kernel eigenvalue
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
%            [2x1] for per class weightings
%            [1x1] relative weight of the positive class
%            N.B. to give each class equal importance use: wght= [1/np 1/nn]*(np+nn)
%                 where np=number positive examples, nn=number negative examples
%  nobias  - [bool] flag we don't want the bias computed                  (false)
% Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)

% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty, express or
% implied
  if ( nargin < 3 ) C(1)=0; end;
  opts=struct('alphab',[],'wb',[],'dim',[],'mu',[],'Jconst',0,...
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-5,'objTol',0,'objTol0',1e-4,...
              'verb',0,'step',0,'wght',[],'X',[],'ridge',0,'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'bPC',[],...
				  'incThresh',.75,'optBias',0,'maxTr',inf,...
				  'compBinp',1,'rescaledv',0);
  [opts,varargin]=parseOpts(opts,varargin{:});
  opts.ridge=opts.ridge(:);
  if ( isempty(opts.maxEval) ) opts.maxEval=5*sum(Y(:)~=0); end
									  % Ensure all inputs have a consistent precision
  if(isa(K,'double') && isa(Y,'single') ) Y=double(Y); end;
  if(isa(K,'single')) eps=1e-7; minp=1e-40; else eps=1e-16; minp=1e-120; end;
  opts.tol=max(opts.tol,eps); % gradient magnitude tolerence

  dim=opts.dim;
  if ( isempty(dim) ) dim=ndims(K); end;
  szK=size(K); szK(end+1:max(dim))=1; % input size (padded with 1 for extra unity dimensions)
  nd=numel(szK); N=prod(szK(dim)); nf=prod(szK(setdiff(1:end,dim)));
										  % reshape X to be 2d for simplicity
  K=reshape(K,[nf N]);

  if( size(Y,1)==N && size(Y,2)~=N ) Y=Y'; end; % ensure Y has examples in last dim
  binp=false;
  if( size(Y,1)==1 )
	 if( all(Y(:)==-1 | Y(:)==0 | Y(:)==1) ) % binary problem
		binp=true;
		Y=[Y; -Y]; % convert to per-class labeling
	 elseif ( all(Y(:)>=0 & Y(:)==ceil(Y(:))) ) % class labels input, convert to indicator matrix
		Yl=Y;key=unique(Y);key(key==0)=[];
		Y=zeros(numel(key),N);for l=1:numel(key); Y(l,:)=Yl==key(l); end;	 
	 end
  elseif (size(Y,1)==2 && all((Y(:)>=0 & Y(:)<=1) | Y(:)==-1)) % binary indicator input
	 binp=true;
  end
  if ( size(Y,2)~=N ) error('Y should be [LxN]'); end;
					%if ( size(Y,1)==1 ) error('1-class problem input....'); end;
  Y(:,any(isnan(Y),1))=0; % convert NaN's to 0 so are ignored
  L=size(Y,1);

										  % check for degenerate inputs
  if ( all(diff(Y,1)==0) ) warning('Degnerate inputs, 1 class problem'); end

  if ( opts.ridge>0 ) % make the ridge relative to the max eigen-value
    opts.ridge = opts.ridge*median(abs(diag(K)));
    ridge = opts.ridge;
  else % negative value means absolute ridge
    ridge = abs(opts.ridge);
  end

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

% check if it's more efficient to sub-set the kernel, because of lots of ignored points
  oK=K; oY=Y;
  incInd=any(Y~=0,1);  exInd=find(~incInd);
  if ( sum(incInd)./size(Y,2) < opts.incThresh ) % if enough ignored to be worth it
    if ( sum(incInd)==0 ) error('Empty training set!'); end;
    K=K(incIdx,incIdx); Y=Y(incIdx);
	 exInd=[];	
  end

										 % generate an initial seed solution if needed
  wb=opts.wb;   % N.B. set the initial solution
  if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
  if ( ~isempty(wb) ) % sub-set the seed
	 if ( binp )
		w=wb(1:end-1); b=wb(end);
	 else
		w=reshape(wb(1:end-L),   nf,L);
		b=reshape(wb(end-L+1:end),1,L);
	 end
	 if ( ~all(size(K)==size(oK)) ) w=w(incInd,:); end;
  else
	 b=zeros(1,size(Y,1));
										  % prototype classifier seed
	 w= single(Y>0)'; if ( size(Y,1)==1 ) w=[single(Y>0)'; single(Y<0)']; end
	 w= repop(w,'./',max(1,sum(w,1))); % guard divide by 0  
										  % discard the 2nd boundary if binary problem
	 if ( binp ) w=w(:,1); b=b(1); end;
  end

% build index expression so can quickly get the predictions on the true labels
Y(Y<0)=0; % remove negative indicators
sY    =double(sum(Y,1)); % pre-comp scaling factor
Y1    =Y(1,:);
% BODGE/TODO : This is silly as if onetrue the only need to learn L-1 weight vectors.......
onetrue=all((sY==1 & sum(Y>0,1)==1) | (sY==0 & sum(Y>0,1)==0));
exInd  =find(sY==0);

% Normalise the kernel to prevent rounding issues causing convergence problems
% = average kernel eigen-value + regularisation const = ave row norm
diagK= K(1:size(K,1)+1:end); 
if ( sum(incInd)<size(K,1) ) diagK=diagK(incInd); end;
muEig=median(diagK); % approx hessian scaling, for numerical precision
					% adjust alpha and regul-constant to leave solution unchanged
if ( muEig < 10 && muEig>.1 )
  muEig=1;
else
  w=w*muEig;
  C(1) = C(1)./muEig;
end; % don't re-scale when near 1

										  % set the bias (i.e. b) pre-conditioner
bPC=opts.bPC;
if ( isempty(bPC) ) % bias pre-condn with the diagonal of the hessian
  bPC  = sqrt(abs(muEig + 2*C(1))./muEig);   % N.B. use sqrt for safety?
  bPC  = 1./bPC;
										  %fprintf('bPC=%g\n',bPC);
end

Kw   = (K*w + ridge.*w)./muEig; % [ N x L ] % N.B. this way round to make wKw cheap = Kw(:)'*w(:)
f    = repop(Kw','+',b(:));% [L x N]
if ( size(f,1)>1 && max(abs(f(:)))>30 ) opts.rescaledv=1; end; % auto-rescale 
if ( size(f,1)>1 && opts.rescaledv )
  f  = repop(f,'-',max(f,[],1)); % re-scale for numerical stability
end
p    = exp(f);
if ( size(p,1)==1 ) % binary problem
  p  = p./(1+p);
  if ( onetrue ) % fast-path common case of single true class with equal example weights
	 dLdf= Y1 - p;
	 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
  else
	 dLdf = Y1 - p.*sY; % dLdf = Yerr % [L x N]
  end
										  %dLdf= Y1 - p.*sY; % dLdf = Yerr % [L x N]
else % multi-class
  p    = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
  if ( onetrue ) % fast-path common case
	 dLdf = Y-p;                % [LxN] = dp_y, i.e. gradient of the log true class probability
	 dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
  else
	 dLdf = Y-repop(p,'*',sY);  % [LxN] = dln(p_y), i.e. gradient of the expected log true class prob
  end
end

				% precond'd gradient:
				%  [K  0  ]^-1 [(lambda*wK-K((1-g).Y))] = [lambda w - (1-g).Y]
				%  [0  bPC]    [ -1'*((1-g).Y)        ]   [ -1'*(1-g).Y./bPC  ] 
MdJdw = (2*C(1)*w - dLdf');
MdJdb = -sum(dLdf,2)'./bPC;
MdJ   = [MdJdw;MdJdb];
dJ    = [(K*MdJdw+ridge*MdJdw)./muEig; ...
         -sum(dLdf,2)'];
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d(:)'*dJ(:));
r2   = dtdJ;
r02  = r2;

					% Ed = entropy loss = expected log-prob of the true class
					%Ed  = -log(max(p(:),eps))'*Y(:); % -ln P(D|w,b,fp)
if ( onetrue ) % fast-path for the common case - 1 true class , equal weight
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
Ew   = C(1)*Kw(:)'*w(:);        % -ln P(w,b|R);
J    = Ed + Ew;       % J=neg log posterior

										  % Set the initial line-search step size
step=abs(opts.step); 
		  %if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
if( step<=0 ) step=min(sqrt(abs(J/max(dtdJ,eps))),1); end %init step assuming opt is at 0
tstep=step;

neval=1; lend='\r';
if(opts.verb>0)   % debug code      
  if ( opts.verb>1 ) lend='\n'; else fprintf('\n'); end;
  fprintf(['%3d) %3d x=[%5f,%5f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],0,neval,w(1),w(2),J,Ew./muEig,Ed,r2);
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
    fprintf('.%d %g=%g @ %g (%g+%g)\n',0,0,dtdJ,J,Ed,Ew./muEig); 
    if ( opts.verb>3 ) 
      hold off;plot(0,dtdJ,'r*');hold on;text(0,double(dtdJ),num2str(0)); 
      grid on;
    end
  end;
  ostep=inf;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
  odtdJ=dtdJ; % one step before is same as current
  f0  = f;
  dw  = d(1:end-1,:); db=d(end,:);
  Kd  = (K*dw+ridge.*dw)./muEig; % [NxL] 
  df  = repop(Kd','+',db(:));
  Kw0 = Kw;
  dKw = Kd(:)'*w(:); dKd=Kd(:)'*dw(:);
  dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
  for j=1:opts.maxLineSrch;
    neval=neval+1;
    oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
    
    % Eval the gradient at this point.  N.B. only gradient needed for secant
	 f    = f0  + tstep*df;
										  % re-scale for numerical stability
	 if(size(f,1)>1 && opts.rescaledv) f=repop(f,'-',max(f,[],1)); end;
	 p    = exp(f);
	 if ( size(p,1)==1 ) % binary problem
		p  = p./(1+p);
		if ( onetrue ) % fast-path common case
		  dLdf = Y1-p;
		  dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
		else
		  dLdf = Y1-p.*sY;
		end
	 else % multi-class
		p    = repop(p,'/',sum(p,1)); % [L x N] =Pr(x|y_i) = exp(w_ix+b)./sum_y(exp(w_yx+b));
		if ( onetrue ) 
		  dLdf = Y-p;  % [LxN] = dp_y, i.e. gradient of the log probability of the true class
		  dLdf(:,exInd)=0; % ensure ignored points aren't included (as not done above)
		else
  % [LxN] = dln(p_y), i.e. gradient of the log probability of the true class
		  dLdf = Y-repop(p,'*',sY);  	 
		end
	 end
    dtdJ  = -(2*C(1)*(dKw+tstep*dKd) - df(:)'*dLdf(:));
    
	 if ( ~opts.rescaledv && isnan(dtdJ) ) % numerical issues detected, restart
		if (opts.verb>=0) fprintf('%d) Numerical issues falling back on re-scaled dv\n',iter); end
		oodtdJ=odtdJ; dtdJ=odtdJ;%reset obj info
		opts.rescaledv=true; continue;
	 end;

		if ( opts.verb > 2 )
      Ed   = -log(max(g,eps))*(Y.*wghtY);         % P(D|w,b,fp)
      Ew   = C(1)*(Kw(:)'*w(:)+tstep*Kw(:)'*dw(:));  % P(w,b|R);
      J    = Ed + Ew;               % J=neg log posterior         
      fprintf('.%d %g=%g @ %g (%g+%g)\n',j,tstep,dtdJ,J,Ew./muEig,Ed); 
      if ( opts.verb > 3 ) 
        plot(tstep,dtdJ,'*'); text(double(tstep),double(dtdJ),num2str(j));
      end
    end;

								% convergence test, and numerical res test
    if(iter>1||j>3) % Ensure we do decent line search for 1st step size!
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
% but prev points gradient, this is necessary stop very steep orginal gradient preventing decent step sizes
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
										  % update the solution with this step
  w  = w   + tstep*dw;
  b  = b   + tstep*db;
  Kw = Kw0 + tstep*Kd; % cached info
  
									 % compute the other bits needed for CG iteration
  MdJdw = (2*C(1)*w - dLdf');
  MdJdb = -sum(dLdf,2)'./bPC;
  MdJ   = [MdJdw;MdJdb];
  dJ    = [(K*MdJdw+ridge*MdJdw)./muEig; ...
           -sum(dLdf,2)'];
%dJ(1:end-1) = (2*C(1)*wK + C(2)*muK./muEig -(Yerr*K)./muEig);%N.B. wK0 and dK already include muEig
%dJ(end)     = bPC*MdJ(end);
  Mr =-MdJ;
  r2 =abs(Mr(:)'*dJ(:)); 
						% compute the function evaluation
						% Ed = entropy loss = expected log-prob of the true class
						%Ed  = -log(max(p(:),eps))'*Y(:); % -ln P(D|w,b,fp)
  if ( onetrue ) % fast-path for the common case - 1 true class , equal weight
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
  Ew   = C(1)*Kw(:)'*w(:);      % -ln P(w,b|R);
  J    = Ed + Ew;       % J=neg log posterior
  if(opts.verb>0)   % debug code      
    fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
            iter,neval,w(1),w(2),J,Ew./muEig,Ed,r2);
  end   

  if ( J > oJ*(1+1e-3) || isnan(J) ) % check for stuckness
    if ( opts.verb>=0 ) warning('Line-search Non-reduction - aborted'); end;
    J=oJ; w=ow; b=ob; break;
  end;
  
									%------------------------------------------------
									% convergence test
  if ( iter==1 )     madJ=abs(oJ-J); dJ0=max(abs(madJ),eps); r02=r2;
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
  delta = max((Mr(:)-oMr(:))'*(-dJ(:))/or2,0); % Polak-Ribier
										  %delta = max(r2/or2,0); % Fletcher-Reeves
  d     = Mr+delta*d;     % conj grad direction
  dtdJ  =-d(:)'*dJ(:);    % new search dir grad.
  if( dtdJ <= 0 )         % non-descent dir switch to steepest
    if ( opts.verb >= 2 ) fprintf('non-descent dir\n'); end;      
    d=Mr; dtdJ=-d(:)'*dJ(:); 
  end; 
  
end;
if ( opts.verb >= 0 ) 
  fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
          iter,neval,w(1),w(2),J,Ew./muEig,Ed,r2);
end

if ( J > J0*(1+1e-4) || isnan(J) ) 
  if ( opts.verb>=0 ) warning('Non-reduction');  end;
  w=w0;b=b0;
end;

										  % fix the stabilising K normalisation
w = w./muEig;

										 % compute final decision values.
if ( ~all(size(K)==size(oK)) ) % map back to the full kernel space, if needed
  nw=zeros(size(oK,1),1); nw(incInd)=w; w=nw;
  K=oK; Y=oY;
  Kw = (K*w + ridge.*w);
  f  = repop(Kw','+',b(:));% [L x N]
end
if( ~binp ) f  = reshape(f,size(Y)); end;
obj= [J Ew./muEig Ed];
wb = [w(:);b(:)];
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
%simple 2d n-class problem
  nd=0; nPts=400; nCls=6; overlap=1;
  cents=cat(2,sin(2*pi*(1:nCls)'/nCls),cos(2*pi*(1:nCls)'/nCls),zeros(nCls,nd));
  [X,Y]=mkMultiClassTst(cents,nPts,[1 1]*overlap*2/nCls/2,[]);[dim,N]=size(X);

  clf;labScatPlot(X,Y,'linewidth',1)

  [wb,flr,Jlr]=mlr_cg(X,Y,0,'verb',1);

  K=X'*X; 
  [alphab,f,Jklr]=mklr_cg(K,Y,0,'verb',1);
  % extract solution for linear kernel
  w=X*reshape(alphab(1:end-L),size(X,2),L); b=alphab(end-L:end);
  
  % plot solution for linear kernel
  clf;plotLinDecisFn(X,Y,w,b);%,[],wght);


