function [wb,f,J,obj,tstep]=ls_cg(X,Y,R,varargin);
% Regularised linear least squares Classifier, solved with conjugate-gradients
%
% [wb,f,J,obj]=lr_cg(X,Y,C,varargin)
% Regularised Logistic Regression Classifier using a pre-conditioned
% conjugate gradient solver on the primal objective function
%
% J = w' R w + w' mu + sum_i (y_i = w'*X_i+b).^2
%
% Inputs:
%  X       - [d1xd2x..xN] data matrix with examples in the *last* dimension
%  Y       - [Nx1] matrix of -1/0/+1 labels, (N.B. 0 label pts are implicitly ignored)
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
%            [2x1] for per class weightings
%            [1x1] relative weight of the positive class
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
              'maxIter',inf,'maxEval',[],'tol',0,'tol0',0,'lstol0',1e-5,'objTol',0,'objTol0',1e-4,...            
              'verb',0,'step',0,'wght',[],'X',[],'maxLineSrch',50,...
              'maxStep',3,'minStep',5e-2,'marate',.95,'bPC',[],'wPC',[],'incThresh',.66,'optBias',0,'maxTr',inf,...
              'getOpts',0);
  [opts,varargin]=parseOpts(opts,varargin{:});
  if ( opts.getOpts ) wb=opts; return; end;
end
if ( isempty(opts.maxEval) ) opts.maxEval=5*max(size(X)); end
% Ensure all inputs have a consistent precision
if(isa(X,'double') && isa(Y,'single') ) Y=double(Y); end;
if(isa(X,'single')) eps=1e-7; else eps=1e-16; end;
opts.tol=max(opts.tol,eps); % gradient magnitude tolerence

szX=size(X); nd=numel(szX); N=szX(end); nf=prod(szX(1:end-1));
Y=Y(:); % ensure Y is col vector

% reshape X to be 2d for simplicity
rdim=opts.rdim;
X=reshape(X,[nf N]);
if ( numel(R)==1 )
  szRw=[nf 1]; RType=1; % scalar
elseif ( size(R,2)==1 && numel(R)==nf ) % weighting vector
  szRw=[nf 1]; RType=2;
elseif ( size(R,2)==1 && numel(R)==nf*nf ) % full matrix
  R=reshape(R,[nf nf]); szRw=[nf 1]; RType=1;
elseif ( isempty(rdim) && size(R,1)==szX(1) ) %nD inputs R should replicate over leading dimensions
  szRw=[szX(1) prod(szX(2:end-1))]; RType=3;
elseif ( ~isempty(rdim) && size(R,1)==prod(szX(rdim)) && min(rdim)==1 ) 
  szRw=[prod(szX(rdim)) prod(szX(max(rdim)+1:end-1))];  RType=3;
elseif ( isempty(rdim) && size(R,1)==szX(end-1) )  %nD inputs R should replicate over trailing dimensions
  szRw=[prod(szX(1:end-2)) szX(end-1)]; RType=4;
elseif ( ~isempty(rdim) && size(R,1)==prod(szX(rdim)) && max(rdim)==nd-1 )
  szRw=[prod(szX(1:min(rdim))) prod(szX(rdim))];      RType=4;
elseif ( ~isempty(rdim) && size(R,1)==prod(szX(rdim)) ) % nD inputs R replicate over middle dims
  szRw=szX(1:end-1); R=reshape(R,[szX(rdim) szX(rdim)]); RType=5;
  rdimIdx=1:nd-1; rdimIdx(rdim)=-rdim;
else
  error('Huh, dont know how to use this regularisor');
end
mu=opts.mu; if ( numel(mu)>1 ) mu=reshape(mu,[nf 1]); else mu=0; end;

% check for degenerate inputs
if ( all(Y>=0) || all(Y<=0) )
  warning('Degnerate inputs, 1 class problem');
end

% N.B. this form of loss weighting has no true probabilistic interpertation!
wght=opts.wght;
if ( ~isempty(opts.wght) ) % point weighting
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
else
  wght=1;
end

% check if it's more efficient to sub-set the data, because of lots of ignored points
oX=X; oY=Y;
incIdx=Y(:)~=0;
if ( sum(incIdx)./numel(Y) < opts.incThresh ) % if enough ignored to be worth it
   if ( sum(incIdx)==0 ) error('Empty training set!'); end;
   X=X(:,incIdx); Y=Y(incIdx); if ( numel(wght)==numel(Y) ) wght=wght(incIdx); end;
end

% generate an initial seed solution if needed
wb=opts.wb;   % N.B. set the initial solution
if ( isempty(wb) && ~isempty(opts.alphab) ) wb=opts.alphab; end;
if ( isempty(wb) )    
  w=zeros(nf,1);b=0;
  % prototype classifier seed
  alpha=zeros(numel(Y),1);
  alpha(Y>0)=.5./sum(wght.*Y(Y>0)); 
  muP=X*alpha;
  alpha(:)=0; alpha(Y<0)=.5./sum(wght.*Y(Y<0)); 
  muN=X*alpha;
  X2 =sum(X.*X,2)-((muP+muN)/2);
  w  = (muP-muN)./X2; % per feature LDA style seed
  wX = w'*X; 
  b  = -mean(wX);
  wX = wX+b;
  switch ( RType ) % diff types regularisor
   case 1; Rw=R*w;
   case 2; Rw=R(:).*w;
   case 3; Rw=R*reshape(w,szRw); % leading dims
   case 4; Rw=reshape(w,szRw)*R; % trailing dims
   case 5; Rw=tprod(w,rdimIdx,R,[-(1:numel(szRw)) 1:numel(szRw)]); % middle dims
  end
  wRw   = w'*Rw(:);
  % find least squares optimal scaling and bias
  sb = pinv([wRw+wX*wX' sum(wX); sum(wX) sum(Y~=0)])*[wX*Y; sum(wght.*Y)];
  w=w*sb(1); b=sb(2);
else
  w=wb(1:end-1); b=wb(end);
end 

switch ( RType ) % diff types regularisor
 case 1; Rw=R*w;
 case 2; Rw=R(:).*w;
 case 3; Rw=R*reshape(w,szRw);
 case 4; Rw=reshape(w,szRw)*R;
 case 5; Rw=tprod(w,rdimIdx,R,[-(1:numel(szRw)) 1:numel(szRw)]); % middle dims
end
f   = (w'*X+b)';
err  = Y-f;
werr = wght.*err;

% set the pre-conditioner
% N.B. the Hessian for this problem is:
%  H  =[2*X*wght*X'+ddR  2*wght'*X;...
%       (2*wght'*X)'     sum(wght)];
%  so diag(H) = [sum(X.*X,2)*wght+2*diag(R);sum(wght)];
wPC=opts.wPC; bPC=opts.bPC;
if ( isempty(wPC) ) 
  if ( numel(wght)>1 )
    wPCx=tprod(X,[1 -2],[],[1 -2])*wght(:); 
  else
    wPCx=X(:)'*X(:)*wght;
  end
  %wPC=1./((X.^2)*wght');%ones(size(X,1),1);%
  % include the effect of the regularisor
  switch ( RType ) % diff types regularisor
   case 1; wPC=wPC+2*diag(R);
   case 2; wPC=wPC+2*R(:);
   case 3; tmp=repmat(diag(R) ,1,nf./size(R,1)); wPC=wPC+2*tmp(:); % leading dim
   case 4; tmp=repmat(diag(R)',nf./size(R,1),1); wPC=wPC+2*tmp(:); % trailing dim
   case 5; rsz=szX(1:end-1); rsz(setdiff(1:end,rdim))=1; wPC=repop(wPC,'+',2*reshape(diag(R),rsz));% middle dims
  end
  wPC(wPC<eps) = 1; 
  wPC=1./wPC;
end;
if ( isempty(bPC) ) % Q: Why this pre-conditioner for the bias term?
  if ( numel(wght)>1 )
    bPC=1./sum(wght);
  else
    bPC=1./size(X,2);
  end
end % ==mean(diag(cov(X)))
if ( isempty(wPC) ) 
  wPC=ones(size(X,1),1); 
end;
if ( numel(wPC)==1 ) % scalar pre-condn
  PC = [ones(size(X,1),1)*wPC;bPC];
elseif ( max(size(wPC))==numel(wPC) ) % vector pre-condn
  PC = [wPC(:);bPC];
elseif ( all(size(wPC)==size(X,1)) )  % matrix pre-cond
  PC = zeros(size(X,1)+1); PC(1:size(wPC,1),1:size(wPC,2))=wPC; PC(end)=bPC;
else
  PC = [];
end

dJ   = [2*Rw(:) + mu - 2*X*werr; ...
        -2*sum(werr)];
% precond'd gradient:
%  [H  0  ]^-1 [ dR-X'*(y-f)] 
%  [0  bPC]    [   -1'*(y-f)] 
if ( size(PC,2)==1 ) % vector pre-conditioner
  MdJ  = PC.*dJ; % pre-conditioned gradient
else % matrix pre-conditioner
  if ( size(PC,1)==size(X,1)+1 ) % PC is full size
    MdJ = PC*dJ;
  elseif ( RType==3 && size(wPC,1)==szX(1) )
    MdJ = wPC*reshape(dJ(1:end-1),szX(1:2)); MdJ=[MdJ(:);bPC*dJ(end)];
  elseif ( RType==4 && size(wPC,1)==szX(end-1) )
    MdJ = reshape(dJ(1:end-1),szX(1:2))*wPC; MdJ=[MdJ(:);bPC*dJ(end)];
  else % now what?
    
  end
end
Mr   =-MdJ;
d    = Mr;
dtdJ =-(d'*dJ);
r2   = dtdJ;

Ed   = err'*werr;
Ew   = w'*Rw(:);     % -ln P(w,b|R);
if( ~isequal(mu,0) ) Emu=w'*mu; else Emu=0; end;
J    = Ed + Ew + Emu + opts.Jconst;       % J=neg log posterior

% Set the initial line-search step size
step=abs(opts.step); 
%if( step<=0 ) step=1; end % N.B. assumes a *perfect* pre-condinator
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
   ostep=inf;step=tstep;%max(tstep,abs(1e-6/dtdJ)); % prev step size is first guess!
   odtdJ=dtdJ; % one step before is same as current
   % pre-compute and cache the linear/quadratic terms
   %f0  = f;
   dw  = d(1:end-1); db=d(end);
   df  = dw'*X+db;
   dwerr0=df*(werr);
   dfwdf =df*(wght.*df)';
   if( ~isequal(mu,0) ) dmu = dw'*mu; else dmu=0; end;
   switch ( RType ) % diff types regularisor
    case 1; Rw=R*w;       dRw=dw'*Rw;  % scalar or full matrix
            Rd=R*dw;      dRd=dw'*Rd;
    case 2; Rw=R(:).*w;   dRw=dw'*Rw(:); % component weighting
            Rd=R(:).*dw;  dRd=dw'*Rd(:);
    case 3; Rw=R*reshape(w,szRw); dRw=dw'*Rw(:); % matrix weighting - leading dims
            Rd=R*reshape(dw,szRw);dRd=dw'*Rd(:); 
    case 4; Rw=reshape(w,szRw)*R; dRw=dw'*Rw(:); % matrix weighting - trailing dims
            Rd=reshape(dw,szRw)*R;dRd=dw'*Rd(:); 
    case 5; Rw=tprod(w,rdimIdx,R,[-(1:numel(szRw)) 1:numel(szRw)]);  dRw=dw'*Rw(:); % middle dims
            Rd=tprod(dw,rdimIdx,R,[-(1:numel(szRw)) 1:numel(szRw)]); dRd=dw'*Rd(:); % middle dims            
   end
   dtdJ0=abs(dtdJ); % initial gradient, for Wolfe 2 convergence test
   for j=1:opts.maxLineSrch;
      neval=neval+1;
      oodtdJ=odtdJ; odtdJ=dtdJ; % prev and 1 before grad values
      
      %f     = f0 + tstep*df;err   = Y-f;werr  = wght.*err; dfwerr= df*werr; % direct computation
      %df*werr = df*(wght.*err) = df*(wght.*(Y-f)) = df*(wght.*(Y-f0-tstep*df) = df*(wght.*(Y-f0))-tstep*df*(wght.*df);
      dfwerr = dwerr0 - tstep*dfwdf; % using the cached values
      dtdJ  = -2*( dRw+tstep*dRd  + dmu - dfwerr);
      if ( 0 ) 
        %sw  = w + tstep*dw; 
        %sb  = b + tstep*db;
        sRw = R.*(w+tstep*dw);%Rw+ tstep*Rd;
        % N.B. don't bother to compute the real gradient... we don't actually use it in the line search        
        dJ    = [2*sRw + mu - X*werr;...
                 -sum(werr)];
        dtdJ   =-d'*dJ;  % gradient along the line @ new position
      end
      
      
      if ( opts.verb > 2 )
         Ed   = err'*werr'
         Ew   = w(:)'*Rw(:)+tstep*2*dRw+tstep.^2*dRd;  % w'*(R*reshape(w,szR));       % P(w,b|R);
         J    = Ed + Ew + opts.Jconst;              % J=neg log posterior
         if( ~isequal(mu,0) ) Emu=(w+tstep*d(1:end-1))'*mu; J=J+Emu; end;
         fprintf('.%d %g=%g @ %g (%g+%g)\n',j,tstep,dtdJ,J,Ed,Ew); 
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
   % update error and it's weighted version
   err = err - tstep*df';
   werr= wght.*err;
   % compute the other bits needed for CG iteration
   dJ = [2*Rw(:) + mu - X*werr;...
         -sum(werr)];
   if ( size(PC,2)==1 ) % vector pre-conditioner
     MdJ  = PC.*dJ; % pre-conditioned gradient
   else % matrix pre-conditioner
     if ( size(PC,1)==size(X,1)+1 ) % PC is full size
       MdJ = PC*dJ;
     elseif ( RType==3 && size(wPC,1)==szX(1) ) % leading dim
       MdJ(1:end-1) = wPC*reshape(dJ(1:end-1),szX(1:2)); MdJ(end)=bPC*dJ(end);
     elseif ( RType==4 && size(wPC,1)==szX(end-1) ) % trailing dim
       MdJ(1:end-1) = reshape(dJ(1:end-1),szX(1:2))*wPC; MdJ(end)=bPC*dJ(end);
     else % now what?
       
     end
   end
   Mr =-MdJ;
   r2 =abs(Mr'*dJ); 
   
   % compute the function evaluation
   Ed   = err'*werr;
   Ew   = w'*Rw(:);% P(w,b|R);
   J    = Ed + Ew + opts.Jconst;       % J=neg log posterior
   if( ~isequal(mu,0) ) Emu=w'*mu; J=J+Emu; end;
   if(opts.verb>0)   % debug code      
      fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g' lend],...
              iter,neval,w(1),w(2),J,Ew,Ed,r2);
   end   

   if ( J > oJ*(1.001) || isnan(J) ) % check for stuckness
      if ( opts.verb>=1 ) warning('Line-search Non-reduction - aborted'); end;
      J=oJ; w=ow; b=ob; 
      f= (w'*X+b)';
      break;
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
   %      PR is much better when have adaptive pre-conditioner so more robust in non-linear optimisation
   delta = max((Mr-oMr)'*(-dJ)/or2,0); % Polak-Ribier
   %delta = max(r2/or2,0); % Fletcher-Reeves
   d     = Mr+delta*d;     % conj grad direction
   dtdJ  = -d'*dJ;         % new search dir grad.
   if( dtdJ <= 0 )         % non-descent dir switch to steepest
      if ( opts.verb >= 2 ) fprintf('non-descent dir\n'); end;      
      d=Mr; dtdJ=-d'*dJ; 
   end; 
   
end;

if ( J > J0*(1+1e-4) || isnan(J) ) 
   if ( opts.verb>=0 ) warning('Non-reduction');  end;
   w=w0; b=b0;
end;

% compute the final performance with untransformed input and solutions
switch ( RType ) % diff types regularisor
 case 1; Rw=R*w;                  % scalar or full matrix
 case 2; Rw=R(:).*w;                 % component weighting
 case 3; Rw=R*reshape(w,szRw);    % matrix weighting - leading dims
 case 4; Rw=reshape(w,szRw)*R; % matrix weighting - trailing dims
 case 5; Rw=tprod(w,rdimIdx,R,[-(1:numel(szRw)) 1:numel(szRw)]); % middle dims
end
Ed   = err(:)'*werr(:);
Ew   = w(:)'*Rw(:);     % -ln P(w,b|R);
J    = Ed + Ew + opts.Jconst;       % J=neg log posterior
if( ~isequal(mu,0) ) Emu=w'*mu; J=J+Emu; end;
if ( opts.verb >= 0 ) 
   fprintf(['%3d) %3d x=[%8f,%8f,.] J=%5f (%5f+%5f) |dJ|=%8g\n'],...
           iter,neval,w(1),w(2),J,Ew,Ed,r2);
end

% compute final decision values.
f = w'*oX + b;
f = reshape(f,size(oY));
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
[X,Y]=mkMultiClassTst([zeros(1,47) -1 0 zeros(1,47); zeros(1,47) 1 0 zeros(1,47); zeros(1,47) .2 .5 zeros(1,47)],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

K=X'*X; tic, [alphab,f,J]=klr_cg(K,Y,0,'verb',1,'objTol0',1e-10);toc
wb=[X*alphab(1:end-1);alphab(end)];
dvk=wb(1:end-1)'*X+wb(end);

tic,[wb,f,Jlr]=lr_cg(X,Y,0,'verb',1,'objTol0',1e-10);toc
dv=wb(1:end-1)'*X+wb(end);
mad(dvk,dv)

% same but now with a linear term
alpha0=randn(N,1); mu0=X*alpha0;
[alphab,J]=kls_cg(K,Y,[1 1],'mu',mu0,'verb',2);
[wb,J]    =ls_cg (X,Y,1,'mu',X*mu0,'verb',2);

% plot the resulting decision function
clf;plotLinDecisFn(X,Y,wb(1:end-1),wb(end));

trnInd=true(size(X,2),1);
fInds=gennFold(Y,10,'perm',1); 
trnInd=fInds(:,end)<0; tstInd=fInds(:,end)>0; trnSet=find(trnInd);

[wb0,f0,J0]=ls_cg(X(:,trnInd),Y(trnInd),1,'verb',1);
dv=wb(1:end-1)'*X(:,tstInd)+alphab(end);
dv2conf(Y(tstInd),dv(:))

% test implicit ignored
[wb,f,Jlr]=ls_cg(X,Y.*single(fInds(:,end)<0),1,'verb',1);

% test using the wght to ignore points
[wb,f,Jlr]=ls_cg(X,Y,1,'verb',1,'wght',single(trnInd));

% test the automatic pre-condnitioner 
ls_cg(X,Y,[10000000;ones(size(X,1)-1,1)],'verb',1,'objTol0',1e-10); % diff reg for diff parameters
ls_cg(X,Y,[10000000;ones(size(X,1)-1,1)],'verb',1,'objTol0',1e-10,'wPC',1./(1+[10000000;ones(size(X,1)-1,1)]));
ls_cg(repop([1e6;ones(size(X,1)-1,1)],'*',X),Y,1,'verb',1,'objTol0',1e-10,'wPC',1); %re-scale the data, no PC
ls_cg(repop([1e6;ones(size(X,1)-1,1)],'*',X),Y,1,'verb',1,'objTol0',1e-10); %re-scale the data, auto PC

% test matrix pre-conditioner
ls_cg(X,Y,diag([10000000;ones(size(X,1)-1,1)]),'verb',1,'objTol0',1e-10,'wPC',diag(1./(1+[10000000;ones(size(X,1)-1,1)])));

% test non-diagonal regularisor
w=randn(size(X,1),1); w=w./norm(w);
R=eye(size(X,1))+10000*w*w';
P=eye(size(X,1))-w*(1-1./10000)*w';
ls_cg(X,Y,R,'verb',1,'objTol0',1e-10);
% and now with the corrospending pre-conditioner
ls_cg(X,Y,R,'verb',1,'objTol0',1e-10,'wPC',P);

% test leading dims block regularisor and PC
X2d=reshape(X,[size(X,1)/4 4 size(X,2)]);
w  =randn(size(X2d,1),1); w=w./norm(w);
R  =eye(size(X2d,1))+100000*w*w';
P  =eye(size(X2d,1))-w*(1-1/100000)*w';
ls_cg(X2d,Y,R,'verb',1,'objTol0',1e-10);
% and now with the corrospending pre-conditioner
ls_cg(X2d,Y,R,'verb',1,'objTol0',1e-10,'wPC',P);


% test re-seeding for increasing Cs
Cs=2.^(1:7);
% without re-seeding
for ci=1:numel(Cs);
  [wbs(:,ci),f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),Cs(ci),'verb',1);
end
% with re-seeding
[wbs(:,1),f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),Cs(1),'verb',1,'wb',[]);
for ci=2:numel(Cs);
  [wbs(:,ci),f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),Cs(ci),'verb',1,'wb',wbs(:,ci-1));
end

% simple regulasor
tic,[wb,f,J]=ls_cg(X2d,Y,1,'verb',1,'dim',3);toc;

% test using a re-scaling regularisor
R=rand(size(X,1),1);
% transform both X and R to give the same as the orginal problem, just re-scaled
tic,[wb,f,Jlr]=ls_cg(repop(X,'*',sqrt(R)),Y,diag(R),'verb',1);toc
tic,[wb,f,Jlr]=ls_cg(repop(X,'*',sqrt(R)),Y,R,'verb',1);toc
% transform only R or only X to give the same problem
[wb,f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),R,'verb',1);
[wb,f,Jlr]=ls_cg(repop(X(:,trnInd),'*',1./sqrt(R)),Y(trnInd),1,'verb',1);

% try a regularisor which imposes that both components should be similar
% R=[1 -1;-1 1] thus high cross correlations are good
R=-ones(size(X,1),size(X,1));R(1:size(R,1)+1:end)=1; 
[wb,f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),R,'verb',1);

% test using mu to specify a prior towards a particular solution
mu=[1 1]';
[wb,f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),64,'verb',1,'mu',64*-2*mu);
Cs=2.^(1:7);
for ci=1:numel(Cs);
  [wbs(:,ci),f,Jlr]=ls_cg(X(:,trnInd),Y(trnInd),Cs(ci),'verb',1,'mu',Cs(ci)*-2*mu,'Jconst',Cs(ci)*mu'*mu);
end
clf;plot(wbs)
