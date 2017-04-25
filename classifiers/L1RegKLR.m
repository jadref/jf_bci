function [Wb,dv,J,alphab]=L1RegKLR(X,Y,C,varargin)
% L1 regularised (Kernel) Logistic Regression
%  
% [Wb,dv,J,alphab]=L1RegKLR(X,Y,C,varargin)
% 
% Inputs:
%  X       -- a n-d array of data
%  Y       -- [size(X,dim) x 1] set of trial data labels in -1/0/+1 format
%             OR
%             [size(X,dim) x M] set of pre-sub-problem labels
%  C       -- [L1,L2] regularisation parameter(s)
% Options:
%  dim     -- epoch/trial dimension of X (-1)
%  grpDim  -- dimensions of X to L1 regularise as a group, 
%             where ndims(X)+1 -> group over sub-problems
%             N.B. all points *along* this dim are regularised together ([])
%  mxDim   -- [int] dimension along which feat trans Mx is applied ([])   !!!!N.B. doesn't work!!!
%  Mx      -- [size(X,mxDim) x F] additional feature space transformation for X's 1st dim before solving ([])
%  iMx     -- [size(X,mxDim) x F] inversion additional feature space transformation ([])
%  maxIter -- maximum number of iterations to perform
%  tol     -- tolerance on change in W, to stop iteration
%  objTol  -- tolerance on change in objective value
%  objTol0 -- tolerance on change % in objective value w.r.t. initial soln value
%  marate  -- moving average decay const, used to smooth J estimates for tol
%  Wb      -- seed solution as {W b}
% Outputs:
%  Wb     -- {W b}
%  dv     -- [size(X,dim) x M] set of classifier predictions
%  J      -- objective function value
%
opts=struct('dim',[],'mxDim',[],'Mx',[],'iMx',[],'grpDim',[],'maxIter',[],'maxEval',inf,'h',1e-2,...
            'tol',0,'tol0',1e-4,'objTol',0,'objTol0',1e-4,'marate',.8,'Wb',[],'alphab',[],'verb',0);
opts=parseOpts(opts,varargin);

szX=size(X); nd=ndims(X); nSubProb=size(Y,2);
dim=opts.dim; if ( isempty(dim) ) dim=nd; end;
if ( numel(dim)>1 ) error('Only 1 trial dim is supported'); end;
if ( ~isempty(opts.grpDim) ) if ( any(dim==opts.grpDim) ) error('Cant group along trials dim!'); end; end
if ( isempty(opts.maxIter) ) opts.maxIter=4*max(szX(dim)); end;
C(end+1:2)=0;
Cc = max(C(:)); C=C./Cc; % split into 3 parts

% Set the convergence thresholds for the inner L_2 regularised sub-problem
cgOpts=struct('maxEval',size(X,dim),'tol',0,'tol0',0,'objTol',0,'objTol0',1e-2);
if( numel(opts.maxEval)>1 ) cgOpts.maxEval=opts.maxEval(2); opts.maxEval=opts.maxEval(1); end
if( numel(opts.tol)>1 )     cgOpts.tol    =opts.tol(2);     opts.tol    =opts.tol(1);     end
if( numel(opts.tol0)>1 )    cgOpts.tol0   =opts.tol0(2);    opts.tol0   =opts.tol0(1);    end
if( numel(opts.objTol)>1 )  cgOpts.objTol =opts.objTol(2);  opts.objTol =opts.objTol(1);  end
if( numel(opts.objTol0)>1 ) cgOpts.objTol0=opts.objTol0(2); opts.objTol0=opts.objTol0(1); end

% Extract the provided seed values
if ( ~isempty(opts.alphab) )
   alphab=opts.alphab;
   W = tprod(X,[1:dim-1 -dim dim+1:ndims(X)],alphab(1:end-1),-dim); b=alphab(end);
elseif ( ~isempty(opts.Wb) )   
   alphab=[];
   W = opts.Wb{1}; b=opts.Wb{2};
else
   W = zeros([szX(1:dim-1) 1 szX(dim+1:end) nSubProb]); b=zeros(nSubProb,1);
end

% feat space trans and its inverse
mxDim=opts.mxDim;
if( ~isempty(mxDim) )
   Mx=opts.Mx; iMx=opts.iMx;
   if( ~isempty(Mx) && isempty(iMx) ) % compute the inverse transform to map solutions back to input space
      iMx=zeros(size(Mx)); for spi=1:size(Mx,3); iMx(:,:,spi)=pinv(Mx(:,:,spi))'; end; 
   end;
end

for spi=1:nSubProb; % setup- the sub-problems
   trnInd(:,spi) = Y(:,spi)~=0; N(spi)=sum(trnInd(:,spi));
   if ( any(~trnInd(:,spi)) )
      % extract the training set
      idx={};for d=1:ndims(X); idx{d}=1:size(X,d); end;
      trnIdx=idx; trnIdx{end}=trnInd(:,spi);
      Xtrn{spi} = X(trnIdx{:});
      Ytrn{spi} = Y(trnInd(:,spi),spi);
   else
      Xtrn{spi}=X; Ytrn{spi}=Y;
   end
end

J=inf; B=1; dW=inf; K=[];
% build some index expressions to get inside different objects
idx={};for d=1:ndims(X); idx{d}=1:size(X,d); end;
Widx=idx; Widx{end+1}=1:size(Y,2); Widx{dim}=1;  % index into the solution matrix
Bidx=idx; Bidx{end+1}=1:size(Y,2); Bidx{dim}=1;  % index into the re-scaling matrix 
if ( ~isempty(opts.grpDim) ) 
   [Bidx{opts.grpDim}]=deal(1); % grp dims are also removed
   grpIdx=1:ndims(X)+1; grpIdx(opts.grpDim)=-grpIdx(opts.grpDim); % indix into grouped dimensions
end 
for iter=0:opts.maxIter;
   oJ=J; oW=W;
   
   % 1) Update the kernel to the new solution   
   % 1.1) compute the weighting needed for the kernel
   if ( max(abs(W(:))) > 0 ) 
      B=0; % R=diag(C(1)./abs(W)) -> B=sqrt(abs(W)/C(1))
      if ( C(1)>eps )
         if ( isempty(opts.grpDim) ) % per element reg  
            B = B + sqrt(abs(W)./C(1)); 
         else % group reg
            B = B + sqrt(sqrt(tprod(W,grpIdx,[],grpIdx))./C(1));
         end
      end
      if ( C(2)>eps ) B = B + sqrt(1./C(2)); end
      idx=abs(B)<opts.h; if ( any(idx) ) B(idx)=opts.h*sign(B(idx)); end;
   end
   
   for spi=1:size(Y,2); % loop over sub-probs
      clear Xtrn2 W2;
      % get the re-scale matrix for this sub-prob 
      if ( numel(B)>1 ) Bspi  = B(Bidx{1:end-1},min(spi,end)); else Bspi=B; end; 
      Xtrn2 = repop(Xtrn{spi},'*',Bspi); % updated inputs
      W2    = repop(W(Widx{1:end-1},spi),'./',Bspi);% updated solution, with inverse B so f remains constant
      if( ~isempty(mxDim) ) % apply the additional Mx transformation
         Xtrn2= tprod(Xtrn2,[1:mxDim-1 -mxDim mxDim+1:ndims(X)],Mx(:,:,min(spi,end)),[-mxDim mxDim]); % transform the data
         W2   = tprod(W2,[1:mxDim-1 -mxDim mxDim+1:ndims(W)],iMx(:,:,min(spi,end)),[-mxDim mxDim]);% inv transform soln so f remains constant
      end
      
      % 1.2) compute the kernel -- include the previous solution as extra point
      if( size(K,1)~=size(Xtrn2,dim) ) K  = zeros(size(Xtrn2,dim)+1,class(Xtrn2)); end; % re-alloc if needed
      K(1:end-1,1:end-1)  = compKernel(Xtrn2,[],'linear','dim',dim); % K = X' * B' * B * X
      Ka = compKernel(Xtrn2,W2,'linear','dim',dim); % Xtrn*B*B^-1*W = Xtrn*W
      Kaa= compKernel(W2,[],'linear','dim',dim);    % B^-1*W*W*B^-1 = W'*(B*B)^-1*W = R_1(W)
      K(end,1:end-1)=Ka; K(1:end-1,end)=Ka; K(end,end)=Kaa; % fill in rest of the kernel
      alphab = zeros(size(K,1)+1,1); alphab(end-1)=1; alphab(end)=b(spi);
   
      % 2) Compute the L2 reg solution with this kernel
      % N.B. Cc/2 to get gradient right, i.e. to cancel the 2 in d(w^2) = 2*w
      %      This approach has the advantage that the displayed info isn't re-scaled hence can
      %      easily check convergence
      [alphab,tf,ttJ,tobj]=klr_cg(K,[Ytrn{spi};0],Cc/2,'alphab',alphab,'verb',opts.verb-1,cgOpts);
   
      % 3) Extract the current input space solution & its L1 reg value
      W2 = tprod(Xtrn2,[1:dim-1 -dim dim+1:ndims(X)],alphab(1:end-2),-dim);
      % N.B. apply extra Bspi to new feature space solution to undo W's inverse Bspi re-scaling      
      if( ~isempty(mxDim) ) % 1st apply the additional inverse Mx transformation, if needed
         W2 = tprod(W2,[1:mxDim-1 -mxDim mxDim+1:ndims(W)],Mx(:,:,min(end,spi)),[mxDim -mxDim]);
      end
      W2 = repop(Bspi,'*',W2); % apply the feature re-scaling
      W(Widx{1:end-1},spi)= W2+oW(Widx{1:end-1},spi)*alphab(end-1); % inc. seed weighting to get updated soln
      b(spi)  = alphab(end);
   end

   dv = repop(tprod(X,[-(1:ndims(X)-1) 1],W,[-(1:ndims(X)-1) 0 2],'n'),'+',b(:)');
   g  = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count
   Ed = -sum(log(g),1);  % -ln(P(D|w,b,fp)) (for each sub-problem)
   if ( ~isempty(opts.grpDim) ) 
      Ew = [sum(sqrt(tprod(W,grpIdx,[],grpIdx))) W(:)'*W(:)];
   else      
      Ew = [sum(abs(W(:))) W(:)'*W(:)]; % L1/L2      
   end
   J  = sum(Ed) + Cc*C(1)*Ew(1) + Cc*C(2)*Ew(2);

   % convergence test stuff
   % initialise stuff in 1st iter, + get initial gradient est, for convergence tests
   if( iter==0 )       J0=J;dJ=J0;madJ=dJ; dJ0=J0; dW = norm(W(:));       dW0=dW;
   elseif ( iter==1 )  dJ=oJ-J;   madJ=dJ; dJ0=dJ; dW = norm(W(:)-oW(:)); dW0=dW;
   else
      dJ=oJ-J; dW = norm(W(:)-oW(:));
      if ( iter < 5 ) dJ0=max(dJ0,dJ); end;  % converg rate is max of 1st few steps
   end
   madJ=madJ*(1-opts.marate)+dJ*(opts.marate); % move-ave grad est
   if( dW  <= norm(W(:))*opts.tol || dW  <= dW0*opts.tol0 || ...
       abs(madJ)<=opts.objTol || abs(madJ) <= dJ0*opts.objTol0 )%|| J>=oJ*(1+1e-2) ) % terminate if obj increased!
      break; 
   end;

   if ( opts.verb > 0 ) 
      fprintf('L1 %2d)\tEw=(%s)=%g\tEd=(%s)\tJ=%8g\tdJ=%g\tdW=%8g\n',...
              iter,sprintf('%g,',Ew),(Ew*C(:)),sprintf('%g,',Ed),J,madJ,dW./norm(W(:)));
   end

end

% compute the full set of predictions
dv = repop(tprod(X,[-(1:ndims(X)-1) 1],W,[-(1:ndims(X)-1) 0 2],'n'),'+',b(:)');
g  = 1./(1+exp(-(Y.*dv))); g=max(g,eps); g(Y==0)=1; % ensure ignored don't count
Ed = -sum(log(g));  % -ln(P(D|w,b,fp))
if ( ~isempty(opts.grpDim) ) 
   Ew = [sum(sqrt(tprod(W,grpIdx,[],grpIdx))) W(:)'*W(:)];
else      
   Ew = [sum(abs(W(:))) W(:)'*W(:)]; % L1/L2      
end
J  = sum(Ed,1) + Cc*C(1)*Ew(1) + Cc*C(2)*Ew(2);
if ( opts.verb >= 0 ) 
   fprintf('L1 %2d)\tEw=(%s)=%g\tEd=(%s)\tJ=%8g\tdJ=%g\tdW=%8g\n',...
           iter,sprintf('%g,',Ew),(Ew*C(:)),sprintf('%g,',Ed),J,madJ,dW./norm(W(:)));
end

Wb={W b};
return;

%------------------------------------------------------------------------------
function testCase()
C=1;
N=400; L=2;
fs = 128; T=3*fs;
Yl = (randn(N,1)>0)+1;     % True labels + 1
Y  = double([Yl==1 Yl==2])*2-1;% indicator N x L
% simple model with 1 class dependent time-domain source and 1 noise source
sources = { {'prod' {'exp' .01} {'sin' 19}} {'coloredNoise' 1}; % ERP
            {'coloredNoise' 1} {};  % pure noise source
          }; % rest just detectors
y2mix=cat(3,[1 .1;.2 .0],[0 .1;.2 .0]); % ch x source x label
mix  =y2mix(:,:,Yl);                % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T,10);

% L2 reg
K = compKernel(X,[],'linear','dim',-1);
tic,[alphabl2,J,f]=klr_cg(K,Y(:,1),C);toc
Wl2=tprod(X,[1:ndims(X)-1 -ndims(X)],alphabl2(1:end-1),-ndims(X));bl2=alphabl2(end);
fl2=tprod(X,[-(1:ndims(X)-1) 1],Wl2,[-(1:ndims(X)-1)])+bl2;

[f,df,ddf,obj]=L1RegLRFn([Wl2(:);bl2],X,Y(:,1),C);
fprintf('L2) |W|_1=%g  Ed=%g J=%g\n',obj(3),obj(2),obj(1));


% L_1 Reg
tic,[Wb,f,J]=L1RegKLR(X,Y(:,1),C,'objTol0',1e-3,'verb',2);toc
szX=size(X);
Uls = Wb{1:end-1}; bls=Wb{end};
Wlsigma = parafac(Sls,Uls{:});
blsigma = bls;
flsigma = tprod(X,[-(1:ndims(X)-1) 1],Wlsigma,-(1:ndims(Wlsigma)))+blsigma;

[f,df,ddf,obj]=L1RegLRFn([Wlsigma(:);blsigma],X,Y(:,1),C/2);
fprintf('L11) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));
clf;image3d(Wlsigma,3)

% Groupwise L_1 reg
tic,[Wb,f,J]=L1RegKLR(X,Y(:,1),C,'objTol0',1e-3,'verb',2,'grpDim',2);toc % time-grouping

% Groupwise L_1 reg with transformation matrix
W = X(:,:)*X(:,:)'./size(X,3)./size(X,2); [U,S]=eig(W); S=diag(S); si=S>max(S)*1e-3; W=repop(U(:,si),'*',sqrt(1./S(si))');
tic,[Wb,f,J]=L1RegKLR(X,Y(:,1),C,'objTol0',1e-3,'verb',2,'grpDim',2,'Mx',W,'mxDim',1);toc % time-grouping

% multiple-sub-problem groupwise L_1 reg
N = floor(size(Y,1)/2); N=[N size(Y,1)-N];
Ys = [Y(1:N(1),1)           zeros(N(1),1);...
      zeros(N(2),1)         Y(N(1)+1:end,1)];
tic,[Wb,f,J]=L1RegKLR(X,Ys,C,'objTol',1e-3,'verb',2,'grpDim',[2 4]);toc % time+sub-pro grouping


% CV training
foldIdxs=gennFold(Y(:,1),10);
fprintf('Condition number of K: %g\n',rcond(K));
Cscale = .1*sqrt((mean(diag(K))-mean(K(:))));
res=cvtrainFn('L1RegKLR_als',X,Y(:,1),Cscale*10.^[-3:3],foldIdxs,...
              'verb',1,'tol',1e-2,'rank',20,'reuseParms','Wb','hps',{Uls{:} bls});



% Real data
global bciroot; bciroot={'~/data/bci','/media/JASON_BACKU/data/bci','/Volumes/BCI_data/'};
expts      = {'eeg/vgrid/nips2007/1-rect230ms'};
subjects   = {{'jh'} };
labels     = {{{'flip_rc_sep' 'flip_rc_mix' 'flip_opt'}}  };
markerdict = {'non-tgt' 'tgt'};
exi=1;si=1;ci=3;
expt  = expts{exi};
subj  = subjects{exi}{si};
label = labels{exi}{si}{ci};
z=jf_load(expt,subj,label);
z=preproc(z,markerdict);
z  = jf_retain(z,'dim','ch','idx',[z.di(n2d(z.di,'ch')).extra.iseeg],'summary','eeg only');
z  = jf_compressDims(z,'dim',{'epoch' 'letter'});
z.Y= balanceYs(z.Y); % make a balanced problem
z  = jf_retain(z,'dim','epoch','idx',any(z.Y~=0,ndims(z.Y)),'summary','bal pts only');
z.foldIdxs = gennFold(z.Y,10); % setup the folding

varx=mvar(z.X,n2d(z.di,'epoch')); % variance along each dir
Cscale = .1*sqrt(mean(varx(:))); % Cscale is ave var
z = jf_cvtrain(z,'verb',1,'reuseParms',1,'objFn','L1RegKLR','Cs',5.^(-3:5),'Cscale',Cscale,'dim','epoch','seedNm','Wb','reuseParms',0);

