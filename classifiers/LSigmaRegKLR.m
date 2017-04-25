function [Wb,dv,J,K,S]=LSigmaRegKLR(X,Y,C,varargin)
% Spectral Norm regularised KLR -- using the penalty and SVD method
%
% Options:
%  penFactor -- the fraction by which we increase the penalty term when converged (1.4)
%  h         -- the linear->quadratic threshold for the huber L1 approx (1e-4)
%  maxIter   -- the max number of iterations to do (inf)
%  maxEval   -- the max number of evals to do (inf)
%  marate    -- decay constant for the moving average computation for termination tests (.7)
%  tol       -- [2x1] convergence tolerance for [L1 L2] optimisations (1e-8)
%  tol0      -- [2x1] convergence tolerance for [L1 L2] optimisations (0)
%  objTol    -- convergence tolerance for the LSigmaKLR objective (1e-4)
%  objTol    -- convergence tolerance for the LSigmaKLR objective relative to initial obj value (1e-2)
%  penTol    -- [2x1] convergence tolerance for when to increase the penalty factor ([0 .05])
%  Wb        -- [n-d] initial estimate for the solution
%  rank      -- max rank for the decomposotion computation
%  rankTol0  -- tolerance for doing the rank decomposition
opts=struct('kerType','linear','dim',-1,'K',[],...
            'penFactor',1.4,'pen',.1,'h',1e-4,'verb',1,...
            'maxIter',inf,'maxEval',inf,...
            'marate',.7,'tol',1e-8,'tol0',0,'objTol',0,'objTol0',1e-2,...
            'penTol',[0 .05],'Wb',[],'rank',inf,'rankTol0',1e-2);
opts=parseOpts(opts,varargin);

if (opts.dim<0) opts.dim=ndims(X)+opts.dim+1; end;
dim=opts.dim;
if ( isempty(opts.K) ) 
   fprintf('Comp Kernel..');
   K=compKernel(X,[],opts.kerType,'dim',opts.dim); % compute the data kernel
   fprintf('..done\n');
   opts.K=K;
elseif ( ndims(opts.K)==2 && size(opts.K,1)==size(opts.K,2) && size(opts.K,1)==size(X,opts.dim) )
   K=opts.K;
else
   error('Cant compute kernel');
end

trnInd = (Y~=0); N=sum(trnInd);
% Train/test kernel includes the prior position
Ktrn   = [K(trnInd,trnInd) zeros(N,1); zeros(1,N+1)];
Ytrn   = [Y(trnInd);0]; % inc prior as unlabelled pt

rho    = opts.pen;

% extract the training set
szX=size(X); dataDims=setdiff(1:ndims(X),dim(1));
idx={};for d=1:ndims(X); idx{d}=1:szX(d); end;
trnIdx=idx; trnIdx{dim(1)}=trnInd;
Xtrn = X(trnIdx{:});

% Generate an intial solution.
if ( ~isempty(opts.Wb) ) % seed with prior if given
   Wlsigma= reshape(opts.Wb(1:end-1),szX(dataDims)); blsigma=opts.Wb(end);
   % setup the current dual weight values
   alphab=zeros(size(Ktrn,1)+1,1);
   alphab(:)=0; alphab(end-1)=1; alphab(end)=blsigma;
   objl2 = [0 0 0 0];
else
   alphab=[]; Cl2=C;
   [alphab,f,J,objl2] = klr_cg(Ktrn,Ytrn,Cl2,'alphab',alphab,...
                               'verb',opts.verb-2,'tol',1e-2); % L2 regularised solution
   Wlsigma=0;
end

S=[]; W=[]; dW=0; J=1e9; W=Wlsigma; Jp=0;
for iter=1:opts.maxIter; % loop to until converged to Spectral Norm reg soln
   oW = W; oalphab=alphab; oS=S; oJ=J; oJp=Jp;
   
   % Compute the SVD
   Wl2    = tprod(Xtrn,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],alphab(1:end-2),[-dim(1)]) ...
            + Wlsigma*alphab(end-1);
   if ( ndims(Wl2)==2 )
      [U{1},S,U{2}]= svd(Wl2,'econ'); S=diag(S);
   else
      % seed with the previous decomposition to speed things up
%       if ( iter>1 )
%          [S,U{1:ndims(Wl2)}]=parfacProj(Wl2,U{:});
%          [ans,si]=sort(abs(S),'descend'); S=S(si);
%          si=si(1:sum(cumsum(abs(S))./sum(abs(S))<.95)); % only best 90%
%          S=S(1:numel(si)); for d=1:numel(U); U{d}=U{d}(:,si); end; % sub-select
%          [S,U{1:ndims(Wl2)}]=parfac_als(Wl2,S,U,'rank',opts.rank,'verb',opts.verb-2,'C',1e-3);
%       else
         [S,U{1:ndims(Wl2)}]=parfac_als(Wl2,'rank',opts.rank,'verb',opts.verb-2,'C',1e-3);
%       end
      [S,U{1:ndims(Wl2)}]=parfacProj(Wl2,U{:});
   end
   if ( iter==1 ) origS = S; origW=Wl2; end;
   
   % Report the L2 results
   Jl2 = C*sum(abs(S))+objl2(2);
   if ( opts.verb > 1 || (opts.verb>0 && iter==1) ) 
      fprintf('\n%d.L2) Pen=%5f Jp=%g  |W|_sigma= %g  Ed=%g J= %g\n',...
              iter,rho,Jp,sum(abs(S)),objl2(2),Jl2);
   end

   %-------------------------------------------------------------
   % Optimise the L1Reg LR for this penalty parameter
   % a) Convert to a simple linear problem space
   ds=setdiff(1:ndims(X),dim(1));
   d=ds(1);
   phiX = tprod(Xtrn,[1:d-1 -d d+1:ndims(X)],U{d},[-d ndims(X)+1]); % acc 1st and add rank dim
   for d=ds(2:end); % accum out all but trials dim, and rank dim
      phiX=tprod(phiX,[1:d-1 -d d+1:ndims(X) ndims(X)+1],U{d},[-d ndims(X)+1]); 
   end;
   phiX = msqueeze(ds,phiX)'; % make [rank x N]
   wb = [S; alphab(end)];
   % dynamically compute h
   opts.h = min(opts.h,max(abs(S))*1e-2);
   % do the L1 reg optimisation
   [wb nfs]=nonLinConjGrad(@(w) primalL1RegLRFn(w,phiX,Ytrn(1:end-1),C,opts.h),...
                           wb,'verb',opts.verb-2,'maxEval',opts.maxEval(1),'tol',opts.tol(1));   
   % record the final performance
   [J1,df1,ddf1,objl1]=primalL1RegLRFn(wb,phiX,Ytrn(1:end-1),C,0);      
   J = C*sum(abs(wb(1:end-1)))+objl1(2);
   if ( opts.verb > 0 ) 
      fprintf('%d.L1) Pen=%5f Jp=%g dW=%g |W|_sigma= %g  Ed=%g J= %g\n',...
              iter,rho,Jp,dW,sum(abs(wb(1:end-1))),objl1(2),J);
   end
   
   S=wb(1:end-1);
   if ( ndims(Wl2)==2 ) % new prior
      Wlsigma = U{1}*diag(S)*U{2}';  % faster and less memory
   else
      Wlsigma = parafac(S,U{:});
   end
   b = wb(end);
   % plot([S wb(1:end-1)])
   % mimage(W,Wlsigma,'diff',1,'clim','minmax'); % plot the changes
   % plot([f [fX'*wb(1:end-1)+wb(end);0] Ktrn*alphab(1:end-1)+alphab(end)]);
   
   % convergence test
   W=Wlsigma;
   if( iter==1 ) madJ=J; J0=J;
   else
      madJ=madJ*(opts.marate)+abs(oJ-J)*(1-opts.marate); % move-ave grad est
   end
   if ( madJ<=opts.objTol || madJ./J0 < opts.objTol0 ) % term tests
      break;
   end; 
   
   %-----------------------------------------------------------------------
   % Optimise the l2 approximation using the kernel method
   % update kernel matrix
   Kprior = compKernel(Xtrn,Wlsigma,opts.kerType,'dim',opts.dim);
   Ktrn(1:end-1,end)=Kprior; Ktrn(end,1:end-1)=Kprior; 
   Ktrn(end,end)=compKernel(Wlsigma,[],'linear','dim',opts.dim);

   mu     = [zeros(sum(trnInd),1);-1];
   
   % 1) Solve the KLR problem with this prior
   alphab(:)=0; alphab(end-1)=1; alphab(end)=b;
   % plot([f [phiX'*S+1;0] Ktrn*alphab(1:end-1)+alphab(end)]); % plot diff objs
   % L2 regularised solution   
   % we include 2 terms:
   %  a) the penalty factor to link the 2 objective functions
   %  b) a l2 approximation to the L_sigma loss such that the gradients match at 
   %     the current point
   W2=Wlsigma(:)'*Wlsigma(:); % the L2 norm of the current solution
   Lsigmal2=C*sum(abs(wb(1:end-1)))/W2/2; % L1/L2 norm ratio
   Cl2 = Lsigmal2*[rho+1; 2*rho];
   [alphab,f2,J2,objl2] = ...
       priorklr_cg(Ktrn,Ytrn,Cl2,mu,'alphab',alphab,...
                   'verb',opts.verb-2,'maxEval',opts.maxEval(min(2,end)),'tol',opts.tol(min(2,end))); 

   Jp = objl2(2)+rho*objl2(3); % true penalty function objective value
   % plot([oalphab alphab])

   %fprintf('\t Jp=%g\n',Jp);
   
   %--------------------------------------------------------------------------
   % Penalty update
   % if converged update the penalty parameter
   dW=norm(W(:)-oW(:)); dJp=Jp-oJp;
   if ( abs(dJp)<=opts.penTol(min(2,end)) ) 
      rho=rho*opts.penFactor;
   end
end

% compute the final solution and prediction
% Compute the SVD
dv   = tprod(X,[-(1:dim(1)-1) 1 -(dim(1):ndims(X)-1)],W,-(1:ndims(W)))+b;
g    = 1./(1+exp(-(Y(trnInd).*dv(trnInd)))); g=max(g,eps);
Ed   = -sum(log(g));  % -ln(P(D|w,b,fp))
Ew   = sum(abs(S));   % L_sigma
R    = sum((1-cumsum(abs(S))./sum(abs(S)))>.01)+1; % rank is 99% size
J    = Ed + C(1)*Ew;
if ( opts.verb >= 0 ) 
   fprintf('\n%d) C=%5f  Pen=%g  R=%d  |W|_sigma=%g  Ed=%g  J=%g ',iter,C,rho,R,Ew,Ed,J);
end
Wb=[W(:);b];

return;
%-----------------------------------------------------------------------------
function testCase()
C=1;
N=400; L=2;
fs = 128; T=3*fs;
Yl = (randn(N,1)>0)+1;     % True labels + 1
Y  = double([Yl==1 Yl==2])*2-1;% indicator N x L
sources = { {'sin' 5} {'coloredNoise' 1};
            {'sin' 8} {'coloredNoise' 1};
            {'none' 1} {}}; % rest just detectors
y2mix=cat(3,[1 1;.5 1],[.5 1;1 1]); % ch x source x label
mix  =y2mix(:,:,Yl);                % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T);
% remove the sources
X=X(3:end,:,:); elect_loc=elect_loc(:,3:end);

% cov decomp
oX=X; X = tprod(X,[1 -2 3],[],[2 -2 3]);

% 3d decomp
oX=X;
[start width]=compWinLoc(size(X,2),[],.5,[],50);
X = windowData(X,start,width,2);
X = fft_posfreq(X,[],2); % fourier trans
X = tprod(real(X),[1 3 4 5],[],[2 3 4 5]) + tprod(imag(X),[1 3 4 5],[],[2 3 4 5]);
X = msqueeze(4,sum(X,4));


clf;image3d(X,1,'plotPos',elect_loc,'colorbar','ne','xlabel','ch','ylabel','ch2','zlabel','epoch')


K=compKernel(X,[],'linear','dim',-1);

% L2 reg
tic,[Wb,J,f]=klr_cg(K,Y(:,1),C);toc
Wl2=tprod(X,[1:ndims(X)-1 -ndims(X)],Wb(1:end-1),-ndims(X));bl2=Wb(end);
fl2=tprod(X,[-(1:ndims(X)-1) 1],Wl2,[-(1:ndims(X)-1)])+bl2;

[f,df,ddf,obj]=LSigmaRegLRFn([Wl2(:);bl2],X,Y(:,1),C);
fprintf('L2) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));

clf;image3d(Wl2,1,'plotPos',elect_loc,'colorbar','ne')

W=Wl2;b=bl2;
[U,S,V]=svd(W,'econ');S=diag(S);
figure(1);clf;fact3Dplot(U(:,1:2),V(:,1:2)*diag(S(1:2)),[],elect_loc);

% LSigma Reg
tic,[Wb,f,J,K,S]=LSigmaRegKLR(X,Y(:,1),C,'objTol0',1e-3,'K',K,'verb',2);toc
szX=size(X);
Wlsigma = reshape(Wb(1:end-1),szX(1:end-1));blsigma=Wb(end);
flsigma = tprod(X,[-(1:ndims(X)-1) 1],Wlsigma,-(1:ndims(Wlsigma)))+blsigma;

[f,df,ddf,obj]=LSigmaRegLRFn([Wlsigma(:);blsigma],X,Y(:,1),C);
fprintf('LSigma1) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));

W=Wlsigma;b=blsigma;
[U,S,V]=svd(W,'econ');S=diag(S);
figure(2);clf;fact3Dplot(U(:,1:2),V(:,1:2)*diag(S(1:2)),[],elect_loc);

clf;image3d(Wlsigma,1,'plotPos',elect_loc,'colorbar','nw')

% Alt soln method
[U,S,V]=svd(Wlsigma,'econ');S=diag(S); sgn=sign(S);
[hps0,fat]=mkfat('rw',repop(U,'*',sqrt(S)'),'cw',repop(V,'*',sqrt(S)'),'b',blsigma(end)); 
hps=hps0;
[f2,df2,ddf2,objl2]=LRMxFactFn(hps0,fat,X,Y(:,1),C/2);
tic,
   hps=nonLinConjGrad(@(w) LRMxFactFn(w,fat,X,Y(:,1),C/2),...
                   hps0,'verb',1,'maxEval',1000,'maxIter',inf,'tol',1e-3);
toc
Wmf=hps(fat.rw)*hps(fat.cw)'; bmf=hps(fat.b);
fmf=tprod(X,[-1 -2 1],Wmf,[-1 -2])+bmf;

W=Wmf;
[U,S,V]=svd(W,'econ');S=diag(S);
figure(3);clf;fact3Dplot(U(:,1:2),V(:,1:2)*diag(S(1:2)),[],elect_loc);

[f,df,ddf,obj]=LSigmaRegLRFn([Wmf(:);bmf],X,Y(:,1),C);
fprintf('LSigma1) |W|_sigma= %g  Ed=%g J= %g\n',obj(3),obj(2),obj(1));


% comparsion with the other ways to compute this type of bi-linear decomposition
% symetric matrix version
[U,S,V]=svd(Wlsigma,'econ');S=diag(S); 
sgn=sign(sum(U.*V)); sgn=sgn(:); % correct for sign differences
[hps0,fat]=mkfat('rw',repop(U,'*',sqrt(S)'),'b',blsigma); hps=hps0;
[f,df,ddf]=LRSymMxFactFn(hps0,fat,XX,Y(:,1),C/2,sgn);
hps=nonLinConjGrad(@(w) LRSymMxFactFn(w,fat,XX,Y(:,1),C/2,sgn),...
                   hps0,'verb',1,'maxEval',5000,'maxIter',inf);
rw=hps(fat.rw);
Wmfsym=rw*diag(sgn)*rw'; bmfsym=hps(fat.b);
fmfsym=tprod(XX,[-1 -2 1],Wmfsym,[-1 -2])+bmfsym;

% re-optimise the alt soln methods solution
[U,S,V]= svd(W); S=diag(S);
phiX = tprod(U,[-1 1],tprod(XX,[1 -2 3],V,[-2 2]),[-1 1 2]);
wb = [S; alphab(end)];
[wb nfs]=nonLinConjGrad(@(w) primalL1RegLRFn(w,phiX,Y(:,1),C),wb,'verb',1,'maxEval',500);

% Eval a new solution
f1 = tprod(XX,[-1 -2 1],W,[-1 -2])+b;
[U,S,V]= svd(W); S=diag(S);
phiX = tprod(U,[-1 1],tprod(XX,[1 -2 3],V,[-2 2]),[-1 1 2]);
wb = [S; b];
[J1,df1,ddf1,objl1]=primalL1RegLRFn(wb,phiX,Y(:,1),C,0);
f11= wb(1:end-1)'*phiX+wb(end);
fprintf('%d.b) |W|_sigma= %g  Ed=%g J= %g\n',0,sum(abs(wb(1:end-1))),objl1(2),J1);


% CV training
foldIdxs=gennFold(Y(:,1),10);
K   = compKernel(X,[],'linear','dim',-1);
fprintf('Condition number of K: %g\n',rcond(K));
Cscale = .1*sqrt((mean(diag(K))-mean(K(:))));
res=cvtrainFn('LSigmaRegKLR',X,Y(:,1),Cscale*10.^[-3:3],foldIdxs,...
              'verb',1,'tol',1e-2,'K',K,'dim',-1,'reuseParms','Wb');
