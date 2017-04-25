function [wb,f,J,obj]=l1lr_cg(X,Y,C,varargin);
%  structMx   -- [nFeat x nReg] matrix which shows how the different features combine to 
%                   make a groupwise L1 structured regularisor           ([])
%                OR
%                 'ascending','descending','ascend+descend'
%               for ideas on how to use this structure matrix to impose structure on the solution see:
%                   Bach, Francis, Rodolphe Jenatton, Julien Mairal, and Guillaume Obozinski. 2011. 
%                   “Structured sparsity through convex optimization.” 1109.2397 (September 12). 
%                   http://arxiv.org/abs/1109.2397.
opts=struct('dim',[],'maxIter',3000,'eta',1e-2,'structMx',[],'objTol0',[1e-6 1e-1],'tol0',[1e-5 1e-1],'verb',1,'wb',[],'etaLR',.2,'zeroStart',0);
[opts,varargin]=parseOpts(opts,varargin);
szX=size(X); X=reshape(X,[],size(X,ndims(X))); % get orginal size, then reshape into [feat x examples]

dim=opts.dim; if ( isempty(dim) ) dim=ndims(X); end;
objTol0=opts.objTol0; objTol0(end+1:2)=objTol0(end);
tol0   =opts.tol0;    tol0(end+1:2)=tol0(end);
maxIter=opts.maxIter; maxIter(end+1:2)=5;
if ( isa(X,'single') ) mineta=1e-8; else mineta=1e-10; end; mineta=1e-2;
eta=1e-6;

structMx=opts.structMx; 
if ( ~isempty(structMx) && (isstr(structMx) || numel(structMx)==1) )
  structMx=mkStructMx(szX(1:end-1),structMx);
end
if ( ndims(structMx)>2 ) 
  structMx=reshape(structMx,[],size(structMx,ndims(structMx))); % work with vector X
  %if ( sum(structMx(:)==0)>numel(structMx)/2 ) structMx=sparse(structMx); end;
end

% est lipsich constant/largest eig of hessian
w=X(:,1); l=sqrt(w'*w); for i=1:5; w=X*(w'*X)'/l; l=sqrt(w'*w); end; rho=l;

wb=opts.wb; seedwb=~isempty(wb);
if ( isempty(wb) ) wb=zeros(size(X,1)+1,1); end;

if ( isempty(structMx) )
  nrm=abs(wb(1:end-1));
  R  = (1./(nrm(:)+eta))/2;  %R  = (1./(nrm+median(eta)))/2;
else
  nrm= sqrt((wb(1:end-1).^2)'*structMx);
  R  = structMx*(1./nrm(:)+eta);     %R  = structMx*(1./(nrm(:)+mean(eta)));     
end

nFeat=[];
if ( C<0 )   
  nFeat=min(-C); if ( ~isempty(structMx) ) nFeat=min(nFeat,size(structMx,2)); end;

  % initial est for C for this number of features...
  dLg=-X*Y(:)/2; % loss gradient assuming all zeros initial weight vector
  dRg=ones(size(dLg));%abs(sign(wb(1:end-1)))+(wb(1:end-1)==0);
  %dRg=abs(2.*R.*wb(1:end-1))+(abs(wb(1:end-1))<mineta); % with the new R
  if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
    dRg = (double(dRg)'*structMx)';
    dLRg= sqrt((double(dLg(:)).^2)'*structMx)'./dRg(:); %N.B. only if all elm in group have same weight in structMx
    dLg = dLRg;
  else
    dLRg = abs(dLg);
    dRg  = ones(size(dLg));
  end
  [sdLRg]=sort(dLRg,'descend'); 
  
  % dLg = -X*Y(:)/2; % loss gradient assuming all zeros initial weight vector
  % dRg = ones(size(dLg));
  % if ( ~isempty(structMx) )  dLg=(dLg'*structMx)'; dRg=(dRg'*structMx)'; end
  % [sdLRg,si]=sort(abs(dLg./dRg),'descend'); 
  C=sdLRg(min(end,nFeat+[0 1]))'*[1e-5;1-1e-5];   
  %C=sqrt(X(:)'*X(:)/size(X,dim));  % initial C is 'optimal' C
end;

% pre-fetch opts for the lr calls
lropts=lr_cg([],[],[],'dim',dim,'verb',opts.verb-2,'objTol0',objTol0(2),'maxIter',maxIter(2),'tol0',tol0(2),varargin{:},'getOpts',1);
% wb=opts.wb; seedwb=~isempty(wb);
% if ( isempty(wb) )
%   % L2 reg call, with reg constant such that it approximates the same l1 cost
%   % assume final weight vector is about 1/std long, then C/std = K*C/std.^2 -> K=std
%   stdX=.05*sqrt(X(:)'*X(:))./size(X,2); 
%   [wb,f,Jlr]=lr_cg(X,Y,C.*R*stdX,lropts);%'verb',opts.verb-1,'objTol0',objTol0(2),'maxIter',maxIter(2),'dim',dim,varargin{:});
% else
%   if ( numel(wb)==size(X,1) ) wb=[wb(:);0]; end;
%   if ( isempty(structMx) )
%     nrm=abs(wb(1:end-1));
%     R  = (1./max(nrm(:),median(eta)))/2;  %R  = (1./(nrm+median(eta)))/2;
%   else
%     nrm=sqrt((W(:).^2)'*structMx);
%     R  = structMx*(1./max(nrm(:),median(eta)));     %R  = structMx*(1./(nrm(:)+mean(eta)));     
%   end
%   lropts.wb=wb;
%   [wb,f,Jlr]=lr_cg(X,Y,C.*R,lropts); % re-train to get a new scaling...
%   wX = wb(1:end-1)'*reshape(X,[],size(X,ndims(X)));;
%   f  = wX'+wb(end);
% end
W  = wb(1:end-1); b=wb(end);
wX = wb(1:end-1)'*X;
f  = wX'+wb(end);
if ( isempty(structMx) ) % true cost of this solution
  nrm=abs(W);
else
  nrm=sqrt((W(:).^2)'*structMx);
end
eta=abs(wb(1:end-1));
% if ( opts.zeroStart );%&& seedwb ) % get the initial variational bound
%   wb(:)=0; f(:)=0;
% end
% if( seedwb ) eta=eta; end;
%rho=X(:)'*X(:)./size(X,1);
J=inf; step=1; zeroFeat=false(size(wb)); actFeat=sum(wb~=0);%nrm=[];
for iter=1:opts.maxIter;
  oJ=J; onrm=nrm; oeta=eta;
  % proximal step
  g  = 1./(1+exp(-Y(:).*f(:)));         % Pr(x|y)
  Yerr =Y(:).*(1-g(:));
  dR = 2*C.*R.*wb(1:end-1);
  dL = [-X*Yerr;-sum(Yerr)];
  %wb = wb - dL/rho; % simple gradient step -- avoid stuck due to eta
  %wb = sign(wb).*max(0,(abs(wb)-C/rho));

  W  = wb(1:end-1); b=wb(end);
  if ( isempty(structMx) ) % true cost of this solution
    nrm=abs(W);
  else
    nrm=sqrt((W(:).^2)'*structMx);
  end
  Ew = sum(nrm(:));
  % wX = W(:)'*X;
  % dv = wX+b;
  g  = 1./(1+exp(-Y(:).*f(:)));         % Pr(x|y)
  Ed = sum(-log(max(g(:),eps))) ; % -ln P(D|w,b,fp)
  J  = Ed+C*Ew;
  
  % update the l1 stabilisation
  if ( iter<2 )
    J0=J;
  else 
    % set eta such that the worse case expected step size has the correct l1 regularisor.
    %eta=abs(wb(1:end-1))*(1-opts.etaLR) + eta*opts.etaLR; % exp move ave est of step size times 2 for each norm
    eta=1e-10;
  end;
  eta=max(mineta*10./rho,eta);
  if ( opts.verb>0 ) 
    fprintf('%3d)\tw=[%s] |w|=%d (%d)\t\tJ=%5.3f\t(%5.3f+%5.3f)\teta=%5.3g\tC=%5.3f\n',iter,sprintf('%5.3f,',W(1:min(end,4))),sum(actFeat),sum(~zeroFeat),J,Ew(1:min(end,3)),Ed,median(eta(:)),C);
  end
  % convergence testing
  if ( abs(oJ-J)<opts.objTol0(1)*J0) 
    break; 
  end;
  % compute variational approx to the desired regularisor
  if ( isempty(structMx) )
    R  = (1./(nrm(:)+eta))/2;
  else
    R  = structMx*(1./(nrm(:)+median(eta)));
    % ? can we make a good inverse reg for pre-conditioner?
  end
  
  oC=C;
  dLg=abs(dL(1:end-1))+2*rho*min(abs(wb(1:end-1)),mean(abs(wb(1:end-1))));% 2nd order approx est of gradient if this feature had value=0% abs(wb(1:end-1));%
  %dRg=ones(size(dLg));%abs(sign(wb(1:end-1)))+(wb(1:end-1)==0);
  dRg=abs(2.*R.*wb(1:end-1))+(abs(wb(1:end-1))<mineta); % with the new R
  if ( ~isempty(structMx) )  % est feature gradient for l1/l2 reg
    dRg = (double(dRg)'*structMx)';
    dLRg= sqrt((double(dLg(:)).^2)'*structMx)'./dRg(:); %N.B. only if all elm in group have same weight in structMx
    dLg = dLRg;
  else
    dLRg = abs(dLg);
    dRg  = ones(size(dLg));
  end
  if ( ~isempty(nFeat) ) % est what correct regularisor should be
    [sdLRg]=sort(dLRg,'descend'); 
    % target C is just big enough to prevent nFeat+1 features..
    if ( iter<=2 ) cFF=0; elseif( iter<200 ) cFF=.1; else cFF=.95; end;
    C=(C*(cFF)+sdLRg(min(end,nFeat+[0 1]))'*[.5;.5]*(1-cFF)); % smooth out sudden value changes
    C=max(C,rho*1e-8); % stop C=0 problems
  end
  zeroFeat = C*abs(dRg)>abs(dLg)*(1+1e-4);

  % if ( ~isempty(nFeat) && iter>1 ) % est what correct regularisor should be
  %   % 2nd order approx for the gradient if this feature is 0.
  %   %  linear term from about + (limited) 2nd order term from the estimated hessian component
  %   %  N.B. hessian contribution is limited as it is unlikely to be right very far from the current value...
  %   dLg =abs(dL(1:end-1)) + 2*rho*min(abs(wb(1:end-1)),mean(abs(wb(1:end-1))));
  %   dRg =abs(2.*R.*wb(1:end-1))+(abs(wb(1:end-1))<mineta); % with the new R
  %   if ( ~isempty(structMx) )  dLg=(dLg'*structMx)'; dRg=(dRg'*structMx)'; end
  %   [sdLRg,si]=sort(dLg./dRg,'descend'); 
  %   C=(C*.4+sdLRg(min(end,nFeat+[0 1]))'*[1e-5;1-1e-5]*.6); % smooth out sudden value changes
  %   C=max(C,rho*1e-6);
  %   %clf;plot([[wb(1:end-1) abs(dRg) abs(dLg(1:end-1))]./norm(wb(1:end-1)) zeroFeat]);legend('wb','dR','dL','zero');
  %   zeroFeat = dRg>dLg*(1+1e-4);
  % end  
  actFeat=(abs(W(:))>max(abs(W(:)))*1e-4);
  
  % N.B. only solve each sub-problem approximately as we're going to update the reg and solve again later anyway
  owb=wb;
  lropts.wb=wb; lropts.step=step; % use structure as input to avoid the option parsing overhead
  [wb,f,Jlr,obj,step]=lr_cg(X,Y,C.*R,lropts);%'dim',dim,'verb',opts.verb-2,'wb',wb,'objTol0',objTol0(2),'maxIter',maxIter(2),'step',step,varargin{:});
  f = (wb(1:end-1)'*X)' + wb(end);
end
return;
function testCase()

%Make a Gaussian balls + outliers test case
[X,Y]=mkMultiClassTst([zeros(1,47) -1 0 zeros(1,47); zeros(1,47) 1 0 zeros(1,47); zeros(1,47) .2 .5 zeros(1,47)],[400 400 50],1,[],[-1 1 1]);[dim,N]=size(X);
wb0=randn(size(X,1)+1,1);

%Make a chxtime test case
z=jf_mksfToy(); X=z.X; Y=z.Y;

% simple l2
tic,
[wb0,f0,J0]=lr_cg(X,Y,1,'verb',1);  
toc

% simple l1
tic,
[wb,f,J]=l1lr_cg(X,Y,1,'verb',1,'maxIter',20);
toc
szX=size(X); W=reshape(wb(1:end-1),[szX(1:end-1) 1]);

wb = l1lr_cg(X,Y,-10); % with target num features
% with a seed solution to test the C determination
wb = l1lr_cg(X,Y,-10,'wb',wb0); % with target num features


% test l1/l2 regularisor
if ( ndims(X)<3 ) 
  structMx=mkStructMx([4 size(X,1)/4],1); structMx=reshape(structMx,[size(X,1) size(structMx,3)]);
else 
  szX=size(X);
  structMx=mkStructMx(szX(1:end-1),1); 
end
tic,[wb,f,J]=l1lr_cg(X,Y,1,'verb',1,'structMx',structMx);toc  
W=reshape(wb(1:end-1),[szX(1:end-1) 1]);clf;imagesc(W);
tic,[wb,f,J]=l1lr_cg(X,Y,-2,'verb',1,'structMx',structMx);toc  

% test with covariance and sensor selection
C=tprod(X,[1 -2 3],[],[2 -2 3]);
structMx=mkStructMx(size(C),'covCh');
tic,[wb,f,J]=l1lr_cg(C,Y,1,'verb',1,'structMx',structMx,'zeroStart',0);toc
W=reshape(wb(1:end-1),[size(C,1),size(C,2)]);clf;imagesc(W);

% convex regions selection
structMx=mkStructMx(size(X,1),'ascend+descend');
tic,[wb,f,J]=l1lr_cg(X,Y,1,'verb',1,'structMx',structMx,'zeroStart',0);toc

clf;subplot(211);plot(wb);subplot(212);plot(log10(abs(wb)))

% test with re-seeding solutions
[wb,f,J]  =l1lr_cg(X2d,Y,.01,'verb',1,'dim',3);  
[wb10,f,J]=l1lr_cg(X2d,Y,10,'verb',1,'dim',3,'wb',wb);  
