function [wb,f,J,obj,tstep]=lsigmalr_rereg(X,Y,C,varargin);
% Train Logistic Regression classifier with trace-norm regularisation 
%
opts=struct('dim',[],'maxIter',3000,'eta',1,'structMx',[],'objTol',0,'objTol0',[1e-7 1e-2],'tol',0,'tol0',[1e-8 1e-2],'verb',1,'wb',[],'shrink',0,'marate',.7,'symDim',[]);
[opts,varargin]=parseOpts(opts,varargin);

szX=size(X); nd=ndims(X);
dim=opts.dim; if ( isempty(dim) ) dim=ndims(X); end;
objTol0=opts.objTol0; objTol0(end+1:2)=objTol0(end);
tol0   =opts.tol0;    tol0(end+1:2)=tol0(end);
maxIter=opts.maxIter; maxIter(end+1:2)=5;

eta=1e-3;mineta=1e-6;%if ( isa(X,'single') ) mineta=1e-6; else mineta=1e-6; end;
if ( ~isempty(opts.symDim) && size(X,opts.symDim(1))~=size(X,opts.symDim(2)) ) 
  error('symetric dims are not the same size!');
end

lropts=lr_cg([],[],[],'dim',dim,'verb',opts.verb-2,'objTol0',objTol0(2),'maxIter',maxIter(2),'tol0',tol0(2),varargin{:},'getOpts',1);
wb=opts.wb;
if ( isempty(wb) ) 
  % L2 reg call, with reg constant such that it approximates the same l1 cost
  % assume final weight vector is about 1/std long, then C/std = K*C/std.^2 -> K=std
  stdX=.1*sqrt(X(:)'*X(:))./size(X,2);
  [wb,f,Jlr]=lr_cg(X,Y,C*stdX,lropts);
else  
  f  = (wb(1:end-1)'*reshape(X,[],size(X,ndims(X))))'+wb(end);
end
J=inf; owb=2*wb; dw0=0; Ws=[]; Us=[]; ss=[]; Vs=[]; step=1;
for iter=1:maxIter(1);
  oJ=J; 
  W=reshape(wb(1:end-1),size(X,1),size(X,2)); b=wb(end);
  if ( isequal(opts.symDim(:),[1;2]) )
    [U,s]  =eig(W); V=U';  s=diag(s); 
  else
    [U,s,V]=svd(W,'econ'); s=diag(s);
  end
  nrms=abs(s);
  if ( ~isempty(opts.shrink) ) s=max(0,s-opts.shrink); end;
  Ew = sum(nrms);
  %wX = W(:)'*reshape(X,[],size(X,ndims(X)));;
  %dv = wX+b;
  g  = 1./(1+exp(-Y.*f));         % Pr(x|y)
  Ed = -log(max(g,eps))'*(Y.*Y); % -ln P(D|w,b,fp)
  J  = Ed+C*Ew;
  dw = norm(owb-wb)./max(1,norm(wb));
  if ( iter<3 ) eta=max(nrms)/2; J0=J; maJ=max(oJ,J)*2; dw0=max(dw0,dw); end; % initial estimate, so small values don't dominate too early
  %if ( (oJ-J)<1e-3*J0 ) eta=max(eta/2,max(nrms)*mineta); end; % increase exactness of the approx over time
  eta =min(eta,max(nrms)*mineta);
  maJ =maJ*(1-opts.marate)+J(1)*(opts.marate); % move-ave obj est
  madJ=maJ-J; 
  if ( opts.verb>0 ) 
    fprintf('%3d) w=[%s]  J=%5.3f (%5.3f+%5.3f) dJ=%5.3f\tdw=%5.3f\teta=%g\n',iter,sprintf('%5.3f,',s(1:min(end,5))),J,Ew,Ed,madJ(1),dw,eta);
  end
  % convergence testing
  if ( dw<opts.tol(1) || dw<opts.tol0(1)*dw0 || madJ<=opts.objTol(1) || madJ<=opts.objTol0(1)*J0) 
    break; 
  end;

  if ( size(X,1)<=size(X,2) ) % reg is based on smaller of the 2 dims
    R=U*diag(1./(eta+nrms))*U'; % leading dim
    iR=U*diag(eta+nrms)*U'; % inverse reg for opt pre-conditioning
  else
    R=V*diag(1./(eta+nrms))*V'; % trailing dim
    iR=V*diag(eta+nrms)*V';
  end
  if ( 0 )
    if ( size(X,1)<=size(X,2) ) % reg is based on smaller of the 2 dims
      R=eye(size(U,1),size(U,1))*1./eta + U*diag(1./(max(nrms,eta))-1/eta)*U'; % leading dim
    else
      R=eye(size(V,1),size(V,1))*1./eta + V*diag(1./(max(nrms,eta))-1/eta)*V'; % trailing dim
    end
  end

  %N.B. only solve each sub-problem approximately as we're going to update the reg and solve again later anyway
  owb=wb;
  lropts.wb=wb;lropts.step=step;lropts.wPC=iR;
  [wb,f,Jlr,obj,step]=lr_cg(X,Y,C/2*R,lropts);

end
if ( opts.verb>=0 ) 
  fprintf('%3d) w=[%s]  J=%5.3f (%5.3f+%5.3f) dJ=%5.3f\tdw=%5.3f\teta=%g\n',iter,sprintf('%5.3f,',s(1:min(end,5))),J,Ew,Ed,madJ(1),dw,eta);
end

return;
function testCase()
% non-sym pos def matrices
wtrue=randn(40,50); [utrue,strue,vtrue]=svd(wtrue,'econ'); strue=diag(strue);
% sym-pd matrices
wtrue=randn(40,500); wtrue=wtrue*wtrue'; [utrue,strue]=eig(wtrue,'econ'); strue=diag(strue); vtrue=utrue';
% re-scale components and make a dataset from it
strue=sort(randn(numel(strue),1).^2,'descend');
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
X    =Xtrue + randn(size(Xtrue))*sum(strue)/10;
wb0  =randn(size(X,1),size(X,2));

% simple l2
tic,
[wb,f,J]=lr_cg(X,Y,1,'verb',1,'dim',3);  
conf2loss(dv2conf(Y,f))
toc

%low rank
tic,
[wb,f,J]=lsigmalr_cg(X,Y,10,'verb',1,'dim',3);  
W=reshape(wb(1:end-1),size(X,1),size(X,2)); [U,s,V]=svd(W); s=diag(s);
conf2loss(dv2conf(Y,f))
toc
w=reshape(wb(1:end-1),size(X,1),size(X,2));b=wb(end);sigma=svd(w);clf;subplot(221);plot(sigma);subplot(222);plot(log10(sigma+eps));subplot(212);imagesc(w);

% test with re-seeding solutions
Cscale=.1*sqrt(CscaleEst(X,2));
[wb10,f,J] =lsigmalr_cg(X,Y,Cscale,'verb',1,'dim',3,'wb',[]);  
[wb,f,J]=lsigmalr_cg(X,Y,Cscale*1e-1,'verb',1,'dim',3);  
[wb102,f,J]=lsigmalr_cg(X,Y,Cscale*1e-1,'verb',1,'dim',3,'wb',wb10);  

% test the symetric version
[wb,f,J] =lsigmalr_cg(X,Y,10,'verb',1,'dim',3);  
[wbs,f,J]=lsigmalr_cg(X,Y,10,'verb',1,'dim',3,'symDim',[1 2]);  


% compare with als version
[wb,f,J]=lsigmalr_cg(X,Y,1,'verb',1,'dim',3);  
[Wb]=LSigmaRegKLR_als(X,Y,1,'verb',1,'dim',3,'rank',20,'Cscale',1);Wb{1}

% compare on real dataset
z=load('sfToy');oz=z;
z=jf_cov(jf_whiten(z));
jf_cvtrain(z,'objFn','LSigmaRegKLR_als','rank',10,'Cs',5.^(1:2),'reorderC',0,'outerSoln',0,'seedNm','Wb')
jf_cvtrain(z,'objFn','lsigmalr_cg','Cs',5.^(-3:5),'reorderC',0,'outerSoln',0,'seedNm','wb')
