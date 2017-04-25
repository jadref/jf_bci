function [C]=covEigTrans(C,type,param,verb)
% transform the eigenvalue structure of a set of covariance matrices, e.g. by log
% 
%  [nC]=covEigTrans(C,type,param)
% 
%  nC(:,:,i) = U_i * diag(f(\lambda)) * U_i';
%  where, [U,\lambda]=eig(C(:,:,i)) and f(.) is the selected transformation
%
% Inputs:
%  C -- [d x d x N] set of covariance matrices
%  type -- {'str'}, one of:
%          'sphere' -- spherical transform, \lambda' = 1
%          'sqrt'   -- square root: \lambda' = sqrt(\lambda)
%          'sqr'    -- square :     \labmda' = \lambda^2
%          'dsqrt'  -- square root of diagnol entries = C ./ sqrt(diag(C))
%          'log'    -- log:         \lambda' = log(\lambda)
%             N.B. param = min eig value, param<0 -> min is rel to param'th weakest component on all data
%          'exp'    -- expionential:\lambda' = exp(\lambda)
%          'pow'    -- raise to power:\lambda' = \lambda.^(param(1))
%          'ridge'  -- add ridge of strength param of this strength
%             param > 0   - literial ridge strength
%             -1<param <0 - fraction of full eigen-spectrum (larger is stronger)
%             param<-1    - ridge strength is the strength of the param'th *strongest* component on
%                               *all* the data! (larger is stronger)
%          'keep'   -- keep strongest param(1) eigenvalues
%             N.B. param<0 -> keep up to the param(1)'th *weakest* component
%          'remove' -- keep weakest param(1) eigenvalues
%          'none'   -- do nothing
%          'oas'    -- Orcale Optimal Shrinkage (see [2]) param=n
%          'rblw'   -- Rao-Blackwell Lediot-Wolf (see [2]) param=n
%  param -- parameter to pass to the different transformation
%  verb  -- [int] verbosity level                           (0)
%
% Refs:
%  [1] Chen, Yilun, Ami Wiesel, Yonina C. Eldar, and Alfred O. Hero. “Shrinkage Algorithms for MMSE Covariance Estimation.” Signal Processing, IEEE Transactions on 58, no. 10 (2010): 5016–29.
%
% TODO: [] OAS and RBLW versions without eigen-decomposition
% 
if( nargin<3 ) param=[]; end;
if( nargin<4 || isempty(verb) ) verb=1; end;
szC=size(C);
if ( szC(1)~=szC(2) ) error('Not symetric covariance matrix input!'); end;
if ( ~iscell(type) ) type={type}; end;

% ridge is relative to full-data eig-spectrum
if ( any(strcmp(type{1},{'ridge','log','sqrt',})) && numel(param)>0 && param(1)<0 ) 
  Call = sum(C(:,:,:),3)./prod(szC(3:end));
  sall = eig(Call);
  [ans,si]=sort(abs(sall),'ascend'); sall=sall(si); % get decreasing order
  % remove the rank defficient/non-PD entries
  si = sall>eps & ~isnan(sall) & abs(imag(sall))<eps; sall=sall(si);
  if ( abs(param(1))<1 )% fraction of the spectrum
	 param(1) = sall(round(numel(sall)*abs(param(1))));
  else % number of components back in the eigen-spectrum
	 param(1) = sall(abs(param(1)));
  end
end

type=lower(type);
% in-place transformation
isreal = all(imag(C(:))<eps);
if ( verb>0 ) fprintf('covEigTrans:'); end;
for ci=1:prod(szC(3:end));
  if ( verb>0 ) textprogressbar(ci,prod(szC(3:end))); end;

  Cci=C(:,:,ci);

  % transformations which can efficiently performed in-place
  switch ( type{1} )
    case 'ridge';
      C(:,:,ci) = C(:,:,ci) + param(1)*eye(size(C(:,:,ci)));
      continue;
    case 'dsqrt'; % directly re-scale the rows % BODGE: apply directly to re-scale the rows
     tmp=sqrt(diag(Cci)); if ( tmp<eps ) tmp=1; end;
     C(:,:,ci) =  repop(Cci,'/',tmp);
     continue;
   case 'ddsqrt'; % directly jointly re-scale rows and cols
     tmp=sqrt(diag(Cci)); if ( tmp<eps ) tmp=1; end;
     tmp=sqrt(tmp);
     C(:,:,ci) =  repop(repop(Cci,'/',tmp),'/',tmp');
     continue;
   case 'dridge'; % ridge towards the diagonal entries
     C(:,:,ci) = (1-param(1)) * C(:,:,ci) + param(1)*diag(diag(Cci));
     continue;
  end
  
  %-----------------------------------------------------------------------------------------
  % re-scaling which require eigen-decomposition from here on
  
  % actual eigen transofrmation
  [U,s]=eig(Cci); s=diag(s); si=true(numel(s),1);
  if ( isreal ) U=real(U); end;
  for ti=1:numel(type);
    switch ( type{ti} ) 
     case {'sphere','wht','whiten'}; s(:)=1;
     case 'sqrt';   
       if ( isempty(param) ) param=1e-6*max(abs(s)); end;
       si=si & ~(s<param(1) | s==0 | isnan(s) | isinf(s) | imag(s)>eps);
       s(si)=sqrt(real(s(si)));
     case 'sqr';   
       s(si)=real(s(si)).*real(s(si));
     case 'pow';   
       if ( isempty(param) ) param=2; end;
       s(si)=real(s(si)).^param(1);
     case 'log';
       if ( isempty(param) ) param=1e-6*max(abs(s)); end;
       si=si & ~(s<param(1) | s==0 | isnan(s) | isinf(s) | imag(s)>eps); 
       s(si)=log(real(s(si))); % log -- avoiding log of 0
     case 'exp';
       s(si)=exp(real(s(si))); % exp
     case 'keep';
		 if (any(param<0) ) param(param<0)=numel(s)+param(param<0); end; % neg count back from end
       if ( numel(param)==1 )[ans,sorti]=sort(abs(s),'descend');  rmv=sorti(param(1)+1:end);
       else                  [ans,sorti]=sort(real(s),'descend');	rmv=sorti([param(1)+1:end-param(2)+1]);
       end
       si(rmv)=false;
     case 'remove';   
		if (any(param<0) ) param(param<0)=numel(s)+param(param<0); end; % neg count back from end
      [ans,sorti]=sort(abs(s),'descend');  rmv=sorti(param(1):end); 
      si(rmv)=false;
     case {'oas','rblw'};
       if ( isempty(param) ) error('Must specify the n for this number examples'); end;
       n=param(1); p=numel(s);
       if ( isempty(param) || numel(param)<2 ) param(2)=1e-6*max(abs(s)); end;
       si=si & ~(s < param(2) | s==0 | isnan(s) | isinf(s) | imag(s)>eps);
       if ( any(imag(s)>eps) ) U=real(U); end; % ensure U is pure real...
       Uest  = p*sum(real(s(si)).^2)/(sum(real(s(si))).^2)-1;
       if ( strcmp(type{ti},'oas') )     alpha = 1   /(n+1-2/p);  beta=(p+1)      /((n+1-2/p)*(p-1));
       elseif( strcmp(type{ti},'rblw') ) alpha =(n-2)/(n*(n+2));  beta=((p+1)*n-2)/(n*(n+2)*(p-1));
       end
       rho   = min( alpha+beta/Uest, 1 );
       s(si) = (1-rho) * abs(s(si)) + rho*sum(real(s(si)))./p;
       s(~si)= rho*sum(real(s(si)))./p; 
       si    = true(numel(s),1); % N.B. ridge applied to *all* eigenvalues
     case 'none';
     otherwise;
      warning(sprintf('Unrecognised eigTranstype : %s\n',type));
    end  
  end
  C(:,:,ci) = U(:,si)*diag(s(si))*U(:,si)';
end
if ( verb>0 ) fprintf('\n'); end;
return;
function testCase()
X=randn(10,1000);
C=X*X'./size(X,2);
clf; plot(sort(eig(C),'descend'),linecol(),'DisplayName','raw'); hold on;
nC=covEigTrans(C,'sphere'); plot(sort(eig(nC),'descend'),linecol(),'DisplayName','sphere');
nC=covEigTrans(C,'sqrt'); plot(sort(eig(nC),'descend'),linecol(),'DisplayName','sqrt');
nC=covEigTrans(C,'dsqrt'); plot(sort(eig(nC),'descend'),linecol(),'DisplayName','dsqrt');
nC=covEigTrans(C,'log'); plot(sort(eig(nC),'descend'),linecol(),'DisplayName','log');
nC=covEigTrans(C,'keep',4); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','keep4');
nC=covEigTrans(C,'keep',[2 2]); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','keep2,2');
nC=covEigTrans(C,'ridge',.2); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','ridge.2');
nC=covEigTrans(C,'ridge',-2); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','ridge-2');
% cascade of transformations
nC=covEigTrans(C,{'ridge' 'sqrt'},.2); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','ridge.2+sqrt');
% optimal shrinkage stuff
nC=covEigTrans(C,'oas',1000); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','ridge.2+sqrt');
nC=covEigTrans(C,'rblw',1000); plot(sort(real(eig(nC)),'descend'),linecol(),'DisplayName','ridge.2+sqrt');

legend('show')

% test with singular inputs
[U,s]=eig(C);s=diag(s);Cs=U(:,1:end-1)*diag(s(1:end-1))*U(:,1:end-1)';
nC=covEigTrans(Cs,'log'); plot(sort(eig(nC),'descend'),linecol(),'DisplayName','log');
