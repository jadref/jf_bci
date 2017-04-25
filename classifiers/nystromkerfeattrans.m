function [Xk,W,Xsamp,samp] = nystromkerfeattrans(X,kerType,rank,varargin)
% compute the nystrom kernel feature transformation
%
%    [Xk,W,samp] = nystromkerfeattrans(X,kerType,rank,varargin)
%
%  Xk = W*K(samp,1:N), where W=K(samp,samp)^-.5
% 
% Inputs:
%  X - [n-d] data
%  kerType - kernel type to use
%  rank    - [1x1] rank>=1 : number of examples to use for the nystrom kernel approximation
%           OR [1 x 1] 0 < rank < 1 : fraction of examples to use
%           OR [ C x 1 ] indices into X to use as examples
%  varargin - other parameters to pass to the kernel computation
% Outputs:
%  Xk    - [rank x N] kernel feature transform
%  W    - [rank x rank] weighting matrix for the raw kernel entries
%  samp - [rank x 1] indices of the examples used to compute the feature map
% Options:
%  dim -- the dimension along which trials lie in X              (ndims(X))
if( nargin<2 || isempty(kerType) ) kerType='linear'; end;

dim = -1; % default dim is last
if ( numel(varargin)>1 && isequal(varargin{1},'dim') ) dim = varargin{2}; varargin(1:2)=[]; end;
dim(dim<0)=dim(dim<0)+ndims(X)+1; 

% 1) pick the required random number of examples
szX=size(X);
N  = prod(szX(dim));
if ( numel(rank)==1 ) % rank is number of examples to randomly choose  
  samp=randperm(N);
  if( rank>=1 ) % rank is number elements
	 samp=samp(1:min(end,rank));
  elseif ( rank>0 & rank<1 ) % rank is fraction of examples to use
	 samp=samp(1:min(end,ceil(N*rank)));
  end
  samp=sort(samp,'ascend'); % process in increasing numerical order
else % rank is an explicit set of examples to choose
  samp=rank;
end

% 2) Compute the kernel between these examples and the full data
idxc={}; for d=1:numel(szX); idxc{d}=1:szX(d); end;
if ( numel(dim)==1 ) idxc{dim}=samp;
else  error('Multiple trial dims not supported yet');
end;
Xsamp= X(idxc{:});
Kcm  = compKernel(Xsamp,X,kerType,'dim',dim,varargin{:});
Kcc  = Kcm(:,samp,:,:);

% 3) Compute the inverse square root of the center kernel
szKcc=size(Kcc);
W=zeros(size(Kcc));
for fi=1:prod(szKcc(3:end));
   [U,s]=eig(Kcc(:,:,fi)); s=diag(s);
   si=s>0 & ~isinf(s) & ~isnan(s) & s>max(s)*1e-6; % ensure is positive definite 
   W(:,1:sum(si),fi)=U(:,si)*diag(1./sqrt(s(si)));
end;

% 4) Combine inv-square-root with the full set to get the kernel feature map
if ( size(W,3)==1 ) 
   Xk  = W'*Kcm;
else
   Xk  = tprod(W,[-1 1 3:ndims(W)],Kcm,[-1 2:ndims(Kcm)]);
end
return;
										  %-----------------------------------------
function testCase()
X=randn(10,20,100);    K=nystromkerfeattrans(X,[],40); % basic test
X=randn(10,20,10,100); K3d=nystromkerfeattrans(X,'covlogppDist',40); % with feature dims


						  % generate problem for which non-linear kernel is needed
  % simple nested ring problem
N=100;
Y=(randn(1,N)>0)*2-1;
Xpolar=rand(2,N)*2*pi; % (r,theta)
Xpolar(1,Y>0) = Xpolar(1,Y>0)+1*2*pi;
X     =[Xpolar(1,:).*sin(Xpolar(2,:));Xpolar(1,:).*cos(Xpolar(2,:))]; % convert to cart

clf;labScatPlot(X,Y,'linewidth',2)

% train with linear classifier
[wb,f]=lr_cg(X,Y,1,'verb',1); conf2loss(dv2conf(Y,f))

% train with non-linear classifier
K=compKernel(X,[],'rbf','dim',-1);
[wb,f]=klr_cg(K,Y',1,'verb',1); conf2loss(dv2conf(Y,f))

% train linear classifier on kernel feature transformed data
Xrbf = nystromkerfeattrans(X,'rbf',40,'dim',-1);
clf;mimage(K,Xrbf'*Xrbf,'diff',1); % check quality of the approximation

[wb,f]=lr_cg(Xrbf,Y,1,'verb',1); conf2loss(dv2conf(Y,f))  % how good is the classifier...
