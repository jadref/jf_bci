function [R,Xr,Xmu,opts]=ccaERP(X,Y,varargin)
% canonical correlation analysis based with averaged response
%
% [R,Xr,Xmu]=ccaERP(X,Y,varargin)
%
% Inputs:
%  X        -- [n-d] data matrix
%  Y        -- [Nx1] +1/0/-1 class indicators, 0 labels are ignored
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%  rank     -- [1x1] rank of solution to use                 (1)
%                     i.e. svd(W) only has largest 'rank' non-zero entries
%  ydim     -- [1x1] dimension of X which contains features extracted with each possible Y
% Outputs:
%  R        -- [size(X,dim) x nF] spatial filtering matrix
%  Xr       -- [n-d] spatially filtered X
%  Xmu      -- [size(X)] class average response used
opts=struct('dim',ndims(X),'ydim',[],'labdim',[],'verb',1,'rank',1);
opts=parseOpts(opts,varargin);
if( isempty(opts.ydim) && ~isempty(opts.labdim) ) opts.ydim=opts.labdim; end;

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;

szY=size(Y); if ( szY(end)==1 ) szY(end)=[]; end;
% Compute a weighting matrix to compute the sum of each class's examples
if ( size(Y,1)==size(X,dim) ) Y=Y'; end; % ensure is [nCls x N]
wght=single(Y>0); % [ nCls x N ]
if ( size(Y,1)==1 ) wght=cat(1,wght,single(Y<0)); end; % double binary, +cls first

% compute the result
if ( isempty(opts.ydim) ) % normal computation
  % convert from a sum to a class specific mean
  wght=repop(wght,'./',size(wght,1)*sum(wght,2)); 

  % apply this weighting to the X to compute the centroids
  Xidx=1:ndims(X); Xidx(dim)=-dim;
  Xmu = tprod(X,Xidx,wght,[-dim dim]);

  % then compute the spatial covariance
  Xidx1=-(1:ndims(X)); Xidx1(1)=1;
  Xidx2=-(1:ndims(X)); Xidx2(1)=2;
  XXmu=tprod(Xmu,Xidx1,[],Xidx2);
  
else % only use the indicated parts of the ydim'th dimension
  % Compute the average where only use the class-specific bits of ydim'th dimension
  Xidx=1:ndims(X); Xidx(dim)=-dim; Xidx(opts.ydim)=-opts.ydim;
  Xmu = tprod(X,Xidx,wght,[-opts.ydim -dim])./sum(wght(:));  % average correct response only

  % now compute the spatial covariance
  Xidx1=-(1:ndims(X)); Xidx1(1)=1;
  Xidx2=-(1:ndims(X)); Xidx2(1)=2;
  XXmu= tprod(Xmu,Xidx1,[],Xidx2);
end

% now apply the low-rank constraint
if ( ~isempty(opts.rank) && opts.rank>0 )
  % decompose the solution	
  [U,s,V]=svd(XXmu);s=diag(s); % N.B. XXmu is a covariance, so could use EIG (faster?)
  % identify the rank's to keep
  % valid rank
  si=find(abs(s)>0 & ~isnan(s) & ~isinf(s));
  % largest magnitude
  [ans,ssi]=sort(abs(s(si)),'descend');
  si=si(ssi(1:min(end,opts.rank)));
  U=U(:,si); s=s(si); V=V(:,si);
  % re-construct the solution
  W=reshape(U*diag(s)*V',size(XXmu));
end

if ( isempty(U) ) 
  R=eye(size(X,1)); 
else 
  R=U; 
end;
if ( nargout>1 ) % compute the filtered data
  Xr=tprod(X,[-1 2:ndims(X)],R,[-1 1]);
end
return;
%--------------------------------------------------------------------------
function testCase()
% make a 2-d feature space
% non-sym pos def matrices
wtrue=randn(40,50); [utrue,strue,vtrue]=svd(wtrue,'econ'); strue=diag(strue);
% sym-pd matrices
wtrue=randn(40,500); wtrue=wtrue*wtrue'; [utrue,strue]=eig(wtrue); strue=diag(strue); vtrue=utrue;
% re-scale components and make a dataset from it
strue=sort(randn(numel(strue),1).^2,'descend');
wtrue=utrue*diag(strue)*vtrue';% true weight
Ytrue=sign(randn(1000,1));
Y    =Ytrue + randn(size(Ytrue))*1e-2;
Xtrue=tprod(wtrue,[1 2],Y,[3]);
noise=randn([size(wtrue),size(Xtrue,3)])*sum(strue)/10; 
if(size(Xtrue,1)==size(Xtrue,2))noise=tprod(noise,[1 -2 3],[],[2 -2 3]); end;
X    =Xtrue + noise;

[wb,f,J]=ccaERP(Xd,Y.*fIdx,[],'ydim',1);

% decomp with positive/negative extraction
ptrue=[sin((1:1000)*2*pi/10);sin((1:1000)*2*pi/15)]'; % 2 sin as true response
y2s  =[1 .1;.1 1]; % mapping between labels and signal strengths
Yl   =(randn(100,1)>0)+1; % labels
Y    =lab2ind(Yl);
X    =tprod(ptrue,[1 -2],y2s(:,Yl),[-2 2]);
Xd   =tprod(X,[-1 2],ptrue,[-1 1]);

[wb,f,J]=ccaERP(Xd,Y,[],'ydim',1);

fIdx=ones(size(Y)); fIdx(1:10)=0; % add some excluded points
[wb,f,J]=ccaERP(Xd,Y.*fIdx,[],'ydim',1);
