function [U,V,UXV,featDims]=svdProtoMap(X,Y,varargin)
% low-rank class centeriod based prototype classifier
%
% [U,V,UXV,featDims]=svdProtoMap(X,Y,varargin)
%
% Inputs:
%  X        -- [n-d] data matrix
%  Y        -- [1xN] +1/0/-1 class indicators, 0 labels are ignored
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
%  rank     -- [1x1] rank of solution to use                 ([])
%                     i.e. svd(W) only has largest 'rank' non-zero entries
%  labdim   -- [1x1] dimension of X which has an element for each input class ([])
% Outputs:
%  U,V      -- [dxr],[Txr] decompositions of the class prototype
%  UXV      -- [rxrxN] X mapped into this sub-space
%  featDims -- [d'x1] dimension indices for the dimensions treated as features to decompose 
opts=struct('dim',ndims(X),'labdim',[],'clsfr',1,'wght',[1 -1],'verb',1,'rank',[]);
opts=parseOpts(opts,varargin);

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;

if ( size(Y,1)==size(X,dim) && size(Y,2)~=size(X,dim) ) 
  persistent warned
  if (isempty(warned) ) 
	 warning('Y looks like [nEp x nSp] transposing to [nSp x nEp] as we want');
    warned=true;
  end
  Y=Y'; 
end;
exInd=all(isnan(reshape(Y,[],size(Y,ndims(Y)))),1) | all(reshape(Y,[],size(Y,ndims(Y)))==0,1);

% record which dims contain features
featDims=true(ndims(X),1); featDims(dim)=false;
if ( ~isempty(opts.labdim) ) featDims(opts.labdim)=false; end
featDims=find(featDims);
sizeX=size(X);

% compute the result
if ( opts.clsfr ) % classification problem

  % Compute a weighting matrix to compute the sum of each class's examples
  wght=single(Y>0); % [ nCls x N ]
  if ( size(Y,1)==1 ) wght=cat(1,wght,single(Y<0)); end; % double binary, +cls first

  if ( isempty(opts.labdim) ) % normal computation
	  
	 % convert from a sum to a class specific mean
	 wght=repop(wght,'./',size(wght,1)*sum(wght,2)); 
	 if ( size(wght,1) > 2 ) error('Prototype only defined for binary input problems'); end;
	 % merge into single weighting matrix, to compute the difference of the class means
	 wght=tprod(wght,[-1 1],opts.wght(:),-1);

	 % apply this weighting to the X to compute the centroids
	 Xidx=1:ndims(X); Xidx(dim)=-dim;
	 wght(:,exInd)=0; % only use training data
	 W   = tprod(X,Xidx,wght,-dim)./sum(abs(wght)); % same space, diff time for +/- class

  else % only use the indicated parts of the labdim'th dimension
	 % Compute the average where only use the class-specific bits of labdim'th dimension
	 Xidx= 1:ndims(X); Xidx(dim)=-dim; Xidx(opts.labdim)=-opts.labdim;
	 % average correct response only = same space and time for correct output
	 wght(:,exInd)=0; % only use training data
	 W   = tprod(X,Xidx,wght,[-opts.labdim -dim])./sum(wght(:));  
  end

else % class-insensitive problem = ignore class info
  if ( isempty(opts.labdim) ) % normal computation, average all epochs
	 wght =ones(1,size(X,dim(1))); 
	 wght(:,exInd)=0; % only use training data
	 Xidx =1:ndims(X); Xidx(dim)=-dim;
	 W    =tprod(X,Xidx,wght,[dim -dim])./sum(wght(:)); 		

  else % average epochs and label-specific responses
	 wght =ones(size(X,opts.labdim),size(X,dim(1))); 
	 wght(:,exInd)=0; % only use training data
	 Xidx=1:ndims(X); Xidx(dim)=-dim; Xidx(opts.labdim)=-opts.labdim;
	 W   = tprod(X,Xidx,wght,[-opts.labdim -dim])./sum(wght(:));  	 

  end
end
sizeW=size(W);

% decompose the solution
% N.B. we assume here 1st dim is X (space) and all other-dims are Y (time)
[U,s,V]=svd(W(:,:));s=diag(s);
% now apply the low-rank constraint
if ( ~isempty(opts.rank) && opts.rank>0 )
  % identify the rank's to keep
  % valid rank
  si=find(abs(s)>0 & ~isnan(s) & ~isinf(s));
  % largest magnitude
  [ans,ssi]=sort(abs(s(si)),'descend');
  keep=si(ssi(1:min(end,opts.rank)));
  U=U(:,keep); s=s(keep); V=V(:,keep);
end

% reshape V back to input dimension size
V=reshape(V,[sizeX(featDims(2:end)) size(V,2)]);

if ( nargout>2 ) % compute the transformed data-set	
  % first map with U -> 1st featDim -> rank components
  Uidx= [-featDims(1) featDims(1)];
  Xidx= [1:featDims(1)-1 -featDims(1) featDims(1)+1:ndims(X)];
  UXV  = tprod(X,Xidx,U,Uidx); % [ch_y x epoch]
  % then with V -> all other featDims -> rank components
  Vidx= [-featDims(2:end) featDims(2)];
  Xidx= 1:ndims(X); Xidx(featDims(2:end))=-featDims(2:end);
  UXV  = tprod(UXV,Xidx,V,Vidx);
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

% non-decomp test
[U,V,UXV]=svdProtoMap(X,Y,[]);
% only weight the positive examples
[U,V,UXV]=svdProtoMap(X,Y,[],'wght',[1 0]);
% decomp test
[U,V,UXV]=svdProtoMap(X,Y,[],'rank',1);

