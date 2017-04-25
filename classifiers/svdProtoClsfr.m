function [wb,f,J]=svdProtoClsfr(X,Y,C,varargin)
% low-rank class centeriod based prototype classifier
%
% [wb,f,J,p,M]=prototypeClass(X,Y,C,varargin)
%
% Inputs:
%  X        -- [n-d] data matrix
%  Y        -- [1xN] +1/0/-1 class indicators, 0 labels are ignored
%  C        -- [1x1] regularisation weight = rank of solution to generate ([])
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
%  rank     -- [1x1] rank of solution to use                 ([])
%                     i.e. svd(W) only has largest 'rank' non-zero entries
% Outputs:
%  wb       -- [] parameter matrix
%  f        -- [Nx1] set of decision values
%  J        -- [1x1] obj fn value
opts=struct('dim',ndims(X),'ydim',[],'labdim',[],'clsfr',1,'wght',[1 -1],'verb',1,'alphab',[],'rank',[],'bias',[]);
opts=parseOpts(opts,varargin);
if ( isempty(opts.rank) && C>0 && abs(C-round(C))<eps )
  opts.rank=C;
end

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;

if ( size(Y,1)==size(X,dim) && size(Y,2)~=size(X,dim) ) 
  warning('Y looks like [nEp x nSp] transposing to [nSp x nEp] as we want');
  Y=Y'; 
end;
exInd=all(isnan(reshape(Y,[],size(Y,ndims(Y)))),1) | all(reshape(Y,[],size(Y,ndims(Y)))==0,1);

% record which dims contain features
featDims=true(ndims(X),1); featDims(dim)=false;
if ( ~isempty(opts.labdim) ) featDims(opts.labdim)=false; end
featDims=find(featDims);

% compute the result
if ( opts.clsfr ) % classification problem, use the class info to find the prototype(s)

  % Compute a weighting matrix to compute the sum of each class's examples
  if ( size(Y,1)==size(X,dim) && size(Y,2)~=size(X,dim) ) 
	 warning('Y looks like [nEp x nSp] transposing to [nSp x nEp] as we want');
	 Y=Y'; 
  end;
  exInd=all(isnan(reshape(Y,[],size(Y,ndims(Y)))),1) | all(reshape(Y,[],size(Y,ndims(Y)))==0,1);
  wght=single(Y>0); % [ nCls x N ]
  if ( size(Y,1)==1 ) wght=cat(1,wght,single(Y<0)); end; % double binary, +cls first

  if ( isempty(opts.labdim) ) % normal computation
	  
	 % convert from a sum to a class specific mean
	 wght=repop(wght,'./',size(wght,1)*sum(wght,2)); 
	 % merge into single weighting matrix, to compute the difference of the class means
	 wght=tprod(wght,[-1 1],opts.wght(:),-1);

	 % apply this weighting to the X to compute the centroids
	 Xidx=1:ndims(X); Xidx(dim)=-dim;
	 W   = tprod(X,Xidx,wght,-dim)./sum(abs(wght)); % same space, diff time for +/- class

  else % only use the indicated parts of the labdim'th dimension
	 % Compute the average where only use the class-specific bits of labdim'th dimension
	 Xidx= 1:ndims(X); Xidx(dim)=-dim; Xidx(opts.labdim)=-opts.labdim;
	 % average correct response only = same space and time for correct output
	 W   = tprod(X,Xidx,wght,[-opts.labdim -dim])./sum(wght(:));  
  end

else % class-insensitive problem = ignore class info
  wght =ones(1,size(X,dim(1))); wght(1,exInd)=0; % only use training data
  Xidx=1:ndims(X); Xidx(dim)=-dim;
  W    =tprod(X,Xidx,wght,[dim -dim]); 

end
% now apply the low-rank constraint
if ( ~isempty(opts.rank) && opts.rank>0 )
  % decompose the solution
  % N.B. we assume here 1st dim is X (space) and all other-dims are Y (time)
  [U,s,V]=svd(W(:,:));s=diag(s);
  % identify the rank's to keep
  % valid rank
  si=find(abs(s)>0 & ~isnan(s) & ~isinf(s));
  % largest magnitude
  [ans,ssi]=sort(abs(s(si)),'descend');
  keep=si(ssi(1:min(end,opts.rank)));
  U=U(:,keep); s=s(keep); V=V(:,keep);
  % re-construct the solution
  W=reshape(U*diag(s)*V',size(W));
end

% compute the predictions on the whole data set
Widx=-(1:ndims(W)); Widx(dim)=0;
Xidx=-(1:ndims(X)); Xidx(dim)=dim;
if ( ~isempty(opts.labdim) ) % same W for all y_ch, y_ch should be 1st output dim
  Xidx(opts.labdim)=1; Widx(opts.labdim)=0; 
end 
f   = tprod(X,Xidx,W,Widx); % [ch_y x epoch]

%% % compute the optimal bias
%% trnInd=Y~=0;
%% b = optbias(Y(trnInd),f(trnInd));
b=0;

% apply the bias & generate return values
f   = f + b;
J   = 0;
wb  = [W(:);b];
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
[wb,f,J]=svdProtoClsfr(X,Y,[]);
W=reshape(wb(1:end-1),[size(X,1) size(X,2)]);
clf;plot(svd(W))

% only weight the positive examples
[wb,f,J]=svdProtoClsfr(X,Y,[],'wght',[1 0]);

% decomp test
[wb,f,J]=svdProtoClsfr(X,Y,[],'rank',1);

% decomp with positive/negative extraction
ptrue=[sin((1:1000)*2*pi/10);sin((1:1000)*2*pi/15)]'; % 2 sin as true response
y2s  =[1 .1;.1 1]; % mapping between labels and signal strengths
Yl   =(randn(100,1)>0)+1; % labels
Y    =lab2ind(Yl);
X    =tprod(ptrue,[1 -2],y2s(:,Yl),[-2 2]);
[wb,f,J]=svdProtoClsfr(X,Y,[]);

% varients were we first pre-convolve with different target stimuli
% Convolve with the 'true' stimulus response
Xd   =tprod(X,[-1 2],ptrue,[-1 1]);
% convolve with the indicator for the stimulus events
dptrue=cat(1,zeros(1,size(ptrue,2)),single(diff(ptrue)>0&diff(ptrue([2:end 1],:))<0));
dptrue=repop(dptrue,'/',sum(dptrue));
Xdd  =tprod(X,[-1 2],dptrue,[-1 1]); % discrete ver
% train classifier aware of the dimensions to class mapping
[wb,f,J]=svdProtoClsfr(Xd,Y,[],'labdim',1); % classifier only from the correct stim-response pairs
[wb,f,J]=svdProtoClsfr(Xd,Y,[]); % classifier for the difference correct/incorrect mapping

clf;plot([Y (f'-mean(f(:)))./std(f(:))],'linewidth',1); % plot res + prediction

fIdx=ones(size(Y)); fIdx(1:10)=0; % add some excluded points
[wb,f,J]=svdProtoClsfr(Xd,Y.*fIdx,[],'labdim',1);

% for stimulus prediction
