function [Cs,Ws,wX]=matcov(X,dim,stepDim,reg,tol,whtFeat,maxIter,verb)
% multi-dimensional matrix covariance computation (and multi-dim whitener)
%
% [C,W]=matcov(X,dim,[stepDim,reg,tol,whtFeat,maxIter,verb])
%
% N.B. warning, degenerates cov of the largest dim, d, if size(X,d)>size(X,~d)
%
% Inputs:
%  X    -- [n-d float] the data
%  dim  -- [int] dimension to outer-product over
%  stepDim -- [int] dimensions to step along, i.e. re-compute for each element of this dim(s) ([])
%  reg  -- [float] regularization strength                                    (0)
%  tol     -- [2xfloat]
%              tol(1)= convergence tolerance on fractional change in L2 norm  (1e-3)
%              tol(2)= min-eigenvalue tolerance, N.B. for normalized cov mx   (1e-8)
%  whtFeat -- 'str' feature to use for computing the covariance for each dimension,
%             one of: 'power','abs'
%  maxIter -- [int] max number of iters                                       (100)
%  verb -- [int] verbosity level                                              (1)
% Outputs:
%  C    -- {size(X,dim) x numel(dim)} covariances along each dim
%  W    -- {size(X,dim) x numel(dim)} whitener along each dim
if ( nargin<3 ) stepDim=[]; end;
if ( nargin<4 || isempty(reg) ) reg=0; end;
if ( nargin<5 || isempty(tol) ) tol=1e-3; end;
if ( nargin<6 || isempty(whtFeat) ) whtFeat={}; end;
if ( nargin<7 || isempty(maxIter) ) maxIter=100; end;
if ( nargin<8 || isempty(verb) ) verb=1; end;
szX=size(X);
eigTol=-1e-8; if ( numel(tol)>1 ) eigTol=tol(2); tol=tol(1); end; 

% initialize the estimates
Ws={}; Cs={};
for di=1:numel(dim);
  Ws{di}=eye(size(X,dim(di)));  Cs{di}=eye(size(X,dim(di)));
  if ( ~isempty(stepDim) )
	 Ws{di}=repmat(Ws{di},[1 1 szX(stepDim)]);
	 Cs{di}=repmat(Cs{di},[1 1 szX(stepDim)]);
  end
end

% TODO: optimise dim-order to speed convergence
dW=ones(numel(dim),1); dS=ones(numel(dim),1); 
for iter=1:maxIter;
  for di=1:ndims(dim);
	 oS = Cs{di}; oW=Ws{di};
	 % apply the other transformations
	 wX=X;
	 for dii=[1:di-1 di+1:numel(dim)];
										  % apply whitener
		wX  = tprod(wX,[1:dim(dii)-1 -dim(dii) dim(dii)+1:ndims(wX)],...
						Ws{dii},[-dim(dii) dim(dii) stepDim]); 
	 end
								 % compute the updated covariance for this dimension
	 idx1=-(1:ndims(wX)); idx1(dim(di))=1;
	 if ( ~isempty(stepDim) ) idx1(stepDim)=2+(1:numel(stepDim)); end; % include step dim
	 idx2=idx1; idx2(dim(di))=2;
										  % cov comp S = [ di x di x stepDims ]
	 if ( isreal(X) || numel(whtFeat)<di || isempty(whtFeat{di}) || strcmp(whtFeat{di},'power') )
		if ( isreal(X) )
		  S=tprod(wX,idx1,[],idx2); 
		else
		  S=tprod(real(wX),idx1,[],idx2)+tprod(imag(wX),idx1,[],idx2); 
		end		  
	 else
		if ( strcmp(whtFeat{di},'abs') )
		  S=tprod(abs(wX),idx1,[],idx2);
		elseif ( strcmp(whtFeat{di},'complex') )
		  S=tprod(wX,idx1,conj(wX),idx2);
		end
	 end	 
	 N=prod(szX)./prod(szX([dim(di) stepDim])); % number elements accumulated away
	 %S=S./N; % average covariance, N.B. do this later when normalizing the cov-anyway
										  % compute the whitener for these covariances
	 W=zeros(size(S));
	 for si=1:size(S(:,:,:),3);
		% normalize each dimensions covariance so that it has unit average diagonal
		nrm=sum(diag(S(:,:,si))./N)./size(S,1); % average power per element of dim
		S(:,:,si)=S(:,:,si)./N./nrm;
		% compute the whitener
		Ssi=S(:,:,si);
		[Ul,dl]=eig(Ssi); dl=diag(dl);
									% dec abs order,  BODGE: Force to be pure real...
		[ans,ssi]=sort(abs(dl),'descend'); dl=real(dl(ssi)); Ul=real(Ul(:,ssi)); 
										  % include reg
		if ( reg>0 ) dl=dl+reg./numel(dl); end; 
										  % compute the whitening re-scaling
		ssi=true(size(dl));  ssi(dl<abs(eigTol))=false; % only sufficiently large are re-scaled
		idl=ones(size(dl),class(S));  idl(ssi)=1./sqrt(dl(ssi));
				  % compute the symetric whitener, ignoring too small components
		W(:,:,si)=Ul(:,ssi)*diag(idl(ssi))*Ul(:,ssi)';
	 end
										  % compute the updated covergence info
	 dS(di)=(norm(oS(:)-S(:))./(norm(oS(:))/2+norm(S(:))/2));
	 dW(di)=(norm(oW(:)-W(:))./(norm(oW(:))/2+norm(W(:))/2));  
										  % convergence test on cov-estimates
	 if ( verb>0 ) fprintf('%2d.%2d) dS=%12g dW=%12g\n',iter,di,dS(di),dW(di)); end;
	 % store the updated info
	 Cs{di}=S; Ws{di}=W;
  end
  % TODO: compute the likelihood for the model and use for convergence testing

										  % convergence test
  if ( max(dS) < tol && max(dW) < tol )
	 break;
  end;
end

% push the scaling parameter into the cov/whiteners
% wX=X;
% for dii=1:numel(dim);
% 	 wX  = tprod(wX,[1:dim(dii)-1 -dim(dii) dim(dii)+1:ndims(wX)],Ws{dii},[-dim(dii) dim(dii) stepDim]); 
% end;
% ii=-(1:ndims(X));ii(stepDim)=1:numel(stepDim);
% nrm=tprod(wX,ii,[],ii)./prod(szX([dim;stepDim])); %ave-power/element after whitening  
sigma=nrm.^(1./numel(dim));% split left-over power equally over covariance estimates
ssigma=sqrt(sigma); % N.B. whitener is included 2*numel(dim) times in power computation
for di=1:numel(dim);
  Cs{di} = Cs{di}.*sigma;
  Ws{di} = Ws{di}./ssigma;
end

						 % compute the fully whitened data by
						 % applying the final whitener + include the re-normalization
if ( nargout>2 )
  wX=X;
  for dii=1:numel(dim);
	 wX  = tprod(wX,[1:dim(dii)-1 -dim(dii) dim(dii)+1:ndims(wX)],Ws{dii},[-dim(dii) dim(dii) stepDim]); 
  end;
end

return;

%----------------------------------------
function testCase()
X=randn(10,100,100);
X=cumsum(cumsum(X,1),2); % add some dependency info    
dim=[1 2];

S=mcov(X,dim);
clf;imagesc(reshape(permute(S,[1 3 2 4]),size(S,1)*size(S,3),size(S,1)*size(S,3)));

% normal test
[Cs,Ws,wX]=matcov(X,[1 2]);
clf;mimage(Cs{:},'clim',[])
S   = tprod(wX,[1 3 -3],[],[2 4 -3])./size(X,3);
mean(diag(reshape(permute(S,[1 3 2 4]),size(X,1)*size(X,2),[])))

										  % degenerate inputs
[Cs,Ws,wX]=matcov(X(:,:,1),[1 2]);
										  % try with stepDim
[Cs,Ws]=matcov(X,[1 2],3);

										  % try with complex inputs
[Cs,Ws]=matcov(complex(X,X),[1 2]);

										  % and with whtFeat
[Cs,Ws]=matcov(complex(X,X),[1 2],[],[],[],{'','abs'});


% check the whitener's work
Xi  = tprod(X,[1:dim(2)-1 -dim(2) dim(2)+1:ndims(X)],Ws{2},[-dim(2) dim(2)]); % apply trans  
Sl  = tprod(Xi,[-(1:dim(2)-1) 1 -(dim(2)+1:ndims(Xi))],[],[-(1:dim(2)-1) 2 -(dim(2)+1:ndims(Xi))]);
Xi  = tprod(X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],Ws{1},[-dim(1) dim(1)]); % apply trans  
Sr  = tprod(Xi,[-(1:dim(1)-1) 1 -(dim(1)+1:ndims(Xi))],[],[-(1:dim(1)-1) 2 -(dim(1)+1:ndims(Xi))]);
Xi  = tprod(X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],Ws{1},[-dim(1) dim(1)]); % apply trans
Xi  = tprod(Xi,[1:dim(2)-1 -dim(2) dim(2)+1:ndims(X)],Ws{2},[-dim(2) dim(2)]); % apply trans
S   = tprod(Xi,[1 3 -3],[],[2 4 -3])./size(X,3);
clf;imagesc(reshape(permute(S,[1 3 2 4]),size(X,1)*size(X,2),[]));colorbar;
mean(diag(reshape(permute(S,[1 3 2 4]),size(X,1)*size(X,2),[])))
clf;mimage(Sl,Sr,reshape(permute(S,[1 3 2 4]),size(X,1)*size(X,3),[]),'clim',[],'colorbar',1);

										  % test if can pre-sum over examples...
S  = tprod(X,[1 2 -3],[],[3 4 -3]);
Sr = aWr'*tprod(S,[-1 1 -3 2],aWl*aWl',[-1 -3])*aWr;
mad(Sr,Salxarr)
Sl = aWl'*tprod(S,[1 -2 2 -4],aWr*aWr',[-2 -4])*aWl;
mad(Sl,Salxarl)

