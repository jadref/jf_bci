function [sf,d,Sigmai,Sigmac,SigmaAll]=csp(X,Y,dim,demonType,ridge,cent,singThresh,powThresh)
% Generate spatial filters using CSP
%
% [sf,d,Sigmai,Sigmac,SigmaAll]=csp(X,Y,[dim,demonType,ridge,singThresh,powThresh]);
%
%   sf = \argmin (sf'*\Sigma_i sf) / (sf' \Sigma_type sf')
% where
%   \Sigma_type = \Sigma_{all}, \Sigma_{rest}, \Sigma_{1v1}
%
% N.B. if inputs are singular then d will contain 0 eigenvalues & sf==0
%
% Inputs:
%  X     -- n-d data matrix, e.g. [nCh x nSamp x nTrials] data set, OR
%           [nCh x nCh x nTrials] set of *trial* covariance matrices, OR
%           [nCh x nCh x nClass ] set of *class* covariance matrices
%  Y     -- [nTrials x 1] set of trial labels, with nClass unique labels, OR
%           [nTrials x nClass] set of +/-1 (&0) trial lables per class, OR
%           [nClass  x 1] set of class labels when X=[nCh x nCh x nClass] (1:size(X,dim))
%           N.B. in all cases a label of 0 indicates ignored trial
%  dim   -- [1 x 3] dimension of X which contains the [trialDim chDim featureDim] where
%             trialDim is the dimension which contains the trials
%             chDim the one which contains the channels (optional) 
%             featureDim the one which contain features to compute separate filters for (optionally)
%           If channel dim not given the next available dim is used. ([-1 1])
%  denomType -- [str] type of demoninator to use in the filter computation. 
%           one of: 'all', 'rest', '1v1'
%  ridge -- [float] ridge (as fraction of mean eigenvalue) to add for numerical stability (1e-7)
%           ridge>0 = add to both numerator and demoninator
%           ridge<0 = add only to demoniator
%  singThresh -- [float] threshold to detect singular values in inputs (1e-3)
%  powThresh  -- [float] fractional power threshold to remove directions (1e-4)
% Outputs:
%  sf    -- [nCh x nCh x nClass] sets of 1-vs-rest spatial *filters*
%           sorted in order of increasing eigenvalue.
%           N.B. sf is normalised such that: mean_i sf'*cov(X_i)*sf = I
%           N.B. to obtain spatial *patterns* just use, sp = Sigma*sf ;
%  d     -- [nCh x nClass] spatial filter eigen values, N.B. d==0 indicates bad direction
%  Sigmai-- [nCh x nCh x nTrials] set of *trial* covariance matrices
%  Sigmac-- [nCh x nCh x nClass]  set of *class* covariance matrices
%  SigmaAll -- [nCh x nCh] all (non excluded) data covariance matrix
%
if ( nargin < 3 || isempty(dim) ) dim=[-1 1]; end;
if ( numel(dim) < 2 ) if ( dim(1)==1 ) dim(2)=2; else dim(2)=1; end; end
dim(dim<0)=ndims(X)+dim(dim<0)+1; % convert negative dims
if ( nargin < 4 || isempty(demonType) ) demonType='all'; end;
if ( nargin < 5 || isempty(ridge) ) 
   if ( isequal(class(X),'single') ) ridge=1e-7; else ridge=0; end;
end;
if ( nargin < 6 || isempty(cent) ) cent=0; end;
if ( nargin < 7 || isempty(singThresh) ) singThresh=1e-3; end
if ( nargin < 8 || isempty(powThresh) )  powThresh=1e-4; end

szX=size(X);
nCh = szX(dim(2)); N=szX(dim(1));
nFeat=1; if( numel(dim)>2 ) nFeat=prod(szX(dim(3:end))); end;
if ( numel(dim)>3 ) error('Multiple feature dims not supported yet...'); end;
nSamp=prod(szX(setdiff(1:numel(szX),dim)));

% compute the per-trial covariances
if ( ~isequal(dim(:),[3;1]) || ndims(X)>3 || nCh ~= size(X,2) )         
   idx1=-(1:ndims(X)); idx2=-(1:ndims(X)); % sum out everything but ch, trials
   idx1(dim(1))=4;     idx2(dim(1))=4;     % linear over trial dimension
   idx1(dim(2))=1;     idx2(dim(2))=2;     % Outer product over ch dimension
   if ( isreal(X) ) 
      Sigmai = tprod(X,idx1,[],idx2);
   else % cope with complex inputs.... by assuming come in conjugate pairs....
      Sigmai = tprod(real(X),idx1,[],idx2) + tprod(imag(X),idx1,[],idx2);% pure real output
   end
   if ( cent ) % center the co-variances, N.B. tprod to comp means for mem
      error('Unsupported -- numerically unsound, center before instead');
%       sizeX=size(X); muSz=sizeX; muSz(dim)=1; mu=ones(muSz,class(X));
%       idx2(dim)=0; mu=tprod(X,idx1,mu,idx2); % nCh x 1 x nTr
%       % subtract the means
%       Sigmai = Sigmai - tprod(mu,[1 0 3],[],[2 0 3])/prod(muSz); 
   end
%    Fallback code
%    if(dim(1)==3)     for i=1:size(X,3); Sigmai(:,:,i)=X(:,:,i)*X(:,:,i)'; end
%    elseif(dim(1)==1) for i=1:size(X,1); Sigmai(:,:,i)=shiftdim(X(i,:,:))*shiftdim(X(i,:,:))'; end
%    end
else
   Sigmai = X;
end
% N.B. Sigmai *must* be [nCh x nCh x nFeat x N]
if ( ndims(Sigmai)==3 )
  Sigmai=reshape(Sigmai,[size(Sigmai,1),size(Sigmai,2),1,size(Sigmai,3)]);
elseif ( ndims(Sigmai)>3 )
  tmp=size(Sigmai); Sigmai=reshape(Sigmai,[tmp(1:2) prod(tmp(3:end-1)) tmp(end)]);
end

if ( ndims(Y)==2 && min(size(Y))==1  ) 
  oY=Y;
  if ( ~(all(Y(:)==-1 | Y(:)==0 | Y(:)==1)) )
	 Y=lab2ind(Y,[],[],[],0); 
  else
	 Y=cat(2,Y(:),-Y(:)); % duplicate labels for 2nd class
  end
end;
nClass=size(Y,2);
%if ( nClass==2 && all(Y(:,1)==-Y(:,2)) ) nClass=1; end; % only 1 for binary problems

allY0 = all(Y==0,2); % trials which have label 0 in all sub-prob
SigmaAll = sum(double(Sigmai(:,:,:,~allY0)),4); % sum all non-0 labeled trials
Sigmac   = zeros([nCh,nCh,nFeat,nClass],class(X));
sf    = zeros([nCh,nCh,nClass,nFeat],class(X)); d=zeros([nCh,nClass,nFeat],class(X));
for c=1:nClass; % generate sf's for each sub-problem
   % N.B. use double to avoid rounding issues with the inv(Sigma) bit
   Sigmac(:,:,:,c) = sum(double(Sigmai(:,:,:,Y(:,c)>0)),4); % +class covariance, [N,N,nFeat]
	switch lower( demonType ) 
	  case 'all';
		 if ( all((Y(:,c)==0)==allY0) ) % rest covariance, i.e. excludes 0 class
			Sigma=SigmaAll; % can use sum of everything
		 else
			Sigma=sum(double(Sigmai(:,:,:,Y(:,c)~=0)),3); % all for this class
		 end
	  case 'rest';  % anything *not* a positive class and is negative or positive in other sub-prob
		 Sigma=sum(double(Sigmai(:,:,:,~(Y(:,c)>0) & (Y(:,c)<0 | any(Y>0,2)))),3); 
	  case '1v1';   % use the -1 class for this problem
		 Sigma=sum(double(Sigmai(:,:,:,Y(:,c)<0)),3); 
	  otherwise; error('Unrecognised demon type');
	end
   % N.B. use double to avoid rounding issues with the inv(Sigma) bit
   Sigma=double(Sigma);

	for fi=1:nFeat;
   % solve the generalised eigenvalue problem, 
   if ( ridge>0 ) % Add ridge to numerator and denominator
      Sigmac(:,:,fi,c)=Sigmac(:,:,fi,c)+eye(size(Sigma,1))*abs(ridge)*mean(diag(Sigmac(:,:,fi,c))); 
      Sigma(:,:,fi)   =Sigma(:,:,fi)   +eye(size(Sigma,1))*abs(ridge)*mean(diag(Sigma(:,:,fi)));
	elseif ( ridge<0 ) % Add ridge to the denominator only
     Sigma(:,:,fi)   =Sigma(:,:,fi)    +eye(size(Sigma,1))*abs(ridge)*mean(diag(Sigma(:,:,fi)));
   end
   if ( 0 ) % gen eig solver method
     [W D]=eig(Sigmac(:,:,fi,c),Sigma(:,:,fi));D=diag(D); % generalised eigen-value formulation!
     W=real(W); % only real part is useful
   else % whiten then eigen method
     [Us,Ds] =eig(Sigma(:,:,fi)); Ds=diag(Ds); 
     si = Ds>0 & imag(Ds)<eps & Ds>max(abs(Ds))*1e-7; % numerical check
     Ws=repop(Us(:,si),'*',1./sqrt(abs(Ds(si)))');%compute the whitener
     WSigmacW=Ws'*Sigmac(:,:,fi,c)*Ws; % apply the whitener to the postive-class covariance
     [W,D]=eig(WSigmacW); D=diag(D);% decompose whitened covariance, i.e. rotate onto CSP directions
     W=Ws*W; % compose the 2 transformations, whitening and rotation
     %D=D/2; % scale to be 0-1
   end
   [dc,di]=sort(D,'descend');   W=W(:,di); % order in decreasing eigenvalue
   
   % Check for and correct for singular inputs
   % singular if eigval out of range
   % eig-value for this direction in full cov - exclude low power dirs!  
	%  ... similar to regularised whitener!
   nf = sum(W.*(Sigma(:,:,fi)*W),1)'; 
   si= dc<0+singThresh | abs(imag(dc))>1e-8 | isnan(dc) | nf<powThresh*sum(abs(nf));
   if ( sum(si)>0 ) % remove the singular eigen values & effect on other sf's

      % Identify singular directions which leak redundant information into
      % the other eigenvectors, i.e. are mapped to 0 by both Sigmac and Sigma
      % N.B. if the numerics are OK this is probably uncessary!
      Na  = sum((double(Sigmac(:,:,fi,c))*W(:,si)).^2)./sum(W(:,si).^2); 
      ssi=find(si);ssi=ssi(abs(Na)<singThresh & imag(Na)==0);%ssi=dc>1-singThresh|dc<0+singThresh;
      if ( ~isempty(ssi) ) % remove anything in this dir in other eigenvectors
         % Compute the projection of the rest onto the singular direction(s)
         Pssi    = repop(W(:,ssi)'*W(:,~si),'./',sum(W(:,ssi).*W(:,ssi),1)');
			if ( any(abs(Pssi)>singThresh) ) %remove this singular contribution
           W(:,~si)= W(:,~si) - W(:,ssi)*(Pssi.*(abs(Pssi)>singThresh)); 
			end
      end

      W=W(:,~si); dc=dc(~si); nf=nf(~si);% discard singular components   
   end
   
   %Normalise, so that diag(W'*Sigma*W)=N, i.e.mean_i W'*(X_i*X_i')*W/nSamp = 1
   % i.e. so that the resulting features have unit variance (and are
   % approx white?)
   W = repop(W,'*',nf'.^-.5)*sqrt(sum(Y(:,c)~=0)*nSamp); 

   % Save the normalised filters & eigenvalues
   sf(:,1:size(W,2),c,fi)= W;  
   d(1:size(W,2),c,fi)   = dc;
	end % features
end
% Compute last class covariance if wanted
if ( nClass==1 && nargout>3 ) Sigmac(:,:,2)=sum(double(Sigmai(:,:,Y(:,1)<0)),3);end;
return;

%-----------------------------------------------------------------------------
function []=testCase()
nCh = 64; nSamp = 100; N=300;
X=randn(nCh,nSamp,N);
Y=sign(randn(N,1));
[sf,d,Sigmai,Sigmac]=csp(X,Y);
% test filter-bank varient
nFeat=3;
X=randn(nCh,nSamp,nFeat,N);
Y=sign(randn(N,1));
[sf,d,Sigmai,Sigmac]=csp(X,Y,[-1 1 3]);

% check the effect of the differenent denominators....
% N.B. for binary problems all and rest should actually be identical, up-to scaling constants
sfa=csp(X,Y,[],'all');
sfr=csp(X,Y,[],'rest');
sf1=csp(X,Y,[],'1v1');


[sf2,d2]=csp(Sigmac,[-1 1]); 
[sf3,d3]=csp(Sigmac);
mimage(sf,sf2,'diff',1,'clim','limits')
