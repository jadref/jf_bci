function [W,Sigma,U,D,wX,mu,ridge]=whiten(X,dim,ridge,centerp,stdp,symp,linMapMx,tol,unitCov,order,verb)
% whiten the input data
%
% [W,Sigma,U,D,wX,mu,ridge]=whiten(X,dim[,ridge,center,stdp,symp,linMapMx,tol,unitCov,order])
% 
% N.B. this whitener leaves average signal power unchanged, but just de-correlates input channels
%
% Inputs:
%  X    - n-d input data set
%         OR
%         [nCh x nCh x n-d] set of data covariance matrices input-- only if dim(1)=0.
%  dim  - dim(1)=dimension to whiten, N.B. if dim(1)==0 then assume covariance matrices input
%         dim(2:end) whiten per each entry in these dim
%  ridge  - [float] regularisation parameter:                         (1) 
%           \Sigma' = (1-ridge)*\Sigma + ridge * I * mean(diag(\Sigma))
%           1=full-reg -> no-whitening, just unit-power/trial, 0=no-reg -> normal-whitening, 
%          'opt' = Optimal-Shrinkage est
%          'oas' = Optimal orcal shrinkage
%           ridge<0 -> 0=no-whitening, -1=normal-whitening, reg with ridge'th eigen-spectrum entry
%          'none' = don't regularise! just map to the eigen-directions
%  centerp- [bool] flag if we should center the data before whitening (1)
%  stdp   - [bool] flag if we should standardize input for numerical stability before whitening (0)
%  symp   - [bool] generate the symetric whitening transform (1)
%  linMapMx - [size(X,dim(2:end)) x size(X,dim(2:end))] linear mapping over non-acc dim 
%              use to smooth over these dimensions ([])
%           e.g. to use the average of 2 epochs to compute the covariance use:
%              linMapMx=spdiags(repmat([1 1]/2,size(X,dim(2)),1),[1 0],size(X,dim(2)),size(X,dim(2)))
%  tol  - [float] relative tolerance w.r.t. largest eigenvalue used to        (1e-6)
%                 reject eigen-values as being effectively 0. 
%           <0 >-1 : reject this percentage of the smallest eigenvalues
%           <-1    : reject this number of smallest eigenvalues
%  unitCov - [bool] make the covariance have unit norm for numerical accuracy (0)
%  order - [float] order of inverse to use (-.5)
% Outputs:
%  W    - [size(X,dim(1)) x nF x size(X,dim(2:end))] 
%          whitening matrix which maps from dim(1) to its whitened version
%          with number of factors nF
%       N.B. whitening matrix: W = U*diag(D.^order); 
%            and inverse whitening matrix: W^-1 = U*diag(D.^-order);
%  D    - [nF x size(X,dim(2:end))] orginal eigenvalues for each coefficient
%  U    - [size(X,dim(1)) x nF x size(X,dim(2:end))] 
%          eigen-decomp of the inputs
%  Sigma- [size(X,dim(1)) size(X,dim(1)) x size(X,dim(2:end))] the
%         covariance matrices for dim(1) for each dim(2:end)
%  wX   - [size(X)] the whitened version of X
%  mu   - [size(X) with dim(2:end)==1] mean to center everything else
if ( nargin < 3 || isempty(ridge) ) ridge=0; end;
if ( nargin < 4 || isempty(centerp) ) centerp=0; end;
if ( nargin < 5 || isempty(stdp) ) stdp=0; end;
if ( nargin < 6 || isempty(symp) ) symp=1; end;
if ( nargin < 7 ) linMapMx=[]; end;
if ( nargin < 8 || isempty(tol) ) % set the tolerance
   if ( isa(X,'single') ) tol=1e-6; else tol=1e-9; end;
end
if ( nargin < 9 || isempty(unitCov) ) unitCov=0; end; % improve condition number before inversion
if ( nargin < 10 || isempty(order) ) order=-.5; end;
if ( nargin < 11 || isempty(verb) ) verb=0; end;

dim(dim<0)=dim(dim<0)+ndims(X)+1;
if( dim(1)==0 ) covIn=true; dim(1)=1; else covIn=false; end;

szX=size(X); szX(end+1:max(dim))=1; % pad with unit dims as necessary
if ( covIn ) 
  accDims=setdiff(1:ndims(X),[2 dim(:)']); % set the dims we should accumulate over
else
  accDims=setdiff(1:ndims(X),dim); % set the dims we should accumulate over
end
N    = prod(szX(accDims));

% Estimate the full-data covariance
if ( covIn ) % covariance matrices input
  sX   = [];
	if ( isnumeric(ridge) || strcmp(ridge,'opt') || strcmp(ridge,'none') )
		Sigma=X; for d=1:numel(accDims) Sigma=sum(Sigma,accDims(d)); end; Sigma=Sigma./N;
	elseif ( any(strcmp(ridge,{'median','riemann','ld','geodisic','harmonic','geometric'})) )
	  % N.B. Riemann mean is very sensitive to numerical issues
	  Sigma = mean_covariances(double(X),ridge); 
	  ridge = 1; % reset the reg
	else
	  error('Unrecognised mean type');
	end
else
  % covariance + eigenvalue method
  idx1 = -(1:ndims(X)); idx1(dim)=[1 1+(2:numel(dim))]; % skip for OP dim
  idx2 = -(1:ndims(X)); idx2(dim)=[2 1+(2:numel(dim))]; % skip for OP dim   
  if ( isreal(X) ) % work with complex inputs
    XX = tprod(X,idx1,[],idx2);%[szX(dim(1)) szX(dim(1)) x szX(dim(2:end))]
  else
    XX = tprod(real(X),idx1,[],idx2) + tprod(imag(X),idx1,[],idx2);
  end

  if ( centerp ) % centered
    sX   = msum(X,accDims);                              % size(X)\dim
    sXsX = tprod(double(real(sX)),idx1,[],idx2);
    if( ~isreal(sX) ) sXsX = sXsX + tprod(double(imag(sX)),idx1,[],idx2); end
    Sigma= (double(XX) - sXsX/N)/N; 
    
  else % uncentered
    sX=[];
    Sigma= double(XX)/N;
  end
  clear XX;

  if ( stdp ) % standardise the channels before whitening
    X2   = tprod(real(X),idx1,[],idx1);     % var each entry
    if( isreal(X) ) X2 = X2 + tprod(imag(X),idx1,[],idx1); end
    if ( centerp ) % include the centering correction
      sX2  = tprod(real(sX),idx1,[],idx1);    % var mean
      if ( ~isreal(X) ) sX2=sX2 + tprod(imag(sX),idx1,[],idx1); end
      varX  = (double(X2) - sX2/N)/N; % channel variance                
    else      
      varX  = X2./N;
    end
    istdX = 1./sqrt(max(varX,eps)); % inverse stdX
                                    % pre+post mult to correct
    szstdX=size(istdX);
    Sigma = repop(istdX,'*',repop(Sigma,'*',reshape(istdX,[szstdX(2) szstdX([1 3:end])]))); 
  end
end
Nsigma=N;
   
if ( ~isempty(linMapMx) ) % smooth the covariance estimates
  if ( numel(dim)~=2 ) error('Not supported yet!'); end;
  mapType='norm';%'acausal';
  if ( iscell(linMapMx) ) mapType=linMapMx{1}; linMapMx=linMapMx{2}; end;
  if ( size(linMapMx,1)==1 && size(linMapMx,2)>1 ) 
    filt = linMapMx;
    if ( numel(filt)>szX(dim(2)) ) filt=filt((numel(filt)-szX(dim(2))+1):end); end; % correct if filt bigger than data
	 filt = filt./sum(abs(filt)); % normalize the scale of the filter, sum to 1
    
    if ( 0 ) 
       linMapMx=spdiags(repmat(filt,szX(dim(2)),1),0:numel(filt)-1,szX(dim(2)),szX(dim(2))); % N.B. implicitly reverses direction of filt
       linMapMx=full(linMapMx);
       if ( strcmp(mapType,'norm') )% remove startup/shutdown effects, by normalizing map weighting
          linMapMx=repop(linMapMx,'.*',1./sum(linMapMx,1)); 
       elseif( strcmp(mapType,'acausal') ) % BODGE: allow to look forward in time for startup entries
          for t=1:numel(filt)-1; linMapMx((t+1):numel(filt),t)=filt(t+1:end); end;
       end
    else
       linMapMx = filt; % N.B. this is reversed in convn later....
    end
  end
  if ( numel(linMapMx)>1 )
     if ( min(size(linMapMx))>1 ) % matrix convolution
        Sigma=tprod(Sigma,[1 2 -(3:ndims(Sigma))],linMapMx,[-(3:ndims(Sigma)) 3:ndims(Sigma)]);
     else % filter to apply
        Sigma=convn(Sigma,shiftdim(linMapMx(:),-ndims(Sigma)+1),'full'); % N.B. implicitly time-reverses filt
        Sigma=Sigma(:,:,1:end-numel(linMapMx)+1); % strip the padding entries
        if ( strcmp(mapType,'norm') )% remove startup/shutdown effects, by normalizing map weighting
           for t=1:numel(filt)-1;
              Sigma(:,:,t) = Sigma(:,:,t)./sum(filt(1:t)); % BODGE: assumes Sigma is [d x d x N]
           end
        end
     end
        

     % updated estimate of the number of samples for each Sigma
     % get the weighting for a typical example
     if( isempty(filt) ) tmp=size(linMapMx); filt=linMapMx(1:prod(tmp(1:(ndims(Sigma)-2)))); end;
     % estimate the number of orginal sigmas which contribute to a single updated one
     % effectivly increased by number of sigma combined
     Nsigma = N*sum(abs(filt)>max(abs(filt))*.1);%BODGE:count if within factor of 10 of max weight
  end
end

% give the covariance matrix unit norm to improve numerical accuracy
if ( unitCov && size(Sigma,3)>1 )  
  unitCov=median(diag(sum(Sigma,3)./size(Sigma,3))); Sigma=Sigma./unitCov; 
end;

% ridge is relative to full-data eig-spectrum
if ( isnumeric(ridge) && numel(ridge)==2 ) 
  Call = sum(Sigma(:,:,:),3)./size(Sigma(:,:,:),3);
  sall = eig(Call);
  [ans,si]=sort(abs(sall),'descend'); sall=sall(si); % get decreasing order
  % remove the rank defficient/non-PD entries
  si = sall>eps & ~isnan(sall) & abs(imag(sall))<eps; sall=sall(si);
  if ( ridge(1)==0) % leave unchanged
     ridge(1)=0;
  elseif ( ridge(1)==-1 ) % inf
     ridge(1)=inf;
  elseif ( abs(ridge(1))<=1 )% fraction of the spectrum
    % ridge=0 -> no ridge, ridge=-1 -> full ridge=no-whiten
	 ridge(1) = sall(max(1,ceil((1-abs(ridge(1)))*numel(sall)))); 
  else % number of components back in the eigen-spectrum
	 ridge(1) = sall(max(1,numel(sall)-abs(ridge(1)))); 
  end
end


W=zeros(size(Sigma),class(X));
if(numel(dim)>1) Dsz=[szX(dim(1)) szX(dim(2:end))];else Dsz=[szX(dim(1)) 1];end
D=zeros(Dsz,class(X));
nF=0;
if ( verb>=0 && size(Sigma,3)*size(Sigma,4)>10 ) fprintf('whiten:'); end;
for dd=1:size(Sigma(:,:,:),3); % for each dir
  Xdd=X; sXdd=sX;
   [Udd,Ddd]=eig(Sigma(:,:,dd)); Ddd=diag(Ddd); 
	% dec abs order,  BODGE: Force to be pure real...
   [ans,si]=sort(abs(Ddd),'descend'); Ddd=real(Ddd(si)); Udd=real(Udd(:,si)); 
   % compute the regularised eigen-spectrum
	rDdd=Ddd;
   if ( ischar(ridge) )     
     switch (ridge);
      case 'opt'; % optimal shrinkage regularisation estimate
       if ( numel(dim)>1 )
         if ( dim(2)~=3 ) error('Opt shrink only supported for dim=[1 3]'); end;
         Xdd=X(:,:,dd); if ( ~isempty(sX) ) sXdd=sX(:,:,dd); else sXdd=[]; end
       end       
       if ( unitCov ) 
          alphaopt=optShrinkage(Xdd,dim(1),Sigma(:,:,dd)*unitCov,sum(sXdd,2)./Nsigma,centerp); 
       else
          alphaopt=optShrinkage(Xdd,dim(1),Sigma(:,:,dd),sum(sXdd,2)./Nsigma,centerp); 
       end
       alphaopt=max(0,min(1,alphaopt));
       %alphaopt=1-alphaopt; % invert type of alpha to be strength of whitening
       %error('not fixed yet!');
       %fprintf('%d) alpha=%g\n',dd,alphaopt);
       rDdd = (1-alphaopt)*Ddd + alphaopt*mean(Ddd);
		case {'oas','rblw'};
        p=numel(Ddd);
        Uest  = p*sum(Ddd.^2)/(sum(Ddd).^2)-1;
        if ( strcmp(ridge,'oas') )        
           alpha = 1   /(Nsigma+1-2/p);            beta=(p+1)           /((Nsigma+1-2/p)*(p-1));
        elseif( strcmp(ridge,'rblw') )     
           alpha =(Nsigma-2)/(Nsigma*(Nsigma+2));  beta=((p+1)*Nsigma-2)/(Nsigma*(Nsigma+2)*(p-1));
        end
        rho   = min( alpha+beta/Uest, 1 );
        rDdd  = (1-rho) * Ddd + rho*mean(Ddd);		  
      case 'none';
       rDdd = ones(size(Ddd));
      otherwise; error('Unrec alpha type');
     end
   elseif( isnumeric(ridge) ) 
      % add ridge in such a way that output power is unity
      if ( numel(ridge)>1 ) % global constant to add
         if ( isinf(ridge(1)) ) rDdd(:)=1; % turn of reg
         elseif( ridge(1)<0 ) % fraction spectrum to use
            t = Ddd(round((1-abs(ridge(1)))*numel(Ddd))); % strength we want
            rDdd = (Ddd + t);                  % ridge
            rDdd = rDdd*sum(Ddd)./(sum(rDdd)); % normalize so total power remains the same
         else % simple constant to add
            rDdd = Ddd + ridge(1);
         end
         if ( ridge(2)==1 ) % re-normalize so total power remains the same
            rDdd = rDdd *sum(Ddd)./sum(rDdd); % ridge+normalize so total power remains the same
         end
      elseif ( ridge>1) % global constant to add
         rDdd = (Ddd + ridge(1))*sum(Ddd)./(sum(Ddd)+numel(Ddd)*ridge(1)); % ridge+normalize so total power remains the same         
      elseif ( ridge>0 && ridge<=1 ) % regularise the covariance
         rDdd = (1-ridge)*Ddd + ridge*mean(Ddd); % relative factor to add, scaled to preserve total power
      elseif ( ridge>=-1 && ridge<0 ) % percentage of spectrum to use
         %s = exp(log(Ddd(1))*(1+ridge)+(-ridge)*log(Ddd(end)));%1-s(round(numel(s)*ridge))./sum(s); % strength is % to leave
         t = Ddd(round((1-abs(ridge))*numel(Ddd))); % strength we want
         rDdd = (Ddd + t);                  % ridge
         rDdd = rDdd*sum(Ddd)./(sum(rDdd)); % normalize so total power remains the same
      end
   end
   % only eig sufficiently big are selected
   si=true(numel(rDdd),1);
   if ( tol>0 )
     si=rDdd>max(abs(rDdd))*tol; % remove small and negative eigenvalues
   elseif ( tol>=-1 ) % percentage to reject
     si(floor(end*(1+tol))+1:end)=false;
   elseif( tol<-1 ) % number to reject
     si(end-abs(tol)+1:end)=false;
   end
   if ( ~any(si) ) continue; end;
   % Now we've got a regularised spectrum compute it's inverse square-root to get a whitener
   iDdd=ones(size(Ddd),class(Ddd)); 
   if( order==-.5 ) iDdd(si) = 1./sqrt(rDdd(si)); else iDdd(si)=power(rDdd(si),order); end;
   % Use the principle-directions mapping and the re-scaling operator to compute the desired whitener
   
   if ( symp ) % symetric whiten = dimensionality preserving
      W(:,:,dd) = repop(Udd(:,si),'*',iDdd(si)')*Udd(:,si)';
      nF=size(W,1);
   else % non-symetric = dimensionality reducing
      W(:,1:sum(si),dd) = repop(Udd(:,si),'*',iDdd(si)');         
      nF = max(nF,sum(si)); % record the max number of factors actually used
   end
   U(:,1:sum(si),dd) = Udd(:,si);
   D(1:sum(si),dd)   = rDdd(si);
	if( verb>=0 && size(Sigma,3)*size(Sigma,4)>10) textprogressbar(dd,size(Sigma,3)*size(Sigma,4)); end;
end
if ( verb>=0 && size(Sigma,3)*size(Sigma,4)>10 ) fprintf('\n'); end;
% Only keep the max nF
W=reshape(W(:,1:nF,:),[szX(dim(1)) nF szX(dim(2:end)) 1]);
if ( isa(X,'single') ) W=single(W); end;
D=reshape(D(1:nF,:),[nF szX(dim(2:end)) 1]);

% undo the effects of the standardisation
if ( stdp && dim(1)~=0 ) W=repop(W,'*',istdX); end

% undo numerical re-scaling
if ( unitCov ) W=W./sqrt(unitCov); D=D.*unitCov; Sigma=Sigma*unitCov; end

if ( nargout>4 ) % compute the whitened output if wanted
  if ( covIn ) % covariance input
    idx1 = 1:ndims(X); idx1(1)=-1; 
    wX = tprod(X, idx1,W,[-1 1 dim(2:end)]); % apply whitening, pre
    idx1 = 1:ndims(X); idx1(2)=-2;
    wX = tprod(wX,idx1,W,[-2 2 dim(2:end)]); % apply whitening, post
  else
    if (centerp)   wX = repop(X,'-',sX./N); else wX=X; end % center the data
    % N.B. would be nice to use the sparsity of W to speed this up
    idx1 = 1:ndims(X); idx1(dim(1))=-dim(1); 
    wX = tprod(wX,idx1,W,[-dim(1) dim(1) dim(2:end)]); % apply whitening
  end
end
if ( nargout>5 && centerp && ~covIn ) mu=sX/N; else mu=[]; end;
return;
%------------------------------------------------------
function testCase()
  X=randn(10,100,100);

  z=jf_mksfToy();
  X=z.X;
clf;image3ddi(z.X,z.di,1,'colorbar','nw','ticklabs','sw');packplots('sizes','equal');



[W,Sigma,U,D,wX]=whiten(X,1);
imagesc(wX(:,:)*wX(:,:)'./size(wX(:,:),2)); % plot output covariance

% via whiten
[W,Sigma,U,Dopt,wX]=whiten(X,1,'opt'); % opt-shrinkage
[W,Sigma,U,Doas,wX]=whiten(X,1,'oas');

% with whiten for each example
[W,Sigma,U,D,wX]=whiten(X,[1 3],1); 

% with opt and per-example whiten
[W,Sigma,U,Dopt,wX]=whiten(X,[1 3],'opt');
[W,Sigma,U,Doas,wX]=whiten(X,[1 3],'oas');

clf;image3d(cat(3,D,Dopt,Doas),3)

% with weighted whiten for each example
% wght averages with previous cov-mx
N=2; wght=spdiags(repmat(ones(1,N)/N,size(X,3),1),N-1:-1:0,size(X,3),size(X,3));
[W,Sigma,U,D,wX]=whiten(X,[1 3],1,[],[],[],wght);
[W,Sigma,U,D,wX]=whiten(X,[1 3],1,[],[],[],ones(1,N)/N);
[W,Sigma,U,D,wX]=whiten(X,[1 3],1,[],[],[],[.1 .2 .3 .4 .5 .5 .5]); % interesting kernel

clf;image3d(W,2)
N=4;wght=spdiags(repmat(ones(1,N)/N,size(X,3),1),N-1:-1:0,size(X,3),size(X,3));
[W,Sigma,U,D,wX]=whiten(X,[1 3],1,[],[],[],wght); 
clf;image3d(W,2)

% test with covariance matrices as input
C=tprod(X,[1 -2 3],[],[2 -2 3])./size(X,2);
[Wc,Dc]=whiten(C,0);

% check that the std-code works
A=randn(11,10);  A(1,:)=A(1,:)*1000;  % spatial filter with lin dependence
sX=randn(10,1000); sC=sX*sX'; [sU,sD]=eig(sC); sD=diag(sD); wsU=repop(sU,'./',sqrt(sD)'); 
X=A*sX; C=X*X';    [U,D]=eig(C);     D=diag(D);   wU=repop(U,'./',sqrt(D)'); 
mimage(wU'*C*wU,wsU'*sC*wsU)
mimage(repop(1./d,'*',wsU)'*C*repop(1./d,'*',wsU))
[W,Sigma,U,D,wX]=whiten(X,1,0,0);
[sW,sD,U,sSigma,swX]=whiten(X,1,0,1);

% Riemann center computation
zc=jf_cov(z);
% direct
Sigmad = mean_covariances(zc.X,'riemann');
Wd     = Sigmad^-.5;
wXd    = tprod(wht,[-1 1],tprod(zc.X,[1 -2 3],wht,[-2 2]),[-1 2 3]);
% using this wrapper
[W,Sigma,U,D,wX]=whiten(zc.X,0,'riemann'); % riemann centering, N.B. use dim=0 to mark as cov-in
mad(Cmu,Sigma)
mad(wht,W)
mad(wX,wXd)

										  % check complex inputs
X =randn(10,1000);
C =(X*X')./size(X,2);
fX=fft_posfreq(X,[],2)./sqrt(size(X,2)); % inc-normalization factor to keep the power equal
fX=fX.*sqrt(size(fX,2)./(floor(size(fX,2)/2)*2)); %N.B. extra correction for inclusion of the 0-hz bin
fC=(real(fX)*real(fX)'+imag(fX)*imag(fX)')./(size(fX,2));
clf;mimage(C,fC,'diff',1,'clim',[],'colorbar',1)
[W,Sigma,U,D]=whiten(X,1);
[fW,fD,fU,fSigma]=whiten(fX,1);
clf;mimage(Sigma,fSigma,'diff',1,'clim',[],'colorbar',1)
clf;mimage(W,fW,'diff',1,'clim',[],'colorbar',1)

										  % check with spectrogram input
[Xspect]=spectrogram(X,2,'windowType',1,'overlap',0,'feat','complex','detrend',0,'center',0);
[W,Sigma,U,D]=whiten(X,1);
[sW,sD,sU,sSigma]=whiten(Xspect,1);
% N.B. whiten averages by the number of bins => need to correct for the 1/2 spect and 0hz bin
mad(Sigma,sSigma.*((size(Xspect,2)*size(Xspect,3))./size(X,2)))
clf;mimage(Sigma,sSigma.*(size(Xspect,2)*size(Xspect,3))./size(X,2),'clim',[],'colorbar',1,'diff',1)
% same for the whitener
mad(W,sW./sqrt((size(Xspect,2)*size(Xspect,3))./size(X,2)))
clf;mimage(W,sW./sqrt((size(Xspect,2)*size(Xspect,3))./size(X,2)),'clim',[],'colorbar',1,'diff',1)

% now with a normal taper
[Xspect]=spectrogram(X,2,'windowType','hamming','feat','complex','detrend',0,'center',0);


