function [cov]=taucov(X,dim,taus,varargin);
% compute set of time-shifted covariance matrices
%
%  [cov]=taucov(X,dim,taus,...)
%
%  For each tau compute the covariance between all the input channels and there time-lagged versions:
%  i.e. cov(:,:,tau) = \sum_d x(t) x(t-tau)'
%
% Input:
%  
% Inputs:
%  X   -- [n-d] data to compute the lagged covariances of
%  dim -- spec of the dimensions to compute covariances matrices     ([1 2])
%         dim(1)=dimension to compute covariance over
%         dim(2)=dimension to sum along + compute delayed covariance, i.e. time-dim
%         dim(3:end)=dimensions to average covariance matrices over
%  taus-- [nTau x 1] set of sample offsets to compute cross covariance for  (0)
%         OR
%         [1x1] and <0, then use all values from 0:-taus
% Options:
%  type-- type of covariance to compute, one-of                ('real')
%          'real' - pure real, 'complex' - complex, 'imag2cart' - complex converted to real
%  shape-- shape of output covariance to compute               ('3d')
%         one-of; '3da'/'3d' - [d x d x nTau] non-symetric, 
%                 '3ds' - [d x d x nTau] symetric 
%                        (prevents complex eignvalues, but gives *invalid* non p.d. cov for \tau!=0)
%                 '2d' - [d*nTau x d*nTau]
%  normalize -- one of 'none','mean','unbiased'                ('mean')
%               'none' - done normalize, 'mean' = divide by num-samples
%               'unbiased' - divide by num overlapping samples. N.B. gives non-positive-definite cov!!!
%  wght  -- [nTau x 1] weighting for the different taus ([])
% Output:
%  cov -- [as defined by shape].  For each tau we get a [dxd] matrix cov_tau = 
opts=struct('type','real','shape','3d','normalize','mean','wght',[],'verb',1,'doublep',0);
opts=parseOpts(opts,varargin);
if ( nargin<2 || isempty(dim) ) dim=[1 2]; end;
if ( nargin<3 || isempty(taus) ) taus=0; end;
if ( numel(taus)==1 && isnumeric(taus) && taus<0 ) taus=0:-taus; end;

wght=opts.wght;
if ( ~isempty(wght) && ~(isnumeric(wght) || numel(wght)>5) )
  if ( strcmp(wght,'2d') && strcmp(opts.shape(1:min(end,2)),'3d') )
	 wght=[numel(taus) 2*(numel(taus)-1:-1:1)]; %weight by total #entries in the 2d-cross-correlation-mx
  elseif ( strcmp(wght,'3d') && strcmp(opts.shape(1:min(end,2)),'2d') )
	 wght=[numel(taus) 2*(numel(taus)-1:-1:1)]; %weight by total #entries in the 2d-cross-correlation-mx
	 wght=1./wght; % invert it, so when convert to 2d each component counts equally
  else
	 wght=mkFilter(numel(taus),wght);
  end
end;

szX=size(X); nd=ndims(X); szX(end+1:max(dim)+1)=1;
if ( szX(dim(1))>500 ) warning('Very large spacial dimension (%d).  Are you sure?',szX(dim(1))); end;
% Map to co-variances, i.e. outer-product over the channel dimension
didx1=1:max([ndims(X);dim(:)+1]); 
% insert extra dim for OP and squeeze out accum dims
shifts=zeros(size(didx1)); shifts(dim(2:end))=-1; % squeeze out accum
shifts(dim(1)+1)=shifts(dim(1)+1)+2; % insert for OP, and for taus
didx1=didx1 + cumsum(shifts);
didx1(dim(2:end))=-dim(2:end);  % mark accum'd
didx2=didx1; didx2(dim(1))=didx2(dim(1))+1;

% up X to double if wanted
if ( ~isa(X,'double') && opts.doublep ) X=double(X); end;

% compute the size of the output, to pre-allocate
szZ=zeros(1,max([didx1,didx2])); szZ(didx1(didx1>0))=szX(didx1>0); szZ(didx2(didx2>0))=szX(didx2>0); 
szZ(dim(1)+2)=numel(taus);
if ( ~isreal(X) )
  if ( strcmp(opts.type,'complex') )
    cov=complex(zeros(szZ,class(X)));
  elseif ( strcmp(opts.type,'imag2cart') )
    szZ(dim(1)+1)=szZ(dim(1)+1)*2;
    cov=zeros(szZ,class(X));
  end
else
  cov=zeros(szZ,class(X));
end
idxZ={}; for d=1:ndims(cov); idxZ{d}=1:szZ(d); end;


idx={}; for d=1:ndims(X); idx{d}=1:szX(d); end; idx2=idx; % index into X with offset tau
if ( opts.verb > 0 && numel(taus)>2 ) fprintf('taucov:'); end;
for ti=1:numel(taus);
  tau=taus(ti); % current offset
  idx{dim(2)} =1:szX(dim(2))-tau; % shift pts in cov-comp
  idx2{dim(2)}=tau+1:szX(dim(2));  
  if ( isreal(X) ) % include the complex part if necessary
    %covtau = tprod(real(X(idx{:})),didx1,X(idx2{:}),didx2);
    % avoid a double sub-set, uses less ram + faster...
    X2=X; if ( tau~=0 ) X2=cat(dim(2),X(idx2{:}),zeros([szX(1:dim(2)-1) tau szX(dim(2)+1:end)])); end;
    covtau = tprod(X,didx1,X2,didx2,'n');
  else
    switch (opts.type);
     case 'real';    % pure real output
      X2=X; if ( tau~=0 ) X2=cat(dim(2),X(idx2{:}),zeros([szX(1:dim(2)-1) tau szX(dim(2)+1:end)])); end;      
      covtau = tprod(real(X),didx1,real(X2),didx2,'n') + tprod(imag(X),didx1,imag(X2),didx2,'n');
     case {'complex','imag2cart'} % pure complex, N.B. need complex conjugate!
      %covtau = tprod(real(X),didx1,cat(dim(2),X(idx2{:}),zeros([szX(1:dim(2)-1) tau szX(dim(2)+1:end)])),didx2);
      X2=X; if ( tau~=0 ) X2=cat(dim(2),X(idx2{:}),complex(zeros([szX(1:dim(2)-1) tau szX(dim(2)+1:end)]))); end;
      covtau = tprod(X,didx1,conj(X2),didx2,'n');
      if ( strcmp(opts.type,'imag2cart') ) % map into equivalent pure-real covMx
        rcovtau=real(covtau); 
        icovtau=imag(covtau); if(isempty(icovtau)) icovtau=zeros(size(covtau),class(covtau)); end;
        covtau = cat(dim(1)+1,cat(dim(1),rcovtau,icovtau),cat(dim(1),-icovtau,rcovtau));% unfold to double size
        clear rcovtau icovtau;
      end
     otherwise; error('Unrecognised type of covariance to compute');
    end
  end
  clear X2;
  if ( numel(dim)>1 ) 
    wghttaui=1; if ( ~isempty(wght) ) wghttaui=wght(min(end,ti)); end;
    switch ( opts.normalize )
     case 'mean';      covtau=covtau.*(wghttaui./prod(szX(dim(2:end)))); 
     case 'unbiased';  covtau=covtau.*(wghttaui.*szX(dim(2))./(szX(dim(2))-abs(tau)));
     case {'none','noner','nonew'};
       if ( wghttauti~=1 ) covtau=covtau.*wghttaui; end;
      otherwise; error('Unrec normalize type: %s',opts.normalize);
    end
  end
  % store the result in the full matrix
  idxZ{dim(1)+2}=ti;
  cov(idxZ{:})=covtau;
  if ( opts.verb > 0 && numel(taus)>2 ) textprogressbar(ti,numel(taus)); end;
end
if ( opts.verb > 0 ) fprintf('\n'); end;
switch (opts.shape);
 case {'3da','3d'};  % do nothing
 case '3ds'; % symetrize the delayed versions
  idx={}; for d=1:ndims(cov); idx{d}=1:size(cov,d); end;
  for ti=1:numel(taus);
    idx{dim(1)+2}=ti;
    covtau = cov(idx{:})/2;
    cov(idx{:}) = covtau + permute(covtau,[1:dim(1)-1 dim(1)+1 dim(1) dim(1)+2:ndims(covtau)]);
  end
  clear covtau;
 case '2d'; % un-fold into 2d matrix equivalent
  cov=taucov3dto2d(cov);
end
return;
%-----------------------------------------------------------------------
function testCase()
X=randn(10,100,100);
C=taucov(X,[1 2],[0 1 2],'shape','3d');
C=taucov(X,[1 2],[0 1 2],'shape','3ds');
C=taucov(X,[1 2],[0 1 2],'shape','2d');
C2=taucov(X,[1 2],-20,'shape','2d');min(eig(C2(:,:,1)))

% with an unbiased estimate
Cu=taucov(X,[1 2],-20,'shape','2d','normalize','unbiased'); min(eig(Cu(:,:,1)))

%with weighting for each time lag
C=taucov(X,[1 2 3],[0 1 2],'wght','hanning2')


% weighting to invert the effect of the 2d version
C=taucov(X,[1 2 3],[0 1 2]);
C2d=taucov(X,[1 2 3],[0 1 2],'shape','2d');
C2d2=taucov3dto2d(C);
mad(C2d2,C2d)
wght=[3 2*(2:-1:1)];
C2dw=taucov(X,[1 2 3],[0 1 2],'shape','2d','wght',1./wght);

i=3;j=3;
CC=zeros(size(C));
for i=1:(size(C2dw,1)./size(X,1)); % rows
  for j=1:(size(C2dw,1)./size(X,1)); % cols
	 Cblk = C2dw((i-1)*size(X,1)+(1:size(X,1)),(j-1)*size(X,1)+(1:size(X,1)));
	 if ( i>j ) Cblk = Cblk'; end;
	 CC(:,:,abs(i-j)+1)=CC(:,:,abs(i-j)+1)+Cblk;
  end
end
for i=1:size(C,3); fprintf('%d = mad=%g\n',i,mad(CC(:,:,i),C(:,:,i)));end;
clf;mimage(Cblk.*wght(abs(i-j)+1),C(:,:,abs(i-j)+1))


clf;jplot([z.di(1).extra.pos2d],shiftdim(mdiag(sum(t.X,4),[1 2])))

% look at Wiener–Khinchin theorem for connection between autocov and PSD
C=taucov(X,[1 2],-100);
clf;plot(abs(fft(X))./sqrt(size(X,2)),'k','linewidth',3);hold on;plot(abs(squeeze(fft(C,size(X,2)))),'linewidth',2,'color','g')
C=taucov(X,[1 2],-100,'wght','hanning2');
plot(abs(squeeze(fft(C,size(X,2)))),'linewidth',2,'color','r')
C=taucov(X,[1 2],-100,'wght',[0 0 10 40]);
plot(abs(squeeze(fft(C,size(X,2)))),'linewidth',2,'color','m')
