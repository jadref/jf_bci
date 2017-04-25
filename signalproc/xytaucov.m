function [cov]=xytaucov(X,Y,dim,taus,varargin);
% compute set of time-shifted cross covariance matrices
%
%  [cov]=xytaucov(X,Y,dim,taus,...)
%
%  For each tau compute the covariance between all the input channels and there time-lagged versions:
%  i.e. cov(:,:,tau) = \sum_t x(:,t) y(:,t-tau)'
%  N.B. we assume ***stationarity*** in the distributions, such that:
%        \sum_t x(t-i) x(t-j) = \sum_(t'=t-i) x(t')x(t'-(j-i)) = \sum_t x(t) x(t-tau)
%       if this is not the case then you should use xy3dtaucov
%       This is also *only* the case if max(tau)<<size(X,dim(2))
%
% Inputs:
%  X   -- [n-d] data to compute the lagged covariances of
%  Y   -- [n-d] data to compute the lagged covariances of. If empty Y=X
%           N.B. Y should have same size as X *except* for dim covariance is computed over, i.e. dim(1)
%                and that singlenton dimensions are automatically expanded
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
%  shape-- shape of output covariance to compute   ('3da')
%         one-of; '3da'/'3d' - [d x d x nTau] non-symetric, 
%                 '3ds'/'3d' - [d x d x nTau] symetric (fixs some numerical problems)
%                 '2d' - [d*nTau x d*nTau]
%  normalize -- one of 'none','mean','unbiased'                             ('noner')
%                postfix letter says how to deal with data from outside analysis window
%                'XXXXw'=pad with wrapped, 'XXXX0'=pad with 0, 'XXXXr'=pad with time-reversed
%  wght      -- [size(X,dim(2)) x 1] OR [size(Y)] 
%                 weighting over time-points **of X (by default)** when computing the cov-matrix. ([])
%                 such that cov(:,:,tau) = \sum_t wght(t) * x(:,t) y(:,t-tau)'
%                 N.B. 3d representation of the data does not correctly support weighting!
%  wghtX/wghtY-- [bool] if true then wght applys to X (or Y)                            (true,false)
%  bias-- [bool] add a virtual constant feature to compute the bias/offset contribution (0)
% Output:
%  cov -- [as defined by shape].  For each tau we get a [dxd] (or [d+1 x d] if bias is true)
%               matrix cov_tau = 
opts=struct('type','real','shape','3d','normalize','noner','verb',1,'bias',0,'wght',[],'wghtX',true,'wghtY',false);
opts=parseOpts(opts,varargin);
if ( isempty(opts.normalize) )       opts.normalize='mean'; end;
if ( strcmp(opts.normalize,'mean') ) opts.normalize='meanr'; end;
if ( nargin<2 || isempty(dim) ) dim=[1 2]; end;
autoCov=false; if ( nargin<3 || isempty(Y) )   Y=X; autoCov=true; end;
if ( nargin<4 || isempty(taus) ) taus=0; end;
if ( numel(taus)==1 && isnumeric(taus) && taus<0 ) taus=0:-taus; end;

szX=size(X); nd=ndims(X); szX(end+1:max(dim)+1)=1;
szY=size(Y); ndy=ndims(Y);szY(end+1:max(dim)+1)=1;
if ( szX(dim(1))>500 || szY(dim(1))>500 ) 
  warning('Very large spacial dimension (%d).  Are you sure?',szX(dim(1))); 
end;
if ( ~isnumeric(X) ) X=single(X); end;
if ( ~isnumeric(Y) ) Y=single(Y); end;
% Map to co-variances, i.e. outer-product over the channel dimension
didxX=1:max([ndims(X);dim(:)+1]); 
% insert extra dim for OP and squeeze out accum dims
shifts=zeros(size(didxX)); shifts(dim(2:end))=-1; % squeeze out accum
shifts(dim(1)+1)=shifts(dim(1)+1)+2; % insert for OP, and for taus
didxX=didxX + cumsum(shifts);
didxX(dim(2:end))=-dim(2:end);  % mark accum'd
didxY=didxX; didxY(dim(1))=(dim(1))+1;
% broadcast over Y
tmp=szX>1  & szY==1; tmp(dim(1))=false; if ( any(tmp) ) didxY(tmp)=0; end;
tmp=szY>1  & szX==1; tmp(dim(1))=false; if ( any(tmp) ) didxX(tmp)=0; end;
  

% compute the size of the output, to pre-allocate
szZ=zeros(1,max([didxX,didxY])); 
szZ(didxX(didxX>0))=szX(didxX>0); szZ(didxY(didxY>0))=szY(didxY>0); 
szZ(dim(1)+2)=numel(taus);
if ( opts.bias ) % storage for the bias feature
  szZ(dim(1))  =szZ(dim(1))+1; 
  %szZ(dim(1)+1)=szZ(dim(1)+1)+1;
end 
if ( (~isreal(X) || ~isreal(Y)) )
  if ( strcmp(opts.type,'complex') )
    cov=complex(zeros(szZ,class(X)));
  elseif ( strcmp(opts.type,'imag2cart') )
    szZ(dim(1)+1)=szZ(dim(1)+1)*2;
    cov=zeros(szZ,class(X));
  end
else
  cov=zeros(szZ,class(X));
end

% weighting for which time points to use in computation
wght=opts.wght; 
if ( isempty(wght) && any(isnan(Y(:))) ) wght=~isnan(Y); end
% fixed weighting over space
if( size(wght,1)==size(X,dim(2)) && (size(wght,2)==1 || size(wght,2)==size(X,dim(2)+1)) ) 
  wght=shiftdim(wght,-(dim(2)-1)); % shift to appropriate dimension of X
end
if( islogical(wght) || isinteger(wght) ) wght=single(wght); end;
if ( ~opts.wghtY && any(isnan(Y(:))) ) 
   warning('Y contains NaNs which are being suppressed');
   Y(isnan(Y(:)))=0;
end;

idx={}; for d=1:ndims(X); idx{d}=int32(1:szX(d)); end; % index into X with offset tau
idxY={}; for d=1:ndims(Y); idxY{d}=int32(1:szY(d)); end; % index into Y 
idxZ={}; for d=1:ndims(cov); idxZ{d}=int32(1:szZ(d)); end;
if ( opts.bias ) 
  idxZ{dim(1)}=int32(1:size(cov,dim(1))-1); %% idxZ{dim(1)+1}=1:size(cov,dim(1)+1)-1;
end;
if ( opts.verb > 0 ) fprintf('taucov:'); end;
for ti=1:numel(taus);
	 
  % compute the indices for the elements in Y, i.e. t-tau, to product with the elements of X
  tau=taus(ti); % current offset
  if ( tau>=0 ) % correlate backwards in time
    idxY{dim(2)}=tau+1:szX(dim(2));      % shift Y backwards
  else          % correlate forwards in time
    idxY{dim(2)}=1:szX(dim(2))-abs(tau); % shift Y forwards
  end

  % get copy of Y to modify in-place to do the correlation computation
  Y2=Y; 
  % include the weighting over time-points if needed
  % wght then shift => wght over Y, i.e. over *shifted* time-points
  if ( ~isempty(wght) && opts.wghtY ) Y2=repop(Y2,'*',wght); end
  % shift and zero pad Y to avoid a double sub-set, uses less ram + faster...
  switch lower(opts.normalize);
	 case {'unbiased','none','none0','mean0'}; 
		Ypad = zeros([szY(1:dim(2)-1) abs(tau) szY(dim(2)+1:end)]);
    case {'meanw','nonew'}; % pad with wrap-around input
		if ( tau>0 )   Ypad=Y2(:,(end-1):-1:(end-abs(tau)),:,:);
		elseif( tau<0) Ypad=Y2(:,abs(tau)+1:-1:1+1,:,:);
		end
	 case {'meanr','noner'}; % pad with time-reversed input
		if ( tau<0 )   Ypad=Y2(:,(end-1):-1:(end-abs(tau)),:,:);
		elseif( tau>0) Ypad=Y2(:,abs(tau)+1:-1:1+1,:,:);
		end
	 otherwise; error('Unrecognised normalize method');
  end
  if ( ~isreal(Y) && any(strcmp(opts.type,{'complex','imag2cart'})) ) % normal real version
    Ypad=complex(Ypad);
  end
  if ( tau>0 )      Y2=cat(dim(2),Y2(idxY{:}),Ypad); 
  elseif ( tau<0 )  Y2=cat(dim(2),Ypad,Y2(idxY{:}));
  end
  % include the weighting over time-points if needed
  % shift then wght => wght over X, i.e. over *non-shifted* points
  if ( ~isempty(wght) && opts.wghtX ) Y2=repop(Y2,'*',wght); end

  % Now do the actual covariance computation
  if ( isreal(X) ) % include the complex part if necessary
    covtau = tprod(X,didxX,Y2,didxY);
  else
    switch (opts.type);
     case 'real';    % pure real output
      covtau = tprod(real(X),didxX,real(Y2),didxY,'n') + tprod(imag(X),didxX,imag(Y2),didxY,'n');
     case {'complex','imag2cart'} % pure complex, N.B. need complex conjugate!
      %covtau = tprod(real(X),didxX,cat(dim(2),X(idxY{:}),zeros([szX(1:dim(2)-1) tau szX(dim(2)+1:end)])),didxY);
      covtau = tprod(X,didxX,conj(Y2),didxY,'n');
      if ( strcmp(opts.type,'imag2cart') ) % map into equivalent pure-real covMx
        rcovtau=real(covtau); 
        icovtau=imag(covtau); if(isempty(icovtau)) icovtau=zeros(size(covtau),class(covtau)); end;
        covtau = cat(dim(1)+1,cat(dim(1),rcovtau,icovtau),cat(dim(1),-icovtau,rcovtau));% unfold to double size
        clear rcovtau icovtau;
      end
     otherwise; error('Unrecognised type of covariance to compute');
    end
  end

  % compute the normalization factor
  NF=[];
  if ( numel(dim)>1 )
    switch ( opts.normalize )
     case {'mean','mean0','meanr'};		
		 if ( isempty(wght) )         NF=prod(szX(dim(2:end))); 
		 else                         NF=shiftdim(msum(wght,dim(2:end)),-1);
		 end
     case 'unbiased';               NF=(size(Y2,dim(2))-abs(tau));
     case {'none','none0','noner','nonew'}; NF=[];
     otherwise; error('Unrec normalize type: %s',opts.normalize);
    end
  end  
  if ( ~isempty(NF) ) covtau=repop(covtau,'/',NF); end; % apply the normalization

  % store the result in the full matrix
  idxZ{dim(1)+2}=int32(ti);
  cov(idxZ{:})=covtau;
  
  % include the bias term if wanted
  if ( opts.bias ) 
	 % sum over time for Y and insert as virtual constant channel for X
	 sumY=msum(Y2,dim(2:end));
	 if( ~isempty(NF) ) sumY=repop(sumY,'/',NF); end;
	 idxZ{dim(1)}  =size(cov,dim(1)); % virtual X channel
	 if (any(didxY==0)) % need to replicate along this dimension
		tmp=ones(1,numel(didxY)); tmp(didxY==0)=szX(didxY==0); sumY=repmat(sumY,tmp);
	 end
	 cov(idxZ{:})  =shiftdim(sumY,-1);
	 idxZ{dim(1)}  =1:size(cov,dim(1))-1;
  end
  clear Y2;
  if ( opts.verb > 0 ) textprogressbar(ti,numel(taus)); end;
end
if ( opts.verb > 0 ) fprintf('\n'); end;
switch (opts.shape);
 case {'3d','3da'};  % do nothing
 case '3d'; % symetrize the delayed versions
  if( autoCov ) 
    idx={}; for d=1:ndims(cov); idx{d}=1:size(cov,d); end;
	 if ( opts.bias ) idx{dim(1)}=1:size(cov,dim(1))-1; end; % don't flip the bias part
    for ti=1:numel(taus);
      idx{dim(1)+2}=ti;
      covtau = cov(idx{:})/2;
      cov(idx{:}) = covtau + permute(covtau,[1:dim(1)-1 dim(1)+1 dim(1) dim(1)+2:ndims(covtau)]);
    end
    clear covtau;
  end
 case '2d'; % un-fold into 2d matrix equivalent
  if ( dim(1) ~= 1 ) error('Not supported for dim(1)~=1 yet!, sorry'); end;
  if ( autoCov ) 
	 cov=taucov3dto2d(cov,opts.bias,taus);
  else
	 cov=taucovXY3dto2d(cov,opts.bias,taus);
  end
end
return;


%-----------------------------------------------------------------------
function testCase()
X=randn(10,100,100);
Y=randn(3,100,100);
C=xytaucov(X,Y,[1 2 3],[0 1 2]);

clf;jplot([z.di(1).extra.pos2d],shiftdim(mdiag(sum(t.X,4),[1 2])))

% with bias term
C=xytaucov(X,Y,[1 2 3],[0 1 2],'bias',1)

% with weighting over time-points
wght=zeros(1,size(X,2)); wght(:,10:end-10)=1; % only central points to be used
Xw=repop(X,'*',wght); Yw=repop(Y,'*',wght); % set certain time-points to zero
Cwx =xytaucov(X,Y,[1 2],[0 1 2 3],'wght',wght,'wghtX',true,'normalize','mean0');
% different weighting for each epoch
Cwex=xytaucov(X,Y,[1 2],[0 1 2 3],'wght',repmat(wght,[1 1 size(X,3)]),'wghtX',true,'normalize','mean0');
CXw =xytaucov(Xw,Y,[1 2],[0 1 2 3],'normalize','mean0');
mad(Cwx,CXw)
Cwy =xytaucov(X,Y,[1 2],[0 1 2 3],'wght',wght,'wghtX',false,'normalize','mean0');
CYw =xytaucov(X,Yw,[1 2],[0 1 2 3],'normalize','mean0');
mad(Cwy,CYw)


% test solving a wiener filter problem
Xin =randn(10,200,100);
X   =Xin;%cat(1,Xin,ones(1,size(X,2),size(X,3))); % add extra bias channel% 
taus=0:1:10;%N.B. tau>0 => Y lags X, tau<0 => Y leads X
irflen=10;
irf =mkSig(irflen,'gaus',irflen/2);%zeros(1,irflen); irf(5)=1; %
irf =irf./sum(abs(irf));
Y   =filter(irf,1,Xin(1,:,:))+1;
bias=1;
wght=[];%zeros(1,size(Y,2)); wght(:,1:10:end)=1;%
normalize='mean';
% try with bias term
XXtau=xytaucov(X,[],[],taus,'bias',bias,'wght',wght,'normalize',normalize);
XYtau=xytaucov(X,Y ,[],taus,'bias',bias,'wght',wght,'normalize',normalize);
XX   =taucov3dto2d  (sum(XXtau,4)./size(XXtau,4),bias); % [ (ch_x*tau)+bias * (ch_x*tau) ]
XY   =taucovXY3dto2d(sum(XYtau,4)./size(XYtau,4),bias); % [ (ch_x*tau)+bias * ch_y ]
W=(XX+eye(size(XX))*mean(diag(XX))*0)\XY; 
B=zeros(size(XYtau,2),1); if ( bias>0 ) B=W(end,:); end;
W=reshape(W(1:end-(bias>0),:),[size(XXtau,2),numel(taus),size(XYtau,2)]);
%clf;imagesc('cdata',W)

% apply W to the data
XW=stfilter(X,wb(1:numel(wb)-(bias>0)),taus);
mad(Y,XW)
clf;mimage(shiftdim(Y),shiftdim(XW),'diff',1)
clf;plot([shiftdim(X(1,:,1)) shiftdim(Y(1,:,1)) shiftdim(XW(1,:,1))]);legend('X','Y','XW');


% stim-sequence classification example
nSamp=100;
irflen=10;
nEpoch=100;
isi=5;
irf=mkSig(irflen,'gaus',irflen/2);
y2s=randn(nSamp/isi,2)>.5; while( abs(diff(sum(y2s>0)))>5 ); y2s=randn(size(y2s,1),2)>.5; end;
tmp=zeros(nSamp,2);tmp(1:isi:end,:)=y2s;y2s=tmp;
xtrue=filter(irf(end:-1:1),1,y2s); % convolve stimulus with irf
%clf;mcplot([y2s(:,1) xtrue(:,1) y2s(:,2) xtrue(:,2)])
Y =(randn(nEpoch,1)>0)+1;
X =shiftdim(xtrue(:,Y),-1);
X =X+randn(size(X))*1e-6;
%X =repop(X,'-',mean(X,2)); % ensure is 0-mean
taus=0:-1:-irflen*3;

%clf;subplot(121);plot(xtrue);subplot(122);plot(shiftdim(cat(3,mean(X(:,:,Y==1),3),mean(X(:,:,Y==2),3))))

bias=1;
wght=[];%zeros(1,size(Y,1)); wght(:,sum(y2s,2)>0)=1;%
% try with bias term
XXtau=xytaucov(X,[],[],taus,'bias',bias,'wght',wght);
XYtau=xytaucov(X,y2s',[],taus,'bias',bias,'wght',wght);
XX   =sum(XXtau,4)./size(XXtau,4);
XY   =(sum(XYtau(:,1,:,Y==1),4)+sum(XYtau(:,2,:,Y==2),4))./size(XYtau,4);
XX   =taucov3dto2d  (XX,bias,taus); % [ (ch_x*tau)+bias * (ch_x*tau) ]
XY   =taucovXY3dto2d(XY,bias,taus); % [ (ch_x*tau)+bias * ch_y ]
W=(XX+eye(size(XX))*mean(diag(XX))*0)\XY; 
B=zeros(size(XYtau,2),1); if ( bias>0 ) B=W(end,:); end;
W=reshape(W(1:end-(bias>0),:),[size(XXtau,2),numel(taus),size(Y,2)]);
clf;plot(W,'linewidth',1);


% apply to data and plot result
XW=zeros(size(Y,2),size(X,2),size(X,3));
for cyi=1:size(Y,2);
  for ei=1:size(X,3);
	 for ci=1:size(X,1);
		XW(cyi,:,ei) = XW(cyi,:,ei) + fliplr(filter(W(ci,:,cyi),1,fliplr(X(ci,:,ei))));
	 end
	 XW(cyi,:,ei) = XW(cyi,:,ei)+B(cyi);
  end
end
clf;mcplot([y2s(:,1) mean(X(:,:,Y==1),3)' mean(XW(:,:,Y==1),3)' y2s(:,2) mean(X(:,:,Y==2),3)' mean(XW(:,:,Y==2),3)']);legend('Y(1)','X(1)','XW(1)','Y(2)','X(2)','XW(2)');
