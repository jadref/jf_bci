function [wb,f,J,cor]=wienerClsfr(X,Y,C,varargin)
% weiner/PLS filter based classifier
%
% [wb,f,J]=wienerClsfr(XXytau,Y,C,varargin)
%
% Inputs:
%  XXytau   -- [n-d] data matrix, this should be cov XX and XY for different time shifts
%                    concatenated along dim 2 [X*X' X*Y']
%                   N.B. we assume, X = [ch_x x [ch_x;ch_y] x tau x epoch]
%  Y        -- [Nx1] +1/0/-1 class indicators, 0 entries are ignored
%  C        -- [1x1] regularisation weight (ignored)
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%                   N.B. we assume, X = [ch_x x [ch_x;ch_y] x tau x epoch]
%                   where ch_x are channels of independent variable (data)
%                         ch_y are channels of dependent variable to be predicted
%                         tau are the temporal offsets
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
%  clsfr    -- [bool] act like a classifier, i.e. treat each ch_y as the stimulus (1)
%                     for a different class and return f-value which should be 
%                     max for the predicted class
%  Y2       -- [float] norm of the target values.  Used to produce correct goodness of fit (0)
%                     estimates
%  predType -- 'str' type of prediction to produce.                    ('cor')
%                one of:    'sim' -- similarity  = y_est'*y
%                           'cor' -- correlation = y_est'*y./sqrt(y_est'*yest*y'*y)
%  rank     -- [int] limit the rank of the solution.
%                rank>0 means using the partial-least-squares algorithm (PLS)
%                rank<0 means using a low rank approx to the spatio-temporal solution
%                  
% Outputs:
%  wb       -- [size(X,1) size(X,3)] spatio-temporal weighting matrix
%  f        -- [Nx1] set of decision values
%  J        -- [1x1] sum-squared-error for this trained system
opts=struct('dim',ndims(X),'ydim',[],'wght',[1 -1],'verb',1,'alphab',[],...
				'clsfr',1,'bias',0,'Y2',[],'predType','cor','rank',[]);
opts=parseOpts(opts,varargin);

% get the trial dim(s)
dim=opts.dim;
if ( isempty(dim) ) dim=ndims(X); end;
dim(dim<0)=dim(dim<0)+ndims(X)+1;
bias=opts.bias;if( isempty(bias) ) bias=0; end;

if ( isempty(Y) || numel(Y)==1 ) Y=ones(size(X,dim(1)),1); end;
if ( size(Y,1)==size(X,dim) ) Y=Y(:,:)'; end; % TODO: this should cope better with n-d Y

szY=size(Y); if ( szY(end)==1 ) szY(end)=[]; end;
% excluded points
exInd=any(isnan(reshape(Y,[],size(Y,ndims(Y)))),1) | all(reshape(Y,[],size(Y,ndims(Y)))==0,1);
if ( opts.clsfr ) 
  % Compute a weighting matrix to compute the sum of each class's examples
  wght=single(Y>0); % [ nCls x N ]
  if ( size(Y,1)==1 ) wght=cat(1,wght,single(Y<0)); end; % double binary
else
  wght =ones(1,size(X,dim(1))); wght(1,exInd)=0; % only use training data
end
N=sum(wght,2);

% identify the X/Y parts of the input.
xch=1:size(X,1)-(opts.bias>0);
bch=[]; if ( opts.bias ) bch=size(X,1); end;
ych=(size(X,1)-(opts.bias>0)+1):size(X,2);

% compute the average response for the training examples
Xidx= 1:ndims(X); Xidx(dim)=-dim;
XXmu= tprod(X,Xidx,wght,[dim -dim]); % [ ch_x(+b) x [ch_x;ch_y] x tau x nCls ]
% split the sum into the XX and XY parts
XYmu=XXmu(:,ych,:,:);    % XY is extra 2nd dim channels [ch_x(+b) x ch_y x tau x nCls]
XXmu=XXmu(:,xch,:,:);    % XX is the normal channels [ch_x(+b) x ch_x x tau x nCls]
% Now only use the Y parts which have the correct label
XX=sum(XXmu,4); % [ch_x x ch_x x tau]
if ( opts.clsfr ) % are we pretending to be a binary classifier
  % sum together the deconv with the *right* stimulus sequence, i.e. ave correct response
  XY=tprod(XYmu,[1 -2 3 -4],eye(size(XYmu,2),size(XYmu,4)),[-2 -4]); % [ch_x(+b) x 1 x tau]
else
  XY=sum(XYmu,4); % [ch_x x ch_y x tau]
end

Y2=opts.Y2;
% this is the real target info, use to compute the norm of Y for loss computation
if ( isempty(Y2) && size(Y,1)==numel(ych) ) 
   if ( ndims(Y)==2 ) Y2=Y; 
   else % Y still contains the time-dimension so sum it out
      Y2=Y; Y2(isnan(Y2))=0; Y2=tprod(Y2,[1 -(2:ndims(Y)-1) 2],[],[1 -(2:ndims(Y)-1) 2]);
   end
   Y2(Y2==0 | isnan(Y2))=1;
end

% Now compute the weighting matrix
% convert into the big 2-d matrices
XX=taucov3dto2d  (XX,opts.bias); % [ (ch_x*tau) * (ch_x*tau) ]
XY=taucovXY3dto2d(XY,opts.bias); % [ (ch_x*tau) * ch_y ]
% add the regularisor, inplace to save ram
if ( C>0 )
  diagIdx=1:size(XX,1)+1:numel(XX);  
  if ( opts.bias ) diagIdx(end)=[]; end  % don't regularise the bias
  XX(diagIdx)=XX(diagIdx)+sum(N)*C;
end

% find min-norm solution
if ( opts.verb>0 ) fprintf('wienerclsfr: Solving the system [%dx%d]...',size(XX,1),size(XY,2)); end;

if ( ~isempty(opts.rank) && opts.rank>0 )
  % find a PLS solution of given rank
  iXX  = inv(XX);
  [Q,R]= eig(XY'*iXX*XY);R=diag(R);
  [ans,si]=sort(abs(R)); 
  R    = R(si(1:opts.rank));  % [ch_y x 1]
  Q    = Q(:,si(1:opts.rank));% [ch_y x ch_y]
  P    = iXX*XY*Q; % [ (ch_x*tau) x ch_y ]
  % Reconstruct W
  WB   = (Q*P')'; % [(ch_x*tau) x ch_y]

else % full-rank min-norm solution
  if ( 0 )
	 WB = XX\XY; % [ (ch_x*tau) x ch_y ]
  else
	 WB = pinv(XX)*XY; % force min-norm, this isn't as good as octave's min-norm solution... why?
  end
end
if ( opts.verb>0 ) fprintf('done\n'); end;

% reshape back to the 3d representation
B =zeros(size(XYmu,2),1); if ( ~isempty(bch) ) B=WB(end,:); end;
if ( opts.clsfr ) % classifier has single weight for all ch_y
  W=reshape(WB(1:(numel(xch)*size(XXmu,3)),:),[numel(xch),size(XXmu,3)]);%[ch_x x tau x 1]
else
  W=reshape(WB(1:(numel(xch)*size(XXmu,3)),:),[numel(xch),size(XXmu,3),numel(ych)]);%[ch_x x tau x ch_y]
end
if ( ~isempty(opts.rank) && opts.rank<0 ) % low rank approx solution
   for yi=1:size(W,3); % for each output find the opt least squares rank -opts.rank solution
      [U,S,V]=svd(W(:,:,yi)); S=diag(S);
      [ans,si]=sort(abs(S),'descend'); si=si(1:abs(opts.rank));
      W(:,:,yi) = U(:,si)*diag(S(si))*V(:,si)';
   end
   WB(1:end-(bias>0),:)=reshape(W,[],size(W,3));
end

XWY =sum(WB.*XY);                % total inner-product with the target
XWXW=XX*WB; XWXW=sum(WB.*XWXW);  % total norm of all predictions
clear XX XY % free up some ram

% goodness of fit metric -- N.B. without Y*Y can't compute correct values
% total ave SSE = (XW-Y)^2/N = (XW*XW' - 2 XW*Y + Y*Y')/N
J = (XWXW-2*XWY)./sum(N);
if( ~isempty(Y2) ) J=J+sum(sum(Y2(:,~exInd),2)./(N+(N==0))); end;
% correlation = XW*Y./sqrt(XW*XW' * Y*Y')
% compute correlation between predictions and true
cor =XWY./sqrt(abs(XWXW));
if ( ~isempty(Y2) ) cor=repop(cor,'./',sqrt(sum(Y2(:,~exInd),2)')); end;
if ( opts.verb>0 ) 
	fprintf('w=[%s] J=[%s]\tcorr=[%s]\n',sprintf('%3.2f ',WB(1:min(end,3))),sprintf('%3.2f ',J),sprintf('%3.2f ',cor));
end

% compute predictions for all the trials = similarity for each trial for each predictor
XY=X(xch,ych,:,:); %[ch_x x ch_y x tau x epoch]
if ( opts.clsfr ) % are we pretending to be a classifier
  % decision value for each epoch is similarity of prediction to ch_y
  %[epoch x ch_y] = y_est'*y_ch = <(<X_tau,W_tau>+b),y_ch> = <<X_tau,y_ch>,W_tau>
  f = repop(tprod(XY,[-1 1 -3 2],W,[-1 -3]),'+',B); 
  % TODO: deal correctly with the bias term....
  % binary is special case, decision value is the diff of this similarity, so >0=+class, <0=-class
  %if ( size(f,1)==2 ) f = diff(f,[],1); end
else
  % decision value for each epoch is similarity of prediction to ch_y
  %[ch_y x epoch] = y_est'*y_ch = <<X_tau,W_tau>,y_ch> = <<X_tau,y_ch>,W_tau>
  f = repop(tprod(XY,[-1 1 -3 2],W,[-1 -3 1]),'+',B); 
end
if ( isequal(opts.predType,'cor') ) % trial prediction is correlation for that trial
  WXXW=taucovWXXW(X,WB,opts.bias);
  % normalize by this strength
  f  = f./sqrt(abs(WXXW));
  if ( ~isempty(Y2) ) f = f./sqrt(Y2); end
end
wb=WB;
return;
  
%--------------------------------------------------------------------------
function testCase()
% stim-sequence classification example
nSamp=100;
irflen=10;
isi   =3;
nEpoch=100;
offset=0;
bias  =0;
nCh   =10;
nCls  =3;
irf=mkSig(irflen,'gaus',irflen/2); irf=irf./sum(abs(irf));
y2s=randn(ceil(nSamp/isi),nCls)>.5; while( abs(diff(sum(y2s>0)))>5 ); y2s=randn(size(y2s,1),nCls)>.5; end;
tmp=zeros(nSamp,nCls);tmp(1:isi:end,:)=y2s;y2s=tmp;
xtrue=filter(irf(end:-1:1),1,y2s); % convolve stimulus with irf
%clf;mcplot([y2s(:,1) xtrue(:,1) y2s(:,2) xtrue(:,2)])
Yl =ceil(rand(nEpoch,1)*nCls);
[Y,key]=lab2ind(Yl);
if ( nCh>1 ) % add a spatial dimension
  A  =mkSig(nCh,'exp',1)-.2;
  X0 =tprod(A,[1 -1],shiftdim(xtrue(:,Yl),-1),[-1 2 3]);
else % no space
  X0 =shiftdim(xtrue(:,Yl),-1);
end
X  = X0+randn(size(X0))*5e-1+offset; % add noise
taus=0:-1:-irflen;

% pre-process by computing the time-shifted auto-correlations
XXy =cat(2,xytaucov(X,[],[],taus,'bias',bias),xytaucov(X,y2s',[],taus,'bias',bias));
%clf;plot(shiftdim(mean(XXy(:,2:end,:,:),4))')
Cscale=covVarEst(XXy,[3 4]);

% now try and train the classifier
[wb,f,J]=wienerClsfr(XXy,Y,Cscale*0,'bias',bias,'clsfr',1);
clf;plot([(Y-1.5)*2 (f(1,:)-mean(f(1,:)))'./std(f(1,:))],'linewidth',1)
W=reshape(wb(1:end-(bias>0)),size(XXy,1)-(bias>0),numel(taus));
clf;plot(W','linewidth',1);legend
pat=taucov3dto2d(mean(XXy(1:size(XXy,1)-(bias>0),1:size(XXy,1)-(bias>0),:,:),4),0)*wb(1:end-(bias>0));
clf;image3d(cat(3,W,reshape(pat,size(W))),1,'zvals',{'filt','pat'},'disptype','plot','plotopts',{'linewidth',1});
XW=stfilter(X,wb(1:end-(bias>0)),taus);
clf;mcplot([y2s(:,1) mean(X(1,:,Y==1),3)' mean(XW(:,:,Y==1),3)' y2s(:,2) mean(X(1,:,Y==2),3)' mean(XW(:,:,Y==2),3)'],'linewidth',1);legend('Y(1)','X(1)','XW(1)','Y(2)','X(2)','XW(2)');

% perception prediction example
nSamp =100;
irflen=10;
nEpoch=100;
offset=0;
bias  =0;
nCh   =2;
irf   =mkSig(irflen,'gaus',irflen/3,.5)-mkSig(irflen,'gaus',irflen*2/3,.5)+.5; %DoG
irf=irf./sum(abs(irf));
Y     =cumsum(randn([1,nSamp,nEpoch]),2); Y=repop(Y,'-',mean(Y,2));% continuous output, 0-mean
xtrue=filter(irf(end:-1:1),1,Y,[],2); % convolve stimulus with irf
%clf;mcplot([Y(:,:,1); xtrue(:,:,1); Y(:,:,2); xtrue(:,:,2)]')
if ( nCh>1 ) % add a spatial dimension
  A  =mkSig(nCh,'exp',1)-.2; A=A./norm(A);
  X0 =tprod(A,[1 -1],xtrue,[-1 2 3]);
else % no space
  X0 =xtrue;
end
X  = X0 + randn(size(X0))*1e-6 + offset;
taus=0:-1:-irflen;

% pre-process by computing the time-shifted auto-correlations
XXy =cat(2,xytaucov(X,[],[],taus,'bias',bias),xytaucov(X,Y,[],taus,'bias',bias));
clf;plot(shiftdim(mean(XXy(:,2:end,:,:),4))')
clf;imagesc('cdata',shiftdim(mean(XXy(1,1,:,:),4)))
% try to train the filter
Y2=Y(:)'*Y(:)./size(Y,2);
[wb,f,J,obj]=wienerClsfr(XXy,[],0,'clsfr',0,'bias',bias,'Y2',Y2);
clf;plot((f(1,:)-mean(f(1,:)))'./std(f(1,:)),'linewidth',1)

% plot the filtered data / re-cons stimuli
XW=stfilter(X,wb(1:end-(bias>0)),taus,wb(end).*(bias>0));
clf;plot([[Y(1,:,1)' X(1,:,1)' XW(1,:,1)']+0 [Y(1,:,2)' X(1,:,2)' XW(1,:,2)']+10],'linewidth',1);legend('Y(1)','X(1)','XW(1)','Y(2)','X(2)','XW(2)');
% compute the true inner products
ftrue=tprod(XW,[-1 -2 2],Y,[-1 -2 2])./size(Y,1)./size(Y,2);
clf;plot([f;ftrue]','linewidth',1)
% compute correlation scores
covtrue=ftrue./(sqrt(tprod(XW,[-1 -2 2],[],[-1 -2 2]).*tprod(Y,[-1 -2 2],[],[-1 -2 2]))./size(Y,1)./size(Y,2));
% compute the sse
ssetrue =tprod(Y-XW,[-1 -2 1],[],[-1 -2 1])./size(Y,1)./size(Y,2);
ssetrue2=tprod(XW,[-1 -2 2],[],[-1 -2 2])./size(Y,1)./size(Y,2)-2*ftrue+tprod(Y,[-1 -2 2],[],[-1 -2 2])./size(Y,1)./size(Y,2);

pat=taucov3dto2d(mean(XXy(1:size(XXy,1)-(bias>0),1:size(XXy,1)-(bias>0),:,:),4))*wb(1:end-(bias>0));
clf;plot([shiftdim(wb) pat],'linewidth',1); legend('filt','pat')

% with a z structure
bias=1;
taus=0:-1:-irflen_samp;
z=jf_mkoverlapToy();
oz=z;
z=jf_xytaucov(z,'taus_samp',taus,'bias',bias); %pre-compute the cov info
% sanity check the computation
e=jf_ERP(z);clf;image3d(squeeze(e.X(1:10,end-1,:,:)),1,'disptype','plot');
% run the classifier
Cs=[1 .1 .01 .001]; % test re-seeding for increasing Cs
Cscale=covVarEst(z.X,[3 4]);
[wb,f,J]=wienerClsfr(z.X,z.Y,0,'dim',4,'clsfr',1,'bias',bias);

cvtrainMap('wienerClsfr',zpp.X(:,:,1:64,:),zpp.Y,Cscale*Cs,zpp.foldIdxs,'ydim',1,'lossFn','est2loss','bias',bias,'clsfr',1)

df=-diff(f,[],1);clf;plot([z.Y df'./std(df)],'linewidth',1);
% get performance
reshape(dv2conf(z.Y,df),[2 2]),conf2loss(ans),
%
jf_cvtrain(z,'cvtrainFn','cvtrainMap','lossFn','est2loss','objFn','wienerClsfr','Cs',0,'clsfr',1,'bias',1,'ydim',1);

% plot the classifier weight vector
W=reshape(wb(1:end-(bias>0)),size(z.X,1)-(bias>0),numel(taus));
clf;plot(W','linewidth',1);legend
pat=taucov3dto2d(mean(z.X(1:size(z.X,1)-(bias>0),1:size(z.X,1)-(bias>0),:,:),4),0)*wb(1:end-(bias>0));
clf;image3d(cat(3,W,reshape(pat,size(W))),1,'zvals',{'filt','pat'},'disptype','plot','plotopts',{'linewidth',1});

% plot the filtered data / re-cons stimuli
XW=stfilter(oz.X,wb(1:end-(bias>0)),taus,wb(end).*(bias>0));
clf;mcplot([mean(oz.X(1,:,oz.Y>0),3)' mean(XW(:,:,oz.Y>0),3)' mean(oz.X(1,:,oz.Y<0),3)' mean(XW(:,:,oz.Y<0),3)'],'linewidth',1);legend('Y(1)','X(1)','XW(1)','Y(2)','X(2)','XW(2)');
