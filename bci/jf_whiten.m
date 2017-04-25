function [z]=jf_whiten(z,varargin);
% whiten the input data, per given dimensions entries
%
% Options:
%  dim -- spec of the dimensions to whiten over.         ('ch')
%         dim(1)=dimension to whiten
%         dim(2:end) whiten per each entry in these dims
%  center - [bool] do we center before whitening (0)
%  stdp -- [bool] do we standardize each channel before whitening (0)
%  symp -- [bool] do we generate a symetric whitening (1)
%  linMapMx - [size(X,dim(2:end)) x size(X,dim(2:end))] linear mapping over non-acc dim 
%              use to smooth over these dimensions ([])
%  tol     -- [float] tolerance to detect zero eigenvalues in comp inverse covariance ([])
%  unitCov -- [bool] re-scale covariance to have unit power for numercial stability ([])
%  reg     -- [float] regularisation parameter for covariance inverse ([])
%  alpha   -- [float] 1-reg = strength of the whitening ([])
%  order   -- [float] order of inverse to use, e.g. .5 = square root  (.5)
%  subIdx  -- {nd x 1} indices of data points to use in computing the inverse
%  blockIdx -- 'str' OR [int] whiten these blocks independently
%  goodOnly -- [bool] if true then do *not* use trials marked as bad (extra(i).isbad==true) to
%                     compute the whitener
%  testRecompute -- [bool] flag that we compute a new whitener for test-cases also (0)
%  whtFeat -- 'str' type of feature to use when computing the whitener.
%             one-of: '' - input, 'abs' - absolute length, 'arg' - phase angle
%  
% Examples:
%  
opts=struct('dim','ch','center',0,'stdp',0,'symp',1,'linMapMx',[],'tol',[],'unitCov',[],...
            'alpha',[],'reg',[],'order',[],'subIdx',[],'goodOnly',0,'blockIdx',[],'whtFeat',[],...
				'verb',0,'testRecompute',0);
[opts]=parseOpts(opts,varargin);

szX=size(z.X); nd=ndims(z.X);
dim=n2d(z,opts.dim);
covIn=false;
if ( dim(1)==0 ) covIn=true; 
elseif ( n2d(z,[z.di(dim(1)).name '_2'],0,0)==dim(1)+1 ) covIn=true; 
end;
if( covIn ) dim(1)=0; end; % mark as covariance input to inner function

% convert alpha to a reg-paramter if needed
if( ~isempty(opts.alpha) && isempty(opts.reg) ) 
   opts.reg=opts.alpha; 
   if ( isnumeric(opts.alpha) ) 
      if ( opts.alpha<=1  && opts.alpha>0 ) opts.reg= 1-opts.alpha; end
      if ( opts.alpha>=-1 && opts.alpha<0 ) opts.reg=-(1+opts.alpha); end
      if ( opts.alpha<-1 )                  opts.reg=-(szX(dim(1))+opts.alpha); end;  
   end
end

% get the input to the inner whitener function
wX = z.X;
if ( ~isempty(opts.whtFeat) ) % pre-transform the data to compute the whitener on
  switch lower(opts.whtFeat);
	 case 'abs'; wX=abs(wX);
	 case {'arg','angle'}; if ( iscomplex(wX) ) wX=arg(wX); end;
	 otherwise; error('Unknown white feature transformation');
  end
end

			 % get list of bad examples if needed+available
isbad=[];
if( opts.goodOnly && isfield(z.di(n2d(z,'epoch')).extra,'isbad') )
  isbad=cat(1,z.di(n2d(z,'epoch')).extra.isbad);
end

% call whiten to do the actual work
if ( ~isempty(opts.blockIdx) ) % whiten in blocks
  [blockIdx,blkD]=getBlockIdx(z,opts.blockIdx);
  % exclude the bad examples from being used to compute the whitener
  if ( ~isempty(isbad) ) blockIdx(isbad,:)=0; end; 
  % convert to indicator
  if ( size(blockIdx,2)==1 )
	 if( numel(opts.blockIdx)>numel('block') ) % single-block whitening
		bi=str2double(opts.blockIdx(numel('block')+1:end)); % get the blockID to use
      if( ~any(blockIdx==bi) ) tmp=unique(blockIdx); bi=tmp(bi); end; % if no blockX then use xth block
		blockIdx= (blockIdx==bi); % mark everything but this block as excluded
	 else % every block whitened independently
		blockIdx=gennFold(blockIdx,'llo','zeroLab',0);
	 end
  end
  if ( ndims(blockIdx)==3 )
	 if ( size(blockIdx,2)~=1 ) error('Not supported yet!'); end
	 blockIdx=blockIdx(:,:);
  end
  % whiten each of the blocks independently
  for bi=1:size(blockIdx,2);
	 bidx=subsrefDimInfo(z.di,'dim',blkD,'idx',blockIdx(:,bi)>0); % which subset
	 [Wbi,Sigmabi,Ubi,Dbi] = whiten(wX(bidx{:}),dim,opts.reg,opts.center,opts.stdp,opts.symp,opts.linMapMx,opts.tol,opts.unitCov,opts.order);
	 % BODGE: using cell's is a hack, and will probably break other things later...
	 W{bi}=Wbi;
	 D{bi}=Dbi;
	 U{bi}=Ubi; 
	 Sigma{bi}=Sigmabi;
		% apply the whitener to the data to compute the fully whitened dataset
	 if ( covIn ) % covariance input
		% apply to all the data
		if( size(blockIdx,2)==1 ) bidx{n2d(z,blkD)}=1:size(z.X,n2d(z,blkD)); end
		dim(1)=1;idx=1:ndims(z.X); idx(dim(1))=-dim(1);
		z.X(bidx{:}) = tprod(z.X(bidx{:}),idx,Wbi,[-dim(1) dim(1) dim(2:end)]); 
		dim(1)=2;idx=1:ndims(z.X); idx(dim(1))=-dim(1);
		z.X(bidx{:}) = tprod(z.X(bidx{:}),idx,Wbi,[-dim(1) dim(1) dim(2:end)]);
	 else
		idx=1:ndims(z.X); idx(dim(1))=-dim(1);
		if( size(blockIdx,2)==1 ) % apply to all the data
		  z.X  = tprod(z.X,idx,Wbi,[-dim(1) dim(1) dim(2:end)]);
		else % apply only to this blocks data
		  z.X(bidx{:}) = tprod(z.X(bidx{:}),idx,Wbi,[-dim(1) dim(1) dim(2:end)]);
		end
	 end
  end

else
  if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
	 bidx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
	 if( ~isempty(isbad) )
		epIdx=bidx{n2d(z,'epoch')};
		if ( islogical(epIdx) )  bidx{n2d(z,'epoch')}= epIdx & ~isbad;
		else                     bidx{n2d(z,'epoch')}= setdiff(bidx{n2d(z,'epoch')},find(isbad));
		end;
	 end
	 wX=wX(bidx{:});
  elseif ( ~isempty(isbad) ) % only use the non-bad marked trials
	 bidx=subsrefDimInfo(z.di,'dim','epoch','idx',~isbad);
	 wX=wX(bidx{:});	 
  end
										  % compute the whiten transform
  [W,Sigma,U,D] = whiten(wX,dim,opts.reg,opts.center,opts.stdp,opts.symp,opts.linMapMx,opts.tol,opts.unitCov,opts.order);
										  % apply the whitener to all the data
  if ( covIn ) % covariance input
	 dim(1)=1;idx=1:ndims(z.X); idx(dim(1))=-dim(1); z.X = tprod(z.X,idx,W,[-dim(1) dim(1) dim(2:end)]); 
	 dim(1)=2;idx=1:ndims(z.X); idx(dim(1))=-dim(1); z.X = tprod(z.X,idx,W,[-dim(1) dim(1) dim(2:end)]);
  else
	 idx=1:ndims(z.X); idx(dim(1))=-dim(1);
	 z.X = tprod(z.X,idx,W,[-dim(1) dim(1) dim(2:end)]); 
  end
end

if ( covIn ) dim(1)=1; end;
% Construct whitening matrix dimInfo structure
wDi=mkDimInfo(size(W),{z.di(dim(1)).name,[z.di(dim(1)).name '_wht']});
wDi(1)=z.di(dim(1));
if ( opts.symp ) % new info is same as old info
  wDi(2)=wDi(1); wDi(2).name=[wDi(2).name '_wht'];
end
for di=2:numel(dim); wDi(di+1)=z.di(dim(di)); end;

% update the dimInfo
z.di(dim(1))=wDi(2);
summary='';
if ( opts.symp ) summary=[summary 'sym ']; end;
if ( ~isempty(opts.order) ) summary=[summary sprintf('(%g order) ',opts.order)]; end;
if ( opts.reg > 0 ) summary=[summary sprintf('(%gReg) ',opts.reg)]; end;
summary=[summary sprintf('over %s ',wDi(1).name)];
if ( numel(dim)>1 ) 
   summary=[summary 'for each ' sprintf('%s,',wDi(3:end-1).name)]; 
end;
if ( ~isempty(opts.blockIdx) )
  summary=[summary sprintf('in %d blocks',size(blockIdx,2))];
else
  summary=sprintf('%s, %d wht comp''nts',summary,size(W,2));
end
if ( covIn ) summary=[summary ' (covIn)']; end;
if ( ~isempty(opts.whtFeat) ) summary=[summary ' using ' opts.whtFeat ' features']; end;
if ( ~iscell(W) ) 
  info=struct('W',W,'wDi',wDi,'D',D,'U',U,'Sigma',Sigma);
else
  info=struct('W',{W},'wDi',wDi,'D',{D},'U',{U},'Sigma',{Sigma});
end
if ( ~opts.testRecompute ) % apply this filter at test time
  info.testFn={'jf_linMapDim' 'mx' W 'di' wDi};
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
w=jf_whiten(z);
jf_disp(w)
clf;image3ddi(w.X,w.di,1);
clf;jplot([w.prep(m2p(w,'jf_whiten')).info.wDi(1).extra.pos2d],w.prep(m2p(w,'jf_whiten')).info.W(:,:),'clim','cent0');

% test application. N.B. we need to center to apply correctly!
wX=tprod(repop(z.X,'-',mean(z.X(:,:),2)),[-1 2 3],w.prep(end).info.W(:,:),[-1 1]);
mad(wX,w.X)


% test with complex inputs
zc   = jf_center(z,'dim','time');

zcfw =jf_whiten(jf_fft(zc),'dim','ch','center',0); % fourier -> whiten
zcwf =jf_fft(jf_whiten(zc,'dim','ch','center',0)); % whiten -> fourier

jf_whiten(zc,'dim','ch','blockIdx',z.foldIdxs)


% test with per-example whitening = trial-whiten
trw=jf_whiten(z,'dim',{'ch' 'epoch'});
% per-example with a smoothing over 10-trials
tr10w=jf_whiten(z,'dim',{'ch' 'epoch'},'linMapMx',ones(1,10));


% test with block whitening
w=jf_mwhiten(z,'blockIdx',[ones(floor(size(z.X,3)/2),1); 2*ones(ceil(size(z.X,3)/2),1)]); % 2 blocks
w=jf_whiten(z,'blockIdx',vec(repmat(1:5,20,1)));
w=jf_whiten(z,'blockIdx','block');
w=jf_whiten(z,'blockIdx','fold');

trwb=jf_whiten(z,'dim',{'ch' 'epoch'},'blockIdx',[ones(50,1); 2*ones(50,1)]);
