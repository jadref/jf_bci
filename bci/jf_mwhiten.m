function [z]=jf_mwhiten(z,varargin);
% multi-dimensional whiten the input data, per given dimensions entries
%
% Options:
%  dim -- spec of the dimension(s) to whiten over.         ('ch')
%  stepDim -- whiten per each entry in these dims
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
%  testRecompute -- [bool] flag that we compute a new whitener for test-cases also (0)
%  whtFeat -- 'str' type of feature to use when computing the whitener.
%             one-of: '' - input, 'abs' - absolute length, 'arg' - phase angle
%
% Examples:
%  
opts=struct('dim','ch','stepDim',[],'center',0,'stdp',0,'symp',1,'linMapMx',[],'tol',[],'unitCov',[],...
            'reg',[],'order',[],'subIdx',[],'blockIdx',[],'whtFeat',[],...
				'verb',0,'testRecompute',0);
[opts]=parseOpts(opts,varargin);

szX=size(z.X); nd=ndims(z.X);
dim=n2d(z,opts.dim);
dim=sort(dim,'ascend'); % ensure in ascending order
stepDim=n2d(z,opts.stepDim);

% compute the permutation parameters if needed
perm=[];
if ( numel(dim)>1 )
  if( any(diff(dim)~=1) ) error('not supported yet-- permute first!'); end;  
%  [md,mi]=min(dim); permd=sort(dim([1:md-1 md+1:end]),'ascend');
%  perm=[1:md permd setdiff(md+1:ndims(z.X),permd)];
%  dim=md+(0:numel(dim)-1); % new set of dims to compress
end

% update stepDims after we've reshaped the input
rstepDim = stepDim; rstepDim(rstepDim>dim(1))=rstepDim-numel(dim)-1; % moved step-dims

wX = z.X;
if ( ~isempty(opts.whtFeat) ) % pre-transform the data to compute the whitener on
  switch lower(opts.whtFeat);
	 case 'abs'; wX=abs(wX);
	 case {'arg','angle'}; if ( iscomplex(wX) ) wX=arg(wX); end;
	 otherwise; error('Unknown white feature transformation');
  end
end
% call whiten to do the actual work
if ( ~isempty(opts.blockIdx) ) % whiten in blocks
  blockIdx=getBlockIdx(z,opts.blockIdx);
  % convert to indicator
  if ( size(blockIdx,2)==1 ) blockIdx=gennFold(blockIdx,'llo'); end
  if ( ndims(blockIdx)==3 )
	 if ( size(blockIdx,2)~=1 ) error('Not supported yet!'); end
	 blockIdx=blockIdx(:,:);
  end
  % whiten each of the blocks independently
  for bi=1:size(blockIdx,2);
	 bidx=subsrefDimInfo(z.di,'dim','epoch','idx',blockIdx(:,bi)>0); % which subset
								  % extract out the block we're currently processing
	 bX=wX(bidx{:});
										  % compress out the target dims
	 sszX=size(bX); bX=reshape(bX,[sszX(1:dim(1)-1) prod(sszX(dim)) sszX(max(dim)+1:end)]);	 
										  % actually compute and apply the whitener
	 [Wbi,Sigmabi,Ubi,Dbi] = whiten(bX,[dim(1) rstepDim],opts.reg,opts.center,opts.stdp,opts.symp,opts.linMapMx,opts.tol,opts.unitCov,opts.order);
										  % undo the reshape on the paramters
	 tmp=size(Wbi);     Wbi  =reshape(Wbi,[szX(dim),szX(dim),tmp(3:end)]); %BODGE: assume symetric whitener computed...
	 tmp=size(Ubi);     Ubi  =reshape(Ubi,[szX(dim),tmp(2:end)]);
	 tmp=size(Sigmabi); Sigmabi=reshape(Sigmabi,[szX(dim) szX(dim) tmp(3:end)]);	 
% BODGE: using cell's is a hack, and will probably break other things later...
	 U{bi}=Ubi;
	 W{bi}=Wbi;
	 D{bi}=Dbi; 
	 Sigma{bi}=Sigmabi;
										  % apply this whitener to the raw data
	 idx=1:ndims(z.X); idx(dim)=-dim;
	 z.X(bidx{:}) = tprod(z.X(bidx{:}),idx,Wbi,[-dim dim stepDim]); 
  end

else
  if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
	 bidx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
	 wX=wX(bidx{:});
  end
										  % compress the whiten dims together
  sszX=size(wX); wX=reshape(wX,[sszX(1:min(dim)-1) prod(sszX(dim)) sszX(max(dim)+1:end)]);  
										  % actually compute and apply the whitener
  [W,Sigma,U,D] = whiten(wX,[dim(1) rstepDim],opts.reg,opts.center,opts.stdp,opts.symp,opts.linMapMx,opts.tol,opts.unitCov,opts.order);
										  % undo the reshape on the parameters
  tmp=size(W);     W  =reshape(W,[szX(dim),szX(dim),tmp(3:end)]);%BODGE: assume symetric whitener computed...
  tmp=size(U);     U  =reshape(U,[szX(dim),tmp(2:end)]);
  tmp=size(Sigma); Sigma=reshape(Sigma,[szX(dim) szX(dim) tmp(3:end)]);

										  % compute fully whitened dataset
  idx=1:ndims(z.X); idx(dim)=-dim;
  z.X = tprod(z.X,idx,W,[-dim(:);dim(:);stepDim]); 

end

% Construct whitening matrix dimInfo structure
wDi=mkDimInfo(size(W),{z.di(dim).name,z.di(dim).name,z.di(stepDim).name});
wDi(1:numel(dim))=z.di(dim);
if ( opts.symp ) % new info is same as old info
  for di=1:numel(dim);
	 wDi(numel(dim)+di)      = z.di(dim(di));
	 wDi(numel(dim)+di).name = [wDi(di).name '_wht'];
  end;
end
for di=1:numel(stepDim); wDi(2*numel(dim)+di)=z.di(stepDim(di)); end;

% update the dimInfo
z.di(dim)=wDi(numel(dim)+(1:numel(dim)));
summary='';
if ( opts.symp ) summary=[summary 'sym ']; end;
if ( ~isempty(opts.order) ) summary=[summary sprintf('(%g order) ',opts.order)]; end;
if ( opts.reg > 0 ) summary=[summary sprintf('(%gReg) ',opts.reg)]; end;
summary=[summary 'over [' sprintf('%s+',wDi(1:numel(dim)).name) ']'];
if ( ~isempty(stepDim) ) 
   summary=[summary 'for each ' sprintf('%s,',z.di(stepDim).name)]; 
end;
if ( ~isempty(opts.blockIdx) )
  summary=[summary sprintf('in %d blocks',size(blockIdx,2))];
else
  summary=sprintf('%s, %d wht components',summary,size(W,2));
end
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
  z=jf_welchpsd(z,'width_ms',250);
w=jf_mwhiten(z,'dim',{'ch','freq'});

clf;mimage(w.prep(end).info.U(:,:,0+(1:9)));

% block test
w=jf_mwhiten(z,'dim',{'ch','freq'},'blockIdx',[ones(floor(size(z.X,3)/2),1); 2*ones(ceil(size(z.X,3)/2),1)]); % 2 blocks

clf;mimage(w.prep(end).info.U{1}(:,:,0+(1:9)));
