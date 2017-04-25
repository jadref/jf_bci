function [z]=jf_matrixwhiten(z,varargin);
% matrix-covariance based whitening of the input data 
%
% Options:
%  dim -- spec of the dimension(s) to whiten over.         ('ch')
%  stepDim -- whiten per each entry in these dims
%  tol     -- [float] tolerance to detect zero eigenvalues in comp inverse covariance ([])
%  unitCov -- [bool] re-scale covariance to have unit power for numercial stability ([])
%  reg     -- [float] regularisation parameter for covariance inverse ([])
%  subIdx  -- {nd x 1} indices of data points to use in computing the inverse
%
% Examples:
%  
opts=struct('dim','ch','stepDim',[],'symp',1,'tol',[],'maxIter',[],...
            'reg',[],'subIdx',[],'blockIdx',[],'whtFeat',[],...
				'verb',1);
[opts]=parseOpts(opts,varargin);

szX=size(z.X); nd=ndims(z.X);
dim=n2d(z,opts.dim);
dim=sort(dim,'ascend'); % ensure in ascending order
stepDim=n2d(z,opts.stepDim);

% call matcov to do the actual work
wX=z.X;
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
										  % actually compute the whitener
	 [Sigmasbi,Wsbi]=matcov(bX,dim,stepDim,opts.reg,opts.tol,opts.whtFeat,opts.maxIter,opts.verb);
	 Ws{bi}=Wsbi;
	 Sigmas{bi}=Sigmasbi;
										  % apply this whitener to the raw data
	 for dii=1:numel(dim);
		z.X(bidx{:})=tprod(z.X(bidx{:}),[1:dim(dii)-1 -dim(dii) dim(dii)+1:ndims(wX)],...
								 Wsbi{dii},[-dim(dii) dim(dii) stepDim]); 
	 end;
  end

else
  if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
	 bidx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
	 wX=wX(bidx{:});
  end
										  % compute the whitener
  [Sigmas,Ws]=matcov(wX,dim,stepDim,opts.reg,opts.tol,opts.whtFeat,opts.maxIter,opts.verb);
										  % apply the whitener to all the raw data
  for dii=1:numel(dim);
	 z.X=tprod(z.X,[1:dim(dii)-1 -dim(dii) dim(dii)+1:ndims(wX)],Ws{dii},[-dim(dii) dim(dii) stepDim]); 
  end;
end


										  % update the dimInfo
odi=z.di;
for di=1:numel(dim);
  z.di(dim(di)).name= [z.di(dim(di)).name '_wht'];
end;
summary='';
if ( opts.symp ) summary=[summary 'sym ']; end;
if ( opts.reg > 0 ) summary=[summary sprintf('(%gReg) ',opts.reg)]; end;
summary=[summary 'over [' sprintf('%s+',odi(dim).name) ']'];
if ( ~isempty(stepDim) ) 
   summary=[summary 'for each ' sprintf('%s,',z.di(stepDim).name)]; 
end;
if ( ~isempty(opts.blockIdx) )
  summary=[summary sprintf('in %d blocks',size(blockIdx,2))];
else
  summary=sprintf('%s, %d wht components',summary,size(Ws,2));
end
if ( ~isempty(opts.whtFeat) )
  summary=[summary ' using {' sprintf('%s,',opts.whtFeat{:}) '} features'];
end;
if ( ~iscell(Ws) ) 
  info=struct('Ws',Ws,'Sigmas',Sigmas);
else
  info=struct('Ws',{Ws},'Sigmas',{Sigmas});
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
z=jf_welchpsd(z,'width_ms',250);
jf_cvtrain(z)

w=jf_matwhiten(z,'dim',{'ch','freq'});
jf_cvtrain(w);

% block test
w=jf_matwhiten(z,'dim',{'ch','freq'},'blockIdx',[ones(floor(size(z.X,3)/2),1); 2*ones(ceil(size(z.X,3)/2),1)]); % 2 blocks
