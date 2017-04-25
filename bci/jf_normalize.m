function [z]=jf_normalize(z,varargin);
% normalise along a given dimension, i.e. make all entries have unit power
%
% Options:
%  dim    -- the dimension(s) over which we compute the statistics (i.e. RMS-power/mean) (ndims(z.X))
%  featdim-- the dimension(s) which are to be normalized
%  feat   -- one of: {'power','rms'} - make unit-power,  {'offset','mean'} - make zero-mean ('power')
%  type   -- *only-when feat=='offset'*.  one of: 'rel' - divide by ave feat, 'abs' - subtract ave feat ([])
%  minStd -- min allowed std                      (1e-5)
%  zeronan -- [bool] set elements which are nan to value 0    (0)
%  wght    -- [n-d] weighting over points for the compuation  ([])
% TODO: convert so dim=dimension to normalize as it's more intutive
opts=struct('dim',[],'featdim',[],'feat','power','type','','minStd',1e-5,'summary',[],'subIdx',[],'verb',0,'zeronan',0,'wght',[],'blockIdx',[]);
opts=parseOpts(opts,varargin);
dim=opts.dim; featdim=opts.featdim; 
if(~isempty(featdim) ) 
   featdim=n2d(z,featdim); dim=setdiff(1:ndims(z.X),featdim); %N.B. featdim overrides dim...
else
   if(isempty(dim)) dim=-1; end;
   dim=n2d(z,dim,0,0);dim(dim==0)=[];
   featdim=setdiff(1:ndims(z.X),dim);
end 
stds=[];
if ( isempty(opts.blockIdx) ) 
   if ( strcmp(opts.feat,'power') )
      [z.X,stds] = normalize(z.X,dim,[],opts.minStd,opts.zeronan,opts.wght);
   elseif ( strcmp(opts.feat,'offset') )
      [z.X,stds] = center(z.X,dim,opts.type,opts.zeronan);
   end
else
  [blockIdx,blkD,blkfound]=getBlockIdx(z,opts.blockIdx);
  % convert to indicator
  if ( size(blockIdx,2)==1 )
	 if( numel(opts.blockIdx)>numel('block') ) % single-block whitening
		bi=str2num(opts.blockIdx(numel('block')+1:end)); % get the blockID to use
		blockIdx= (blockIdx==bi); % mark everything but this block as excluded
	 else % every block normalized independently
		blockIdx=gennFold(blockIdx,'llo');
	 end
  end
  if ( ndims(blockIdx)==3 )
	 if ( size(blockIdx,2)~=1 ) error('Not supported yet!'); end
	 blockIdx=blockIdx(:,:);
  end
   for bi=1:size(blockIdx,2);
      blkidx=subsrefDimInfo(z.di,'dim','epoch','idx',blockIdx(:,bi)>0); % which subset
      if ( any(strcmp(opts.feat,{'power','rms'})) )
         [zX,stds{bi},op]=normalize(z.X(blkidx{:}),dim,[],opts.minStd,opts.zeronan,opts.wght);
      elseif ( any(strcmp(opts.feat,{'offset','mean'})) )
         [zX,stds{bi},op]=center(z.X(blkidx{:}),dim,opts.type,opts.zeronan);
      else
         error('Unrecognised feature type: %s!',opts.feat);
      end
		if( size(blockIdx,2)>1 ) % apply only to this block
		  z.X(blkidx{:})=zX;
		else % apply to all the data
		  z.X=repop(z.X,op,stds{bi});
		end
   end
end
summary = [opts.feat ' ' opts.type ' for each ' sprintf('%s+',z.di(featdim).name)];
if ( ~isempty(opts.summary) ) summary=[summary sprintf(' (%s)',opts.summary)]; end;
if ( ~isempty(opts.blockIdx) )
  summary=[summary sprintf(' in %d blocks',size(blockIdx,2))];
end
info= struct('stds',stds);
z   = jf_addprep(z,mfilename,summary,opts,info);
return;
%----------------------------------------------------------------------
function testCase()

zn = jf_normalize(z)
zn = jf_normalize(z,'blockIdx','block');
