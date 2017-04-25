function z=jf_addFolding(z,varargin)
% add folding information to a jf data structure
%
% Options:
%  dim   - [] dimension of z.Y which contain the sub-problems
%  nFold - [int] number of folds to make
%  perm  - [bool] permute the label order before generating folds (0)
%  foldSize - [1x1] number of points in each fold (N/nFold)
%               OR
%               [nClass x 1] number of points of each class for each fold
%  randseed - [1x1] seed for the random number generator
%  repeats  - [1x1] how many sets of nFolds to generate (1)
%  zeroLab  - [bool] do we treat 0 labels as ignored or true labels (0)
%
% See Also : gennFold
opts=struct('dim',[],'nFold',[],'outfIdxs',[]);
[opts,varargin]=parseOpts(opts,varargin);
dim=opts.dim;
if ( isfield(z,'Y') ) 
  Y=z.Y; 
  if ( isempty(dim) ) 
	 dim=ndims(Y);
	 if ( isfield(z,'Ydi') ) dim=n2d(z.Ydi,'subProb',0,0); end;
  else
     dim=n2d(z.Ydi,dim);
  end
else
  if ( isempty(dim) ) dim=ndims(z.X); end;
  Y=ones(size(z.X,dim),1);
  dim=2;
end
summary=''; % logging summary string

outfIdxs=opts.outfIdxs;
if ( isempty(outfIdxs) && isfield(z,'outfIdxs') ) outfIdxs=z.outfIdxs; end;
if ( ischar(opts.nFold) && strncmpi(opts.nFold,'testblock',numel('testblock')) ) % 1st blk folding -> outer fold info
  blockIdx=getBlockIdx(z,opts.nFold);
  % setup outer fold info
  if( numel(opts.nFold)>numel('testblock') ) % single-block whitening
	 bi=str2num(opts.nFold(numel('testblock')+1:end)); % get the blockID to use
    if( ~any(blockIdx==bi) ) tmp=unique(blockIdx); bi=tmp(bi); end; % if no blockX then use xth block    
	 outfIdxs= (blockIdx==bi); % mark everything but this block as excluded
  end
  summary=opts.nFold;
  opts.nFold=10;
end
if ( ~isempty(outfIdxs) ) 
if ( islogical(outfIdxs) )
  tmp=outfIdxs; outfIdxs=-ones(size(outfIdxs)); outfIdxs(tmp)=1;
elseif( isnumeric(outfIdxs) && min(outfIdxs)>=1 && max(outfIdxs)>1 ) % indices
  tmp=outfIdxs; outfIdxs=-ones(size(z.Y,1),1); outfIdxs(tmp)=1;
end
end

if ( ischar(opts.nFold) && any(strncmpi(opts.nFold,'lbo',3)) ) % leave block out -> leave label out
  summary='leave block out';
  inblkfrac=[];
  if( numel(opts.nFold)>numel('lbo') && opts.nFold(numel('lbo')+1)=='+' ) % leave-most-of-block-out
     inblkfrac=str2num(opts.nFold(numel('lbo')+2:end)); % amount from the block to use
  end
  [Yl,blkD,blkFound]=getBlockIdx(z,'block');
  if ( ischar(blkD) ) blkD=n2d(z.Ydi,blkD); end;
  if( blkFound )
     szY=size(Y);
     Yl = reshape(Yl,[szY([1:dim-1 dim+1:end]) 1 1]); % ensure is same size as Y
     z.foldIdxs=gennFold(Yl,[],'dim',dim,'nFold','llo',varargin{:});
     if( ~isempty(inblkfrac) ) % add the extra within-block examples
        fldsz=size(z.foldIdxs); if( sum(fldsz>1)~=2 ) error('Only for [N x 1 x nFold] folding currently'); end;
        fIdxs=reshape(z.foldIdxs,[fldsz(fldsz>1) 1]);
        for fi=1:size(z.foldIdxs,ndims(z.foldIdxs));
           % get the testing examples for this fold
           tstIdx = find(fIdxs(:,fi)>0);
           fSize = ceil(numel(tstIdx)*inblkfrac);
           fIdxs(tstIdx(1:min(end,fSize)),fi)=-1; % mark as extra training examples
        end
        z.foldIdxs=reshape(fIdxs,fldsz);
        summary = sprintf('%s +(%3.2f)',summary,inblkfrac);
     end
  else % no-block info, fall back on loo
     z.foldIdxs=gennFold(z.Y,[],'dim',dim,'nFold','loo',varargin{:});
  end

elseif ( ischar(opts.nFold) && strncmpi(opts.nFold,'block',numel('block')) ) % per-block nFold
   summary='per-block';
   blockIdx=opts.nFold;
   if( numel(opts.nFold)>numel('block') ) % single-block
      blockIdx  =opts.nFold(1:numel('block'));
      opts.nFold=str2num(opts.nFold(numel('block')+1:end)); % number folds to use
   end
   [blockIdx,blkD]=getBlockIdx(z,blockIdx);
   if ( ischar(blkD) ) blkD=n2d(z.Ydi,blkD); end;
   bId=unique(blockIdx); nBlock  =numel(bId); 
   summary = sprintf('%s %d-blocks',summary,nBlock);
   % loop over blocks getting the folding info for each...   
   szY=size(z.Y); szY(end+1:max(dim,blkD))=1; 
   szfIdxs=[ones(1,numel(szY)) abs(opts.nFold)]; szfIdxs(blkD)=szY(blkD); % only blkD has fold info
   z.foldIdxs=zeros(szfIdxs);
   %index expr to get the elements of Y for this block, N.B. assume all dim except spD and blkD have identical values
   idx=repmat({':'},numel(szY),1); [idx(setdiff(1:numel(idx),[dim,blkD]))]={1};
   fidx=repmat({':'},numel(szfIdxs),1); % index exprfor the folding info
   for bi=bId(:)';
      blkIdx=(blockIdx==bi);
      idx{blkD}=blkIdx;
      fidx{blkD}=blkIdx;
      z.foldIdxs(fidx{:}) = gennFold(z.Y(idx{:}),[],'dim',dim,'nFold',opts.nFold,varargin{:});
   end
 
elseif ( ~isempty(outfIdxs) ) % inner/outer folding
  summary='inner/outer folding';
  % setup 1st block only folding
  trnInd   = outfIdxs>0;
  foldIdxs = gennFold(Y(trnInd,:,:),[],'dim',dim,'nFold',opts.nFold,varargin{:});
  if ( ndims(Y)>2 ) warning('Multi-dim Y with block folding not supported yet!'); end;
  z.foldIdxs=zeros(size(Y,1),10);   % full-size folding matrix
  z.foldIdxs(trnInd,1:10)=foldIdxs; % insert the 10-fold for the 1st block elements
  % setup the outer-folding matrix
  z.outfIdxs=zeros(size(Y,1),1);z.outfIdxs(trnInd)=-1; z.outfIdxs(~trnInd)=1;

elseif( ischar(opts.nFold) && (strncmp(opts.nFold,'ldo',3) || strncmp(opts.nFold,'lndo',4)) ) % leave dim out, leave-n-dim-out
   fdim = opts.nFold(strfind(opts.nFold,'_')+1:end);
   fdim = n2d(z.Ydi,fdim); % get dim to work along
       % make new labelling with different values for each entry along fdim
   szY  = size(Y);
   if( strncmp(opts.nFold,'ldo',3) ) nFold= 'llo'; else nFold='lno'; end;
   z.foldIdxs=gennFold([1:size(z.Y,fdim)]',[],'dim',2,'nFold',nFold,varargin{:});
   z.foldIdxs=reshape(z.foldIdxs,[ones(1,fdim-1) size(z.foldIdxs,1) ones(1,numel(szY)-(fdim+1)) size(z.foldIdxs,ndims(z.foldIdxs))]);
   summary=sprintf('leave dim %s out',z.Ydi(fdim).name);

else % normal call to gennFold
  z.foldIdxs=gennFold(Y,[],'dim',dim,'nFold',opts.nFold,varargin{:});

end
z=jf_addprep(z,mfilename,sprintf('%s %d folds',summary,size(z.foldIdxs,ndims(z.foldIdxs))),[],[]);
return;
										  %---------------------------
function testCase()
  z=jf_addFolding(z); % normal 10-fold
  z=jf_addFolding(z,'outfIdxs',1:50); % 1st 50 for cv, rest for testing
