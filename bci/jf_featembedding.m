function [z]=jf_featembedding(z,varargin)
% computed the weighted mean and remove a particular dimension
% Options:
%  dim -- the dim(s) used to compute the new features ('epoch')
%  type -- one-f : 'const' simple constant feature
%                  'mean' average feature value over dimensions 'dim'
%                  'tauX' previous X epoch features over dimension 'dim'
%                  'ar'   weighted combination of the X previous 'dim' features
%  wght-- weighting for the included elements, average over idx if empty. ([])
%  summary -- additional summary string
%  
opts=struct('dim','epoch','wght',[],'summary','','subIdx',[],'verb',0,'blockIdx',[],'type','const');
[opts,varargin]=parseOpts(opts,varargin);
dim = n2d(z.di,opts.dim);

if ( isempty(opts.blockIdx) ) 
   z.X = featEmbedding(z.X,dim,opts.type,opts.wght);
else
  blockIdx=getBlockIdx(z,opts.blockIdx);
  % convert to indicator
  if ( size(blockIdx,2)==1 ) blockIdx=gennFold(blockIdx,'llo'); end
  if ( ndims(blockIdx)==3 )
	 if ( size(blockIdx,2)~=1 ) error('Not supported yet!'); end
	 blockIdx=blockIdx(:,:);
  end
  % loop over the blocks adding the features
  nX=[]; szX=size(z.X);
  for bi=1:size(blockIdx,2);
     blkidx=subsrefDimInfo(z.di,'dim','epoch','idx',blockIdx(:,bi)>0); % which subset
     Xbi = featEmbedding(z.X(blkidx{:}),dim,opts.type,opts.wght,bi);
     if ( isempty(nX) ) nX=zeros([size(Xbi,1) szX(2:end)]); end
     nX(1:size(Xbi,1),blkidx{2:end})=Xbi;
  end
  z.X=nX;
end

if ( iscell(z.di(1).vals) )
   for fi=numel(z.di(1).vals)+1:size(z.X,1); z.di(1).vals{fi}=sprintf('embed_%d',fi); end;
elseif ( isnumeric(z.di(1).vals) ) 
   for fi=numel(z.di(1).vals)+1:size(z.X,1); z.di(1).vals(fi)=fi; end;   
end
z.di(1).name = [z.di(1).name '_embed'];
% update the meta-info
summary='';
if ( ischar(opts.type) ) summary=opts.type; 
elseif(isnumeric(opts.type)) summary=sprintf('const=%g',opts.type); 
end;
if ( ~isempty(opts.wght) ) summary =sprintf('%s wght(%d)',summary,numel(opts.wght)); end;
summary = [summary ' along ' sprintf('%s,',z.di(dim(1:end-1)).name) sprintf('%s',z.di(dim(end)).name)];
if( numel(dim)>1 ) summary = [summary sprintf('+%s',odi(dim(2:end)).name)]; end;
if(~isempty(opts.summary)) summary=[summary  sprintf(' (%s)',opts.summary)];end
info=[];
z=jf_addprep(z,mfilename,summary,opts,info);
return;

function X=featEmbedding(X,dim,type,wght,bi)
if ( nargin<5 ) bi=[]; end;
szX = size(X);
idx={}; for di=1:ndims(X); idx{di}=1:szX(di); end;

if ( isnumeric(type) || strcmp(type,'const') ) % add a simple constant feature
   idx{1}=szX(1)+1;
   const=1; if ( isnumeric(type) ) const=type; end;
   if ( ~isempty(bi) ) % make a new feature for each block
      if ( dim(1)~=2 ) idx{1} = szX(1)+ceil(bi/szX(2));  idx{2}=mod(bi-1,szX(2))+1; 
      else             idx{1} = szX(1)+ceil(bi/szX(3));  idx{3}=mod(bi-1,szX(3))+1; 
      end;
   end;
   X(idx{:})=const;

elseif ( strcmp(type,'mean') ) % average over dim feature
   mu  =X; for di=1:numel(dim); mu=sum(mu,dim(di)); end; mu=mu./prod(szX(dim));
   rszMu=ones(size(szX)); rszMu(dim)=szX(dim);
   X   =cat(1,X,repmat(mu,rszMu));

elseif ( strcmp(type,'tau') )
   nszX=szX; nszX(1)=nszX(1)*(1+numel(wght));
   nX     =zeros(nszX,class(X));
   nX(idx{:})=X;
   idxtau =idx;
   for taui=1:numel(wght);
      idx{1}=szX(1)*taui+(1:szX(1));
      for ei=taui+1:szX(dim(1));
         idx{dim(1)}=ei; idxtau{dim(1)}=ei-taui;
         nX(idx{:})=X(idxtau{:});
      end
   end
   X=nX;
   
elseif ( strcmp(type,'ar') || strcmp(type,'ar0') )
   if ( strcmp(type,'ar0') ) wght=[0;wght(:)]; end;
   arX = filter(wght(:),1,X,[],dim(1));
   % correct the startup effects
   for ei=1:numel(wght); 
      idx{dim(1)}=ei;  
      nf=sum(abs(wght(1:ei))); if(nf==0)nf=1;end; arX(idx{:})=arX(idx{:}).*sum(abs(wght))./nf;
   end;
   X   = cat(1,X,arX);

end
return


%--------------------------------------------------------------------------
function testCase()
oz=z;
z=jf_featembedding(oz,'dim','epoch','type','const');jf_disp(z)
z=jf_featembedding(oz,'dim','epoch','type',1);jf_disp(z)
z=jf_featembedding(oz,'dim','epoch','type','mean');jf_disp(z)
z=jf_featembedding(oz,'dim','epoch','type','ar','wght',ones(5,1));jf_disp(z)
z=jf_featembedding(oz,'dim','epoch','type','tau','wght',ones(3,1));

% with block structure
z=jf_featembedding(oz,'dim','epoch','type',1,'blockIdx','block')
clf;imagesc(shiftdim(z.X(end,1:50,:))) % check unique feature for each block

z=jf_featembedding(oz,'dim','epoch','type','mean','blockIdx','block');jf_disp(z)
clf;imagesc(squeeze(mean(abs(z.X(:,:,:)),2))); % check constant in blocks




