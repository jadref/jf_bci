function [z]=jf_linMapDim(z,varargin);
% code to linearly map a given dimension in z to a new space
% Options:
%  dim -- the dimension(s) of X to re-map
%  mx  -- [size(X,dim) x nFeat] the matrix linearly transforming to the
%         new space.
%  di  -- [3x1 dimInfo struct] dimInfo structure for the mapping matrix
%         OR
%         [1x1 dimInfo struct] for the new features description
%  sparse -- [bool] do we use a the sparse mapping code
%  sparseTol -- [double] max deviation from 0 to be treated as 0
%  minSparse -- \in [0:1], min faction of 0's to use the sparse code
opts=struct('dim',[],'mx',[],'di',[],'sparse',1,'sparseTol',0,'minSparse',.33,'summary',[],'subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

dim=[]; % get the dim to work along
if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1; % convert neg dim specs
if ( iscell(opts.di) ) opts.di=mkDimInfo(size(opts.mx),opts.di); end;
if ( isempty(dim) && ~isempty(opts.di) && numel(opts.di)>1 )
   dim = n2d(z.di,{opts.di(1:end-1).name},0,0); dim(dim==0)=[];
end

mx=opts.mx;
if(~opts.sparse || numel(dim)>1 || ...
   sum(abs(mx(:))<=opts.sparseTol)/numel(mx)<opts.minSparse )
   if ( issparse(mx) ) mx=full(mx); end;
   xIdx=1:(max(ndims(z.X),max(dim))); xIdx(dim)=-xIdx(dim);
   z.X = tprod(z.X,xIdx,mx,[-dim(:)' dim(1)],'n');

else % attempt to use the sparsity of the input map to speed up the
     % computation
   sz=size(z.X); 
   X=zeros([sz(1:dim(1)-1) size(mx,2) sz(dim(1)+1:end)],class(z.X));
   idx={};for di=1:ndims(z.X); idx{di}=1:size(z.X,di); end; % index into result
   spIdx=idx; % sparse index into z.X
   for di=1:size(mx,2);
      if ( issparse(mx) ) % N.B.tprod can't use sparse at the moment!
         spIdx{dim} = find(mx(:,di)); 
      else % input is full so find the sparse elements we want
         spIdx{dim} = abs(mx(:,di))>opts.sparseTol;% non-sparse vals in input
      end
      idx{dim}   = di; % destination value
      % N.B. this indexing operation is potentially hugely expensive as it makes
      %      a copy of z.X!
      X(idx{:})   = tprod(z.X(spIdx{:}),[1:dim(1)-1 -dim(:)' dim(1)+1:ndims(z.X)],...
                          mx(spIdx{dim},di),[-dim(:)' dim(:)'],'n');
   end   
   z.X = X; % put in the new value
end

% Update the objects dim-info
odi=z.di;
di=opts.di;
if ( ~isempty(di) )
   if ( ischar(di) ) di={di}; end;
   if ( iscell(di) ) 
      if ( numel(di)==1 ) di={z.di(dim).name di{:}}; end;
      di=mkDimInfo(size(mx),di);
   end;
   if ( numel(di)~=numel(dim)+2 && numel(di)~=1 && numel(di)~=2 ) 
      error('di size is unreasonable, should be 1,2 or 3');
   end
   if ( numel(di)==1 )
      z.di(dim(1))=di;
   else
      z.di(dim(1)) = di(min(numel(dim)+1,end)); % use the info in the given dimInfo
   end
else
   z.di(dim(1)) = mkDimInfo(size(z.X,dim(1)),1,[z.di(dim(1)).name '_linMap'],[],[]);
end
% squeeze out analiated dims
if ( numel(dim)>1 ) 
   szX=size(z.X); szX(end+1:max(dim))=1; szX(dim(2:end))=[]; z.X=reshape(z.X,szX);
   z.di(dim(2:end))=[];
end

summary=sprintf('%s->%d %ss',dispDimInfo(odi([dim(:)' end])),numel(z.di(dim(1)).vals),z.di(dim(1)).name);
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
info=struct('odi',odi(dim));
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
fs=128;
mix=cat(3,[1 2;.5 2],[.5 2;1 2],[0 0;0 0]); % power switch with label + noise
Y=ceil(rand(N,1)*L); oY         = lab2ind(Y);   % True labels
z=jf_mksfToy(Y,'y2mix',mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'period',[fs/16;fs/16],'phaseStd',pi/2);

nFeat = 2;
mx = randn(size(z.X,1),2); % raw
zl=jf_linMapDim(z,'dim','ch','mx',mx);

% inc di
mxDi = mkDimInfo(size(mx),'ch',[],[],'nfeat',[],[])
zl=jf_linMapDim(z,'mx',mx,'di',mxDi);

% with sparsity
smx=mx; smx(randn(size(mx))<0)=0; % 50% sparse
zl=jf_linMapDim(z,'mx',smx,'di',mxDi,'sparse',0);
szl=jf_linMapDim(z,'mx',smx,'di',mxDi);
