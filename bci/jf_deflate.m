function [z]=jf_deflate(z,varargin);
% remove any signal correlated with the input signals from the data
%
% Options:
%  dim -- the dimension to re-map
%  mx  -- [size(X,dim) x nFeat] the set of directions to deflate away
%  di  -- [3x1 dimInfo struct] dimInfo structure for set of deflation directions
opts=struct('dim',[],'mx',[],'di',[],'sparse',1,'sparseTol',0,'minSparse',.33,'subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

dim=[]; % get the dim to work along
if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1; % convert neg dim specs
if ( isempty(dim) && ~isempty(opts.di) && numel(opts.di)>1 )    
   dim = strmatch(opts.di(1).name,{z.di.name}); % extract dim from the di...
end
dim=sort(dim,'ascend'); % ensure are in X's order

% use the input di to ensure dim-order is the same
p = opts.mx; 
szP = size(p);
if ( ~isempty(opts.di) ) 
   [p2z,z2p]=matchstrs({opts.di(1:end-1).name},{z.di(dim).name,'def_dir'}); % get mapping from p->z
   if ( any(szP(z2p==0))>1 ) error('non-matched dims of p must be singlenton'); end;
   p = reshape(p,szP(z2p~=0)); p2z(p2z==0)=[]; % remove ignored singlentons
   p = permute(p,p2z);           % finally reorder dims to match those of z.X
elseif ( ndims(p) > numel(dim)+1 ) 
   error('p doesnt match dim');
end

z.X = deflate(z.X,p,dim);
z=jf_addprep(z,mfilename,['over ',sprintf('%s ',z.di(dim).name)],opts,[]);
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
d=jf_deflate(z,'dim','time','mx',shiftdim(z.X(2,:,1)));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','mcplot','colorbar','ne');clickplots

d=jf_deflate(z,'dim',{'time','epoch'},'mx',shiftdim(z.X(2,:,:)));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','mcplot','colorbar','ne');clickplots

d=jf_deflate(z,'dim',{'time','epoch'},'mx',permute(z.X(1:2,:,:),[2 3 1]));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','mcplot','colorbar','ne');clickplots

di=z.di; di(1).name='def_dir';
d=jf_deflate(z,'dim',{'time','epoch'},'mx',z.X([1:2],:,:),'di',di);
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','mcplot','colorbar','ne');clickplots
