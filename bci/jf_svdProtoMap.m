function [z]=jf_svdProtoMap(z,varargin);
% transform the eigenvalue structure of a set of covariance matrices, e.g. by log
%
% See Also: svdProtoMap
opts=struct('subIdx',[],'verb',0,'dim',[],'labdim',[],'rank',[],'clsfr',1);
[opts,varargin]=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);
% use Ydi to set the dim if not already set
if ( isempty(dim) && ~isempty(z.Ydi) ) 
   dim = n2d(z.di,{z.Ydi(1:end-1).name},1,0); dim(dim==0)=[]; % get dim to work along
   if ( numel(dim)>1 && all(diff(dim)~=1) ) error('only for conseq dims'); end;
end

% call the function to do the actual work
[U,V,z.X,featDims]=svdProtoMap(z.X,z.Y,'dim',dim,'labdim',opts.labdim,'rank',opts.rank,'clsfr',opts.clsfr,varargin{:});

% update the meta-info
z.di(featDims)   = mkDimInfo(ones(numel(featDims),1),1);
z.di(featDims(1))= mkDimInfo(size(z.X,featDims(1)),1,'comp_1');
z.di(featDims(2))= mkDimInfo(size(z.X,featDims(2)),1,'comp_2');
summary=sprintf('%d components',opts.rank);
if ( opts.clsfr ) summary=[summary " (clsfr)"]; end;
z=jf_addprep(z,mfilename,summary,opts,struct('U',U,'V',V));
return;
%---------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
zc=jf_cov(z);
zr=jf_svdProtoMap(zc,'rank',2);
zr=jf_svdProtoMap(zc,'rank',2,'clsfr',0); % pure-pre-processing non-class specific varient
