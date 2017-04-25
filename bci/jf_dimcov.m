function [z]=jf_dimcov(z,varargin);
% compute stack of covariance matrices for each dimension
%
%  [z]=jf_dimcov(z,varargin);
%
% Options:
%  dim -- the set of dimensions to compute covariances over
%  stepDim -- [int] dimensions to step along, i.e. re-compute for each element of this dim(s) ([])
%  type-- type of covariance to compute, one-of  ('real')
%          'real' - pure real, 'complex' - complex, 'imag2cart' - complex converted to real
opts=struct('dim',{'ch'},'stepDim',{'epoch'},'type','real','subIdx',[],'verb',0,'whtFeat',[]);
opts=parseOpts(opts,varargin);

% convert from names to dims
dim=n2d(z.di,opts.dim,[],0);dim(dim==0)=[];
stepDim=n2d(z,opts.stepDim,[],0); stepDim(stepDim==0)=[];
accdim=setdiff(1:ndims(z.X),[dim(:);stepDim(:)]);accdim(accdim==0)=[]; %n2d(z.di,opts.accdim,[],0); 

sz=size(z.X); sz(end+1:max(dim))=1; nd=ndims(z.X);

% work out the max size cov-mx we will need
maxd = max(sz(dim));

% compute the dimension indices for the products
accind = false(1,numel(sz)); accind(dim)=true; accind(accdim)=true;% dimensions which will be removed
shifts=zeros(1,numel(sz));
shifts(accind)   =-1; % squeeze out accum
shifts(min(dim)) = 2; % insert for OP and dim-stack
odim = (1:ndims(z.X)) + cumsum(shifts); % where does each input dim go-to in the output, i.e. old->new
ocovdim= odim(min(dim))-2+(0:2); % where the cov go in the output [d1 x d2 x numel(dim)]

% template product index for the tensor products
tpidx = odim; 
tpidx(dim)=-dim; tpidx(accdim)=-accdim; 

% compute the size of the output and index expressions for inserting into this matrix
ndim = zeros(1,max(odim)); ndim(odim(~accind))=find(~accind); % where new dim came from, i.e. new->old
nsz  = zeros(size(ndim)); nsz(ndim>0)=sz(ndim(ndim>0));
nsz(ocovdim) = [maxd maxd numel(dim)];
% template index expression for storing the result
nidx={}; for di=1:numel(nsz); nidx{di}=1:nsz(di); end;

% matrix variate covariance
if ( strcmp(opts.type,'matcov') )
  [Cs,Ws]=matcov(z.X,dim,stepDim,[],[],opts.whtFeat);				  

else % raw covariance for each dimension
  for di=1:numel(dim); % loop over dims computing the cov in turn
                       % setup the OP over the target dimensions
    idx1=tpidx;                     idx2=tpidx;
    idx1(dim(di))=ocovdim(1);       idx2(dim(di))=ocovdim(2);

		  % Map to co-variances, i.e. outer-product over the channel dimension
    if ( isreal(z.X) )
      Xdi = tprod(z.X,idx1,[],idx2);
    else
      switch (opts.type);
        case 'real';       Xdi = tprod(real(z.X),idx1,[],idx2) + tprod(imag(z.X),idx1,[],idx2);% pure real output
        case 'complex';    Xdi = tprod(z.X,idx1,conj(z.X),idx2); % pure complex, N.B. need complex conjugate!
        case 'imag2cart';  error('Not supported for this method');
        otherwise; error('Unrecognised type of covariance to compute');
      end
    end
    if ( numel(dim)>1 ) Xdi=Xdi/prod(sz(dim([1:di-1 di+1:end]))); end
	 Cs{di}=Xdi;
  end
end

nX = zeros(nsz,class(z.X));
for di=1:numel(dim);
										  % insert result into the output array
   nidx{ocovdim(1)}=1:sz(dim(di));
   nidx{ocovdim(2)}=1:sz(dim(di));
   nidx{ocovdim(3)}=di;
   nX(nidx{:})=Cs{di}; % do the insertion
end

% update the object
z.X=nX;
odi=z.di;
ndim(ocovdim)=min(dim);
z.di=z.di([ndim end]);
z.di(ocovdim(1)) = mkDimInfo(maxd,1,'feat');
z.di(ocovdim(2)) = mkDimInfo(maxd,1,'feat_2');
z.di(ocovdim(3)) = mkDimInfo(numel(dim),1,'dimcov',[],{odi(dim).name});
if ( ~isempty(z.di(end).units) ) z.di(end).units=[z.di(end).units '^2'];end

summary=sprintf('over %ss',odi(dim(1)).name);
if( ~strcmp(opts.type,'real') ) summary = [opts.type ' ' summary]; end;
if(numel(dim)>1) summary=[summary ' x (' sprintf('%s ',odi(dim(2:end)).name) ')'];end 
info=struct('sz',sz,'accdi',odi(dim(2:end)));
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
z=jf_spectrogram(z,'width_ms',250,'feat','complex');
oz=z;
d=jf_dimcov(z,'dim',[1 2]);

