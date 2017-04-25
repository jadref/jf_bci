function [z]=jf_cov2d(z,varargin);
% comute 2d-covariance matrices, i.e. [X'*X X;X' X*X']
%
%  [z]=jf_2dcov(z,varargin);
%
% Options:
%  dim -- spec of the dimensions to compute covariances matrices
%         dim(1)=dimension to compute covariance over
%         dim(2:end) dimensions to sum along
%  type-- type of covariance to compute, one-of  ('real')
%          'real' - pure real, 'complex' - complex, 'imag2cart' - complex converted to real
opts=struct('dim',{{'ch' 'time'}},'type','real','subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

% convert from names to dims
dim=n2d(z.di,opts.dim,[],0);dim(dim==0)=[];

if ( numel(dim)~=2 ) error('Only 2d supported so far'); end;
if ( ~all(diff(dim)==1) ) error('Only for consequetive dimensions at the moment!'); end;


z.X=cov2d(z.X,dim);

odi=z.di;
z.di=z.di([newDs end]);
z.di(covd).name = [odi(dim(1)).name '_' odi(dim(2)).name];
z.di(covd+1).name=[z.di(covd).name '_2'];

% set the new element names
z.di(covd).vals  =cell(size(X,covd),1);
nvi=0;
for di=1:numel(dim);
  valdi=odi(dim(di)).vals;
  for vi=1:numel(valdi);
	 if ( iscell(valdi) )
		val=valdi{vi};
	 elseif ( isnumeric(valdi) )
		val=sprintf('%g%s',valdi{vi},odi(dim(di)).units);
	 end;
	 nvi=nvi+1;
	 z.di(covd).vals{nvi}=val;
  end
end

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
  z=jf_fft(z);
  z=jf_retain(z,'dim','freq','range','between','vals',[8 30]);
