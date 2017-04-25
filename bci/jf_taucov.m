function [z]=jf_taucov(z,varargin);
% compute covariance matrices -- N.B. *NOT* centered
%
%  [z]=jf_taucov(z,varargin);
%
% Options:
%  dim -- spec of the dimensions to compute covariances matrices
%         dim(1)=dimension to compute covariance over
%         dim(2)=dimension to sum along + compute delayed covariance, i.e. time
%         dim(3:end)=dimension to sum out
%  type-- type of covariance to compute, one-of  ('real')
%          'real' - pure real, 'complex' - complex, 'imag2cart' - complex converted to real
%  taus_samp/taus_ms-- [nTau x 1] set of sample offsets to compute cross covariance for
%         OR
%         [1x1] and <0, then use all values from 0:-taus
%  shape-- shape of output covariance to compute   ('3d')
%         one-of; '3d' - [d x d x nTau], '2d' - [d*nTau x d*nTau]
opts=struct('dim',{{'ch' 'time'}},'type','real','taus_samp',[],'taus_ms',[],'shape','3d','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% convert from names to dims
dim=n2d(z.di,opts.dim); 
nd=max(ndims(z.X),numel(z.di)-1);
sz=size(z.X);sz(end+1:nd)=1; 

taus=opts.taus_samp;
if ( isempty(taus) && ~isempty(opts.taus_ms) )
  fs=getSampRate(z);
  taus = round(opts.taus_ms / 1000 * fs);
  if ( numel(unique(taus))<numel(taus) )
    warning('some duplicated taps removed');
    taus=unique(taus);
  end
end
if ( isempty(taus) ) taus=0; end;
if ( numel(taus)==1 && isnumeric(taus) && taus<0 ) taus=0:-taus; end;


% call taucov to do the real work...
cov=taucov(z.X,dim,taus,'type',opts.type,'shape',opts.shape,varargin{:});

z.X = cov; 
clear cov;
if ( strcmpi(opts.shape,'2d') )
  newDs=[setdiff(1:dim(1),dim(2:end)) dim(1) setdiff(dim(1)+1:nd,dim(2:end))];
else
  newDs=[setdiff(1:dim(1),dim(2:end)) dim(1) dim(1) setdiff(dim(1)+1:nd,dim(2:end))];
end
odi=z.di;
z.di=z.di([newDs end]);
nchD=find(newDs==dim(1),1,'first');
z.di(nchD+1)=z.di(nchD);
z.di(nchD+1).name=[odi(dim(1)).name '_2'];
if ( strcmp(opts.type,'imag2cart') && ~isreal(z.X) )   
  z.di(nchD).vals = repmat(z.di(nchD).vals,[1 2]);
  z.di(nchD+1).vals=repmat(z.di(nchD+1).vals,[1 2]);
end
if ( strcmp(opts.shape,'2d') )
  z.di(nchD).vals = repmat(z.di(nchD).vals,[1 numel(taus)]);
  z.di(nchD+1).vals=repmat(z.di(nchD+1).vals,[1 numel(taus)]);
else
  if ( numel(opts.taus_ms)~=numel(taus) )
    z.di(nchD+2)=mkDimInfo(numel(taus),1,'tau','samp',taus);
  else        
    z.di(nchD+2)=mkDimInfo(numel(taus),1,'tau','ms',opts.taus_ms);
  end
end
if ( ~isempty(z.di(end).units) ) z.di(end).units=[z.di(end).units '^2'];end

summary=sprintf('over %ss',odi(dim(1)).name);
if( ~strcmp(opts.type,'real') ) summary = [opts.type ' ' summary]; end;
if(numel(dim)>1) summary=[summary ' x (' sprintf('%s ',odi(dim(2:end)).name) ')'];end 
summary=[summary sprintf(' %d taus',numel(taus))];
info=struct('sz',sz,'accdi',odi(dim(2:end)));
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
t=jf_taucov(z,'taus_samp',[0:3]);
t=jf_taucov(z,'taus_samp',[0:3],'shape','2d');
t=jf_taucov(z,'taus_ms',[0 10 20]);

t=jf_taucov(z,'taus_samp',[0 7 10 15]); % optimised tap to signal..

clf;jplot([z.di(1).extra.pos2d],shiftdim(mdiag(sum(t.X,4),[1 2])))
