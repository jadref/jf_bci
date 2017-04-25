function [z]=utri(z,varargin);
% Extract the upper triangle from the selected pair of dims
opts=struct('dim',{{'ch','ch2'}});
% convert from names to dims
dim=opts.dim;
if ( iscell(dim) )
   dim = [strmatch(dim{1},{z.di.name},'exact') strmatch(dim{2},{z.di.name},'exact')];
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1;

sz=size(z.X); nd=ndims(z.X);
% Strip to only have the upper triangle
% Compute forward and reverse index expressions
nCh   =size(z.X,dim(1));
triIdx=zeros(1,(nCh*(nCh+1))/2,'int32'); Cnames2={};
revIdx=zeros(nCh,nCh,'int32');
k=1;
for d2=1:nCh;
   for d1=1:d2;
      triIdx(k) = d1 + (d2-1)*nCh;
      Cnames2{k}= [ z.di(dim(1)).vals{d1} '*' z.di(dim(2)).vals{d2} ];
      revIdx(d1,d2)=k; revIdx(d2,d1)=k;
      k=k+1;
   end
end
idx{dim(1)}=triIdx;

z.X=reshape(z.X,[sz(1:dim(1)-1) sz(dim(1))*sz(dim(2)) ...
                  sz(dim(2)+1:end)]);
% make an index expression
idx={}; for d=1:ndims(z.X); idx{d}=1:size(z.X,d); end;
idx{dim(1)}=triIdx; % only the upper tri parts for the ch dims
z.X=z.X(idx{:})*2; % inc correction for loosing 1/2 the entries
% rescale so this is numerically equivalent to 
%  z.X = utri(cov+cov'-diag(cov))
diagIdx=cumsum(1:nCh); idx{dim(1)}=diagIdx;
z.X(idx{:})=z.X(idx{:})/2; % don't double count the diag entries

odi=z.di;
z.di=z.di([1:dim(1) dim(2)+1:end]);
z.di(dim(1)).vals=Cnames2; 
z.di(dim(1)).name=sprintf('%sX%s_UT',odi(dim).name);
% Store fwd/rev indexs
z.di(dim(1)).info.fwdIdx=triIdx;
z.di(dim(1)).info.revIdx=revIdx;

summary=sprintf('over %s x %s',odi(dim).name);
info=struct('fwdIdx',triIdx,'revIdx',revIdx);
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function testCase()
