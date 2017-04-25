function [z,opts]=jf_sfvar(z,varargin)
% computed spatiall filtered variance
% Options:
%  dim -- dimension(s) to squared and summed ('time')
%  ave -- [bool] average of the power?
%  log -- [bool] log of the power
%  di  -- [struct ndim x 1] dimension Info struct describing the linear mapping matrix
%  mx  -- [size(z.X,di.name) x nComp] matrix mapping this dimension
%  summary -- additional summary string
opts=struct('dim','time','mx',[],'di',[],'log',0,'ave',1,'summary','','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
dim = n2d(z.di,opts.dim);

% Get the dimension to apply the linear mapping to
di=opts.di;
sfdim = n2d(z.di,{di(1:end-1).name},0,0); sfdim(sfdim==0)=[];
% apply the linear mapping
mx=opts.mx;
xIdx=1:(max(ndims(z.X),max(sfdim))); xIdx(sfdim)=-xIdx(sfdim);
z.X = tprod(z.X,xIdx,mx,[-sfdim(:)' sfdim(1)]);
z.di(sfdim(1))=di(min(numel(sfdim)+1,end)); % update the meta-info

% compute the power along this dimension
szX=size(z.X);
idx = 1:ndims(z.X); idx(dim)=-dim;
z.X = tprod(z.X,idx,[],idx);
if ( opts.ave ) z.X=z.X./prod(szX(dim)); end;
if ( opts.log ) z.X=log(abs(z.X)); end;

% compress out the removed dimension
szNu=size(z.X); 
z.X=reshape(z.X,[szNu(setdiff(1:end,dim)) 1]);
% update the meta-info
odi=z.di;
z.di(dim)=[];
% update the labelling if needed
if ( isfield(z,'Ydi') && any(n2d(z.Ydi,{odi(dim).name},0,0)) )
  szY=size(z.Y);
  ydim=n2d(z.Ydi,{odi(dim).name});
  idx={};for d=1:ndims(z.Y); idx{d}=1:szY(d); end; idx{ydim}=1;
  ymatch=repop(z.Y,'~=',z.Y(idx{:}));
  if ( any(ymatch(:)) )
    warning(sprintf('Labels arent identical over: %s, 1st label used',z.Ydi(ydim(1)).name));
  end
  % re-label
  z.Y = reshape(z.Y(idx{:}),[szY(setdiff(1:numel(szY),ydim)) 1]);
  z.Ydi= z.Ydi(setdiff(1:numel(z.Ydi),ydim));
  if ( isfield(z,'foldIdxs') ) % update the fold info also
    szfoldIdxs=size(z.foldIdxs);
    idx={};for d=1:numel(szfoldIdxs); idx{d}=1:szfoldIdxs(d); end; idx{ydim}=1;
    z.foldIdxs=reshape(z.foldIdxs(idx{:}),[szfoldIdxs(setdiff(1:numel(szfoldIdxs),ydim)) 1]);
  end
end
summary = sprintf('over %s',odi(dim(1)).name);
if( numel(dim)>1 ) summary = [summary sprintf('+%s',odi(dim(2:end)).name)]; end;
if(~isempty(opts.summary)) summary=[summary  sprintf(' (%s)',opts.summary)];end
info=[];
z=jf_addprep(z,mfilename,summary,opts,info);
return;
%------------------------------------------------------------------
function testCase()
z=jf_import('jf_var','tst','1',randn(10,11,12),{'ch' 'time' 'epoch'});
jf_disp(jf_sfvar(z,'dim','time','mx',randn(10,2),'di',mkDimInfo([10 2],{'ch','sf'})))
jf_disp(jf_sfvar(z,'dim',{'ch' 'time'}))