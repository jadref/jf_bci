function [z,opts]=jf_var(z,varargin)
% computed the weighted mean and remove a particular dimension
% Options:
%  dim -- dimension(s) to squared and summed ('time')
%  ave -- [bool] average of the power?
%  log -- [bool] log of the power
%  summary -- additional summary string
opts=struct('dim','time','log',0,'ave',1,'summary','','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
dim = n2d(z.di,opts.dim);

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
jf_disp(jf_var(z,'dim','time'))
jf_disp(jf_var(z,'dim',{'ch' 'time'}))