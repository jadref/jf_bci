function [z,opts]=jf_mean(z,varargin)
% computed the weighted mean and remove a particular dimension
% Options:
%  dim -- dimension(s) to be meaned ('epoch')
%  wght-- weighting for the included elements, average over idx if empty. ([])
%         OR 'robust' to use a robust median estimate
%  summary -- additional summary string
opts=struct('dim','epoch','wght',[],'summary','','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
if( ~isempty(opts.wght) && isnumeric(opts.wght) && (numel(opts.wght)>1 && numel(opts.wght)~=numel(opts.dim)) ) 
   error('Wght should match num ref pos');
end;
wght = opts.wght;
dim = n2d(z.di,opts.dim);

% compute the average
if ( isempty(wght) ) % simple average
   mu=z.X; N=1;
   for di=1:numel(dim); N=N*size(mu,dim(di)); mu=sum(mu,dim(di));  end; 
   mu=mu./N;
elseif ( isnumeric(wght) && numel(wght)==1 ) % sum
   mu=z.X; for di=1:numel(dim); mu=sum(mu,dim(di));  end; mu=mu*wght;
elseif ( isnumeric(wght) && numel(wght)>1 )  % weighted average
   wIdx=[1:ndims(z.X)]; wIdx(dim)=-wIdx(dim); mu=tprod(z.X,wIdx,wght,-dim)
elseif ( strcmp(wght,'robust') ) % robust average
   mu= median(z.X,dim); % do a robust centering
end
szMu=size(mu); 
z.X=reshape(mu,[szMu(setdiff(1:end,dim)) 1]);
% update the meta-info
odi=z.di;
z.di(dim)=[];
% update the labelling if needed
if ( isfield(z,'Ydi') && any(n2d(z.Ydi,{odi(dim).name},0,0)) )
  szY=size(z.Y);
  ydim=n2d(z.Ydi,{odi(dim).name},1,0); ydim(ydim==0)=[];
  idx={};for d=1:ndims(z.Y); idx{d}=1:szY(d); end; [idx{ydim}]=deal(1);
  ymatch=repop(z.Y,'~=',z.Y(idx{:}));
  if ( any(ymatch(:)) )
    warning(sprintf('Labels arent identical over: %s, 1st label used',z.Ydi(ydim(1)).name));
  end
  if ( numel(z.Y(idx{:}))==1 ) % remove Y
	  z=rmfield(z,{'Y','Ydi'});
	  if (isfield(z,'foldIdxs')) 	  z=rmfield(z,'foldIdxs'); end;
  else
	 % re-label
	 z.Y = reshape(z.Y(idx{:}),[szY(setdiff(1:numel(szY),ydim)) 1]);
	 z.Ydi= z.Ydi(setdiff(1:numel(z.Ydi),ydim));
	 if ( isfield(z,'foldIdxs') ) % update the fold info also
		szfoldIdxs=size(z.foldIdxs);
		idx={};for d=1:numel(szfoldIdxs); idx{d}=1:szfoldIdxs(d); end; idx{ydim}=1;
		z.foldIdxs=reshape(z.foldIdxs(idx{:}),[szfoldIdxs(setdiff(1:numel(szfoldIdxs),ydim)) 1]);
	 end
  end
end
summary = sprintf('over %s',odi(dim(1)).name);
if( numel(dim)>1 ) summary = [summary sprintf('+%s',odi(dim(2:end)).name)]; end;
if(ischar(opts.wght)) summary = [summary ' ' opts.wght]; end;
if(~isempty(opts.summary)) summary=[summary  sprintf(' (%s)',opts.summary)];end
info=[];
z=jf_addprep(z,mfilename,summary,opts,info);
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_import('jf_mean','tst','1',randn(10,11,12),{'ch' 'time' 'epoch'},'Y',sign(randn(12,1)));
jf_disp(jf_mean(z,'dim','ch'))
jf_disp(jf_mean(z,'dim',{'ch' 'time'}))
jf_disp(jf_mean(z,'dim',{'time' 'epoch'}))

z=jf_import('jf_mean','tst','1',randn(10,11,12),{'ch' 'time' 'epoch'},'Y',sign(randn(11,12)),'Ydi',{'time' 'epoch'});
jf_mean(z,'dim',{'time' 'epoch'})
