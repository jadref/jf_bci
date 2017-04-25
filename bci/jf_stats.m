function [z]=jf_stats(z,varargin);
% compute mean and variance over a given dimension
%
% Options:
%  dim    -- the dimension(s) along which to compute the statistics  ('epoch')
%  minStd -- min allowed std                      (1e-5)
opts=struct('dim','epoch','minStd',1e-5,'summary',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim);
sz  = size(z.X); mus=[];stds=[];
if ( isreal(z.X) ) mus = msum(z.X,dim)./prod(sz(dim));
else               mus = msum(abs(z.X),dim)./prod(sz(dim));
end
mus(isnan(mus))=0;
idx = 1:numel(sz); idx(dim)=-idx(dim);
if ( isreal(z.X) ) stds= tprod(z.X,idx,[],idx)./prod(sz(dim));
else stds=(tprod(real(z.X),idx,[],idx)+tprod(imag(z.X),idx,[],idx))./prod(sz(dim));
end
stds= stds - mus.^2; % remove any mean effect
stds= sqrt(abs(stds)); stds(stds==0 | isnan(stds))=1; stds=max(stds,opts.minStd);

z.X = cat(dim(1),mus,stds);
odi = z.di(dim);
z.di(dim(2:end))=[];
z.di(dim(1))=mkDimInfo(2,1,'statistic',[],{'mean','std'});
z.di(dim(1)).info.N=prod(sz(dim));
info.odi= odi;
info.N  = prod(sz(dim));
if ( isempty(dim) ) 
   summary=['over everything']; 
else
   summary = ['over ' sprintf('%s,',odi(1:end-1).name) sprintf('%s',odi(end).name)];
end
if ( ~isempty(opts.summary) ) summary=[summary ' (' opts.summary ')']; end;
z   = jf_addprep(z,mfilename,summary,opts,info);
return;
