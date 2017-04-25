function [z]=jf_standardize(z,varargin);
% normalise the distribution of features, i.e. 0-mean 1-std dev
%
% Options:
%  dim    -- the dimension(s) which contain the features to standardize
%            i.e. for each of these dims we std the variance over the remaining
%            dims ('ch')
%  center -- [1x1 bool] flag do we center and std (1)
%  minStd -- min allowed std                      (1e-5)
%  exIdx  -- {cell} cell array of arguments to pass to jf_retain to specify the sub-set of
%                   X which *will not* be standardized
%
opts=struct('dim','ch','center',1,'minStd',1e-5,'exIdx',[],'summary',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim);
exIdx={};
if ( ~isempty(opts.exIdx) ) % set subset to apply leave alone
  exIdx = subsrefDimInfo(z.di(dim),opts.exIdx{:}); % set of indexs to keep
end

sz  = size(z.X); mus=[];stds=[];
aveDims = setdiff(1:numel(sz),dim);
if ( isreal(z.X) ) mus = msum(z.X,aveDims)./prod(sz(aveDims));
else               mus = msum(abs(z.X),aveDims)./prod(sz(aveDims));
end
mus(isnan(mus))=0;
if ( ~isempty(exIdx) ) mus(exIdx{:})=0; end;
if ( opts.center )
   z.X = repop(z.X,'-',mus);    
end
idx = 1:numel(sz); idx(aveDims)=-idx(aveDims);
if ( isreal(z.X) ) stds= tprod(z.X,idx,[],idx)./prod(sz(aveDims));
else stds=(tprod(real(z.X),idx,[],idx)+tprod(imag(z.X),idx,[],idx))./prod(sz(aveDims));
end
if ( ~opts.center ) stds= stds - mus.^2; end % remove any mean effect
stds= sqrt(abs(stds)); stds(stds==0 | isnan(stds))=1; stds=max(stds,opts.minStd);
if ( ~isempty(exIdx) ) stds(exIdx{:})=1; end;
z.X = repop(z.X,'./',stds);
if ( isempty(dim) ) 
   summary=['over everything']; 
else
   summary = ['per ' sprintf('%s,',z.di(dim(1:end-1)).name) sprintf('%s',z.di(dim(end)).name)];
end
if ( ~isempty(opts.summary) ) summary=[summary ' (' opts.summary ')']; end;
info=struct('stds',stds,'mus',mus);
if ( opts.center )
   info.testfn={'jf_repop' 'op' {'-' './'} 'mx' {mus stds}};
else
   info.testfn={'jf_repop' 'op' './' 'mx' stds};
end
z   = jf_addprep(z,mfilename,summary,opts,info);
return;
%----------------------------------------------------------------------
function testCase()

zn = jf_normalize(z)
