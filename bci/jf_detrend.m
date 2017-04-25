function [z]=jf_detrend(z,varargin)
% remove linear/quadratic trends from the input
%
% Options:
%  dim    -- dim to detrend along  ('time')
%  wght   -- weighting over detrend dim to use ([])
%  order  -- type of detrending, 1-linear, 2-quadratic (N.B. *BUGGY*) (1)
%  subIdx -- [ndims(X) x 1 cell] array of dimension indices used to subset the set of dims to detrend
%            each entry can be : range indicies to include, []-use all values, {{bgn} {end}} stat end idxs

opts=struct('dim','time','wght',[],'order',1,'subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% get the info about the dim to uses
sz=size(z.X); nd=ndims(z.X);
dim=n2d(z,opts.dim);

% call detrend to do the actual work
if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
  idx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
  z.X(idx{:})= detrend(z.X(idx{:}),dim,opts.order,opts.wght);
else
  z.X = detrend(z.X,dim,opts.order,opts.wght);
end

summary = ['over ',sprintf('%s ',z.di(dim).name)];
if ( opts.order>1 ) summary=sprintf('%s (%dth order)',summary,opts.order); end;
z=jf_addprep(z,mfilename,summary,opts,[]);
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();

z=jf_detrend(z);
z=jf_detrend(z,'dim','time','subIdx',{5:10})