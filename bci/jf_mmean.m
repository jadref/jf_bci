function [z]=mmean(z,varargin);
% average away a dimension
% Options:
%  dim -- dimension to average over
opts=struct('dim',[],'subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

dim=n2d(z,opts.dim); 

% do the work
sz  = size(z.X); rmDims=setdiff(1:numel(sz),dim);
z.X = reshape(mmean(z.X,dim),sz(rmDims));

odi=z.di;
z.di = z.di([rmDims end]);
summary = sprintf('over %s',sprintf('%ss ',odi(dim).name));
info = struct('odi',odi(dim));
z = jf_addprep(z,mfilename,summary,opts,info);
return
%-------------------------------------------------------------------------
function testCase()
