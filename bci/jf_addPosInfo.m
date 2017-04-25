function [z]=jf_addPosInfo(z,varargin);
% add electrode position information to the data
opts=struct('dim','ch','capFile','1010','prefixMatch',[],'verb',[],'overridechnms',[]);
opts=parseOpts(opts,varargin);
[z.di(n2d(z,opts.dim))]=addPosInfo(z.di(n2d(z,opts.dim)),opts.capFile,opts.overridechnms,opts.prefixMatch,opts.verb);
z=jf_addprep(z,mfilename,[],opts,[]);
return;
