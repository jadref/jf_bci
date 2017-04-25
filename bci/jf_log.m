function [z]=jf_log(z,varargin);
% compute the logorithm of the inputs
%
opts=struct('db',false,'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);
z.X=log(z.X);
z.di(end).units=['log(' z.di(end).units ')'];
if ( opts.db ) z.X=z.X*10; z.di(end).units='db'; end; % convert to db's
z =jf_addprep(z,mfilename,'',[],[]);
return;

