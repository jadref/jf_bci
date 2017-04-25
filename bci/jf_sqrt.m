function [z]=jf_sqrt(z,varargin);
% compute the square-root of the inputs for each feature value
%
opts=struct('subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);
nF  = sqrt(abs(z.X)); nF(nF==0)=1; % guard for divide by 0
z.X = z.X./nF;
z.di(end).units=['sqrt(' z.di(end).units ')'];
z   = jf_addprep(z,mfilename,'',[],[]);
return;

