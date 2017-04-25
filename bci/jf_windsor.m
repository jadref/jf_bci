function [z]=jf_windsor(z,varargin);
% windsor-ize the data to suppress artifacts, i.e. replace outliers with max/min values
%
opts=struct('subIdx',[],'verb',0,'dim','time','thresh',2.75,'mode','huber','lentype','mad');
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim);
z.X=windsor(z.X,dim,opts.mode,opts.thresh,opts.lentype,opts.verb);
z =jf_addprep(z,mfilename,sprintf('over %s',z.di(dim).name),[],[]);
return;
%-----------------------------------------
function testcase()
  z=jf_windsor(z)
  z=jf_windsor(z,'mode','sqrt')
