function [z]=jf_abs(z,varargin);
% get the absolute value of the inputs
%
opts=struct('subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);
z.X=abs(z.X);
z =jf_addprep(z,mfilename,'',opts,[]);
return;
%---------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
f=jf_abs(z);
jf_disp(f)
jf_plotERP(jf_abs(f));
