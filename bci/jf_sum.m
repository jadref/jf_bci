function [z]=jf_sum(z,varargin);
% sum-out the given dimension(s)
z=jf_mean(z,'wght',1,varargin{:});
return