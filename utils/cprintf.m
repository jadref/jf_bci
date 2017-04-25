function [out]=cprintf(format,v1,varargin)
% N.B. we assume v1 has number of rows number output.
for i=1:size(v1,1)
  out{i}=sprintf(format,v1,varargin{:});
end
