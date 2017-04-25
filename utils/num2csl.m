function varargout=num2csl(A,dims);
% Convert matrix to comma-separated list,
%
% [A(..,1,..) A(..,2,..) A(..,3,..).. A(..,size(A,dim),..)]=num2cls(A,dim);
%
% Inputs:
%  A   -- matrix to split up
%  dim -- dimensions to keep, in the same format as num2cell
% Outputs:
%  A(..,i,..) -- the i'th entry of A along the requested dimension
if ( nargin < 2 || isempty(dims) ) dims=find(size(A)==1,1); if(isempty(dims)) dims=1;end; end;
varargout = num2cell(A,dims);
varargout = varargout(:);
return;
%---------------------------------------------------------------------
function testCases()
