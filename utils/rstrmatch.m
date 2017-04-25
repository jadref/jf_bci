function [match]=rstrmatch(str,strs,varargin)
% Recursive str match function, -- which looks inside cell arrays of
% cell-arrays of strings
if ( ischar(strs) ) match=strmatch(str,strs,varargin{:}); return; end;
if ( iscell(strs) )
   for i=1:numel(strs)
      match(i)=~isempty(strmatch(str,strs{i},varargin{:}));
   end
   match = find(match);
end