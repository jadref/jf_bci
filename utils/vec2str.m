function [str]=vec2str(vec,chr); % pretty print vector of ints
if ( nargin < 2 || isempty(chr) ) chr=' '; end;
str='';
if( isnumeric(vec) )
   str=sprintf('%g',vec(1));
   if ( numel(vec)>1 ) str = [str sprintf([chr '%g'],vec(2:end))]; end
elseif ( iscell(vec) && ischar(vec{1}) )
   str=sprintf('%s',vec{1});
   if ( numel(vec)>1 ) str = [str sprintf([chr '%s'],vec{2:end})]; end
end
return;