function [stat]=fscanto(fid,flag,past)
% scan until flag is met.
% flag can be a simple string or a function.
%  if a function it must return the index of where should continue reading
%   from. (past ignored in this case) 
if ( nargin < 2 ) error('Insufficient arguments'); end;
if ( ischar(flag) )
  endp=@(instuff) strIdx(instuff,flag) 
elseif ( isa(flag,'function_handle') )
  endp=flag
else
  error('Unrecognised type of flag');
end
% first scan until we see flag or eof
instuff=fgets(fid)
while ( ischar(instuff) && ~endp(instuff) ) 
  instuff=fgets(fid)
end
if ( ischar(instuff) )
  fidx=endp(instuff); % get the start of the flag.
  if ( nargin < 3 | ~ischar(flag) | ~past ) % scan to
    stat=fseek(fid,fidx-length(instuff)-1,0);
  else % scan past
    stat=fseek(fid,fidx+length(flag)-length(instuff)-1,0); 
  end      
else
  stat=-1; % return error status
end

