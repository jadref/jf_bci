function [idx]=strIdx(st,substr)
%% fast find first occurace of substr in st
if ( nargin < 2 ) warning('Needs 2 arguments'); return; end

[starts]=strfind(st,substr);
if ( ~isempty(starts) ) idx=starts(1); else idx=0; end;

% [starts] = find ( st(1:end-length(substr)) == substr(1) ); 
% idx=0; l=length(substr)-1;
% for i=starts
%   if ( all(st(i:i+l)==substr) ) idx=i; break; end
% end

% for i=2:length(substr)
%   starts=starts(find( st(starts+1) == substr(i) ));
%   if(isempty(starts)) break; end;
% end
% idx=starts(1);

