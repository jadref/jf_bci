function [d]=m2p(prep,nm,exactmatchp,mustmatchp,mmatchp)
% convert name prep structure
%
%  [d]=m2p(prep,nm,exactmatchp,mustmatchp)
if ( nargin < 3 ) exactmatchp=[]; end;
if ( nargin < 4 ) mustmatchp=[]; end;
if ( nargin < 5 ) mmatchp=0; end;
if ( numel(prep)==1 && isfield(prep,'prep') ) prep=prep.prep; end;
if ( iscell(prep) && ischar(prep{1}) ) strs=prep; 
elseif ( isstruct(prep) ) strs={prep.method}; 
end;
d = n2d(strs,nm,exactmatchp,mustmatchp,mmatchp); % use n2d to do the work
return;
