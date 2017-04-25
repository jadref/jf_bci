function [lab]=ind2lab(ind,key)
% function [lab]=ind2lab(ind,key)
% convert indicator matrix to labels list.
% ind = (N x L) array of indicator functions.
if ( min(size(ind)) == 1 ) 
  error('ind must be matrix of indicator functions');return
elseif ( any(sum(ind,2)>1) ) 
  warning('Elements with more than 1 label...');
end
[ans,lab]=max(ind,[],2);
if ( nargin > 1 & ~isempty(key) ) lab=key(lab); end;
