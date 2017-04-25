function [s]=markerType(c)
mlock;  % lock this function in memory so clears don't affect it.
styles='.ox+*sdv^<>ph';
persistent curmarker;
if ( isempty(curmarker) ) curmarker=0; else curmarker=curmarker+1; end; % init static
if ( nargin > 0 && ~isempty(c) )  curmarker=c; end; % arg over-rides
% set line type
s=[styles(mod(curmarker-1,length(styles))+1)];
