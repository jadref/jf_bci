function [s]=linetype(c)
mlock;  % lock this function in memory so clears don't affect it.
cols='bgrcmyk';
styles={'-','--','-.',':'};
persistent curcol;
if ( isempty(curcol) ) curcol=0; else curcol=curcol+1; end; % init static
if ( nargin > 0 && ~isempty(c) )  curcol=c; end; % arg over-rides
% set line type
s=[cols(mod(curcol-1,length(cols))+1) ...
   styles{mod(ceil(curcol/length(cols))-1,length(styles))+1}];
