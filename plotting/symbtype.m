function [s]=symbtype(c)
% [s]=symbtype(c)
mlock;  % lock this function in memory so clears don't affect it.
symbs='sdv^<>ph*+o';
persistent cursymb; 
if ( isempty(cursymb) ) cursymb=0; else cursymb=cursymb+1; end; % init static
if ( nargin > 0 && ~isempty(c) )  cursymb=c; end; % arg over-rides
s=[symbs(mod(cursymb-1,length(symbs))+1)];        
