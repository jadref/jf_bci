function [z,opts]=center(z,varargin)
% center over the given dimension -- N.B. just a wrapper round re_ref
z=jf_reref(z,varargin{:},'summary','centering');
return;
