function [z]=jf_reject(z,varargin);
% function to remove elements of the jf-data-struct
%
% Options:
%  dim  - dimension to reject along
%  vals - element values to remove/keep
%  idx  - element indicies to remove/keep
%         N.B. negative idx index from the end back
%  range- idx/vals specifies a range to remove. which is either:
%         'before','after','between','outside' the spec elements
%  mode - {'reject','retain'} reject or retain the indicated elements
%  strregexp - [bool] allow regular expression string value matches?
%  summary -- optional additional information summary string
z=jf_retain(z,'mode','reject',varargin{:});
z.prep(end).method=mfilename;