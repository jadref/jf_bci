function [z]=jf_linDetrend(z,varargin)
% linear detrending function
z = jf_detrend(z,'order',1,varargin{:});
