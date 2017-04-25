function [z]=jf_plotERP(z,varargin)
% ERP visualation -- plot the mean of each class
%
% Options:
%   subIdx -- {cell ndims(z.X)x1} sub indicies of each dimension to plot
%   Y   -- [size(z.X,dim) x 1] or [size(z.X,dim) x nSubProb] labeling to 
%           use for the plotting                                             (z.Y)
%   Ydi -- [di-struct ndims(Y)+1 x 1] dimension info for the Y               (z.Ydi)
erpOpts=struct('Y',[],'Ydi',[],'method','mean','zeroLab',0);
[erpOpts,varargin]=parseOpts(erpOpts,varargin);

% compute the ERPs
z=jf_ERP(z,erpOpts);
% plot ERPs
jf_plot(z,varargin{:});

return;
