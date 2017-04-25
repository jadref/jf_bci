function [a]=jf_plotAUC(z,varargin)
% AUC visualisation
%
% Options:
%   Y   -- [size(z.X,dim) x 1] or [size(z.X,dim) x nSubProb] labeling to 
%           use for the plotting                                             (z.Y)
%   Ydi -- [di-struct ndims(Y)+1 x 1] dimension info for the Y               (z.Ydi)
%          OR
%          {str} list of dimension names which contain the trials
%  dim -- dim to treat as epoch dimension
%  subIdx -- subset of z to plot
aucopts=struct('Y',[],'Ydi',[]);
[aucopts,varargin]=parseOpts(aucopts,varargin);

% compute the auc
a=jf_AUC(z,aucopts);
% plot the result
jf_plot(a,'disptype','imaget','clim',[.2 .8],varargin{:});
set(findobj('type','axes'),'ygrid','on');
colormap ikelvin
return;
