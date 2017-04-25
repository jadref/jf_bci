function []=scatPlot(xs,varargin)
% Plot set points
% function []=scatPlot(xs,varargin)
%if ( size(xs,ndims(xs)) < size(xs,1) ) xs=permute(xs,fliplr(1:ndims(xs))); end
if ( size(xs,1) == 1 ) plot(xs,varargin{:});
elseif ( size(xs,1) == 2) plot(xs(1,:),xs(2,:),varargin{:});
else plot3(xs(1,:),xs(2,:),xs(3,:),varargin{:});
end