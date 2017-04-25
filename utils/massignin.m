function []=massignin(ww,varargin)
for vi=1:2:numel(varargin);
  assignin(ww,varargin{[vi vi+1]})
end