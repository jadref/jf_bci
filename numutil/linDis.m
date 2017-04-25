function [D]=linDis(X,Y,dim)
% Compute ecludian Distances between sets of points
%
% D = malDis(X,Z[,dim])
% Inputs:
%  X     - n-d matrix of N points
%  Z     - n-d matrix of M points, if empty then Z=X
%  dim   - dimensio of X,Z which contains the examples
% Outputs:
%  D     - [ N x M ] matrix of point pairwise distances
D=malDis(X,Y,[],dim);