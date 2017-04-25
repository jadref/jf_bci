function [maxA,I]=mmax(A,dims)
% multi-dimensional maximum
[maxA,I]=mmin(-A,dims); maxA=-maxA;