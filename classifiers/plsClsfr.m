function [wb,f,J]=plsClsfr(X,Y,C,varargin)
% weiner filter based classifier
%
% [wb,f,J,p,M]=plsClsfr(XXytau,Y,C,varargin)
%
% Inputs:
%  XXytau   -- [n-d] data matrix, this should be cov XX and XY for different time shifts
%                    concatenated along dim 2 [X*X' X*Y']
%                   N.B. we assume, X = [ch_x x [ch_x;ch_y] x tau x epoch]
%  Y        -- [N x nCls] +1/0/-1 class indicators, 0/NaN entries are ignored
%  C        -- [1x1] regularisation weight (ignored)
% Options:
%  dim      -- [1x1] dimension which contains epochs in X (ndims(X))
%                   N.B. we assume, X = [ch_x x [ch_x;ch_y] x tau x epoch]
%                   where ch_x are channels of independent variable (data)
%                         ch_y are channels of dependent variable to be predicted
%                         tau are the temporal offsets
%  wght     -- [2x1] class weighting for the prototype,      ([1 -1])
%                     W = mean(X;Y>0)*wght(1) + mean(X;Y<0)*wght(2)
%  clsfr    -- [bool] act like a classifier, i.e. treat each ch_y as the stimulus (1)
%                     for a different class and return f-value which should be 
%                     max for the predicted class
%  rank     -- [int] rank of inner solution to generate
% Outputs:
%  wb       -- [size(X,1) size(X,3)] spatio-temporal weighting matrix
%  f        -- [Nx1] set of decision values
%  J        -- [1x1] obj fn value
[wb,f,J]=wienerClsfr(X,Y,C,'rank',1,varargin);
return;
