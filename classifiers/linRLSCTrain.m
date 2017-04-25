function [alpha,svs,bias]=linRLSCTrain(X,Y,lambda,trainType)
% function [alpha,svs,bias]=linRLSCTrain(X,Y,L,lambda,trainType,penScl)
% 
% Train an linear least squares regularised classifier.
% Inputs:
%  X -- the input data set, [N x d]
%  Y -- the target values   [N x L]
%  lambda -- the regularisation parameter. [1x1] (1/N)
%  trainType -- type of training, either 'class','regress'.  When using
%               'class' Y must contain a discrete set of labels.  If
%               using 'regress' then Y is treated as a real valued output
%               vector. (class)
% Ouputs:
%  alpha -- the weighting factor for each support vector.
%  svs   -- the support vectors.
%
% The general least squares classifier is the solution to the equation:
%  (K + N\lamba\eye)w = Y
% which simplifies in the linear case to:
%  (X'X + N\lambda\eye) w = Y
% which is solved efficiently using the Sherman-Morrision-Woodbury
% identity to give:
%  w = (\eye/(N\lambda) - X'(\eye + XX'/(N\lambda))^-1 X)/(N\lambda)^2 Y
%  w = (\eye - X'(\eye*(N\lambda) + XX')^-1 X) Y / (N\lambda)
% This allows us to solve this problem by only storing and inverting a
% [d x d] matrix rather than [N x N].

if ( nargin < 2 ) help linRLSCTrain; return; end;
if ( nargin < 3 | isempty(lambda) ) lambda=[]; end;
if ( nargin < 4 | isempty(trainType) ) trainType='class'; end;

if ( strcmp(trainType,'class') ) % classifier training
  if ( size(Y, 1) == 1 | size(Y, 2) == 1 )   % If form [1 2 2 3 3 1 2 ...]
    Y=lab2ind(Y);
  end
  L=size(Y,2);
elseif ( strcmp(trainType,'regress') ) % regression training
  L=size(Y,2);  % L is number of output dimensions.
else
  error('Unknown type of training...');
end
[N,d]=size(X);
if ( isempty(lambda) ) lambda=1/N; end % default regularisation..

% N.B. we use an agumented feature vector rather than an explict bias,
% so X_2 = [X, 1] = [ X , ones(1,N) ]
% hence: [X,1]'*[X,1] = [ X'X X'1 ] = [ X'X   X'1 ] = [ X'X    sum(X,1)]
%                       [ 1'X 1'1 ]   [(X'1)'  N  ]   [sum(X,1)'   N   ]
N1 = ones(N,1);
if ( lambda ) 
  w=zeros(N,length(L));
  SX = sum(X,1);
%   t=([X,N1]*[X,N1]'+eye(N,N)*(lambda*N));
%   w=lscov(t,Y(:,1));
%   w2=t\Y(:,1);
%   % compute the compressed weighting vector.
%   % from the representor theorm: f(x) = \sum_{i\in svs} \alpha_i K(x,svs_i)
%   % in the linear case this degenerates to: 
%   %  \sum_{i \in svs} \alpha_i x' x_i = x' (\alpha'X)' = (\alpha'X)x
%   % which is quivalent to having single SV of (\alpha'X) of unit weight.
%   svs=w'*X;
%   bias=sum(w,1);
%   alpha=1;

  %Use matlab's least squares solver to find the solution -- this seems
  %to be the most stable.
  w=(eye(d+1,d+1)*(lambda*N)+[X'*X, SX'; SX N])\([X, N1]'*Y);
  %w=lscov(eye(d+1,d+1)*(lambda*N)+[X'*X, SX'; SX N],([X, N1]'*Y));
  % extract the bias term and turn into svs format again.
  svs=w';
  bias=svs(:,end);
  svs=svs(:,1:end-1);
  alpha=ones(length(L),1);
else % no-regularisation so use the peusdo inverse, which is the limiting case.
  t=pinv([X, N1]); % pinv = (X'X)^-1 X'
  svs=(t*Y)';
  svs = ([X,N1]\Y)'; % just use least squares solution
  % extract the bias term and turn into svs format again.
  bias=svs(:,end);
  svs=svs(:,1:end-1);
  alpha=ones(1,length(L));
end
