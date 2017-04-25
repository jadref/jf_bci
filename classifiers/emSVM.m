function [alpha,b,svIdxs,Yp,Ymu,Ysigma]=emSVM(X,pY,wght,kerFn,pen,alpha,b,termParms,clsParms,verb)
% function [alpha,b,svIdxs,Yp,Ymu,Ysigma]=emSVM(X,pY,wght,kerFn,pen,alpha,b,termParms,clsParms,verb)
%
% An implementation of the em-SVM algorithm for computing a maximum
% likelihood classifier when the data has uncertain/unknown labels.
%Inputs:
% X  -- [Nxd]
% pY -- prior probability that this point has label 1, thus pY(i,1)=1 means
%       certainty that point i has a label of Y==1. [NxL] 
%                       or for binary classification problems [Nx1]
% wght    -- importance weight for this point in the loss function.
% kerFn -- kernel function, K(i,j) = kerFn(X(i,:),X(j,:))
% pen -- penalty term in the loss function
% alpha,b -- initial solutions.[Nx1],[1x1]
% termParms -- termination parameters for the em, [maxIt,tol]
% clsParms  -- classifier parameters, for the SVM training.
% verb -- verbosity level.
%Outputs:
% alpha,b -- final solution, [Nx1],[1x1]
% svIdxs  -- indices in X of the support vectors.
% Yp,Ymu,Ysigma -- parameters for the probabilistic label predictions,
%                  essentially this is modelled as a gaussian mixture over
%                  f(x) with 1 component for each label. Therefore,
%                         Pr(y|x) = Pr(f(x)|y)Pr(y)/Pr(f(x)), 
%                  where Pr(f(x)|y)=Norm(f(x)|Ymu,Ysigma), Pr(y)=Yp
%
% $Id: emSVM.m,v 1.4 2006-11-16 18:23:48 jdrf Exp $

%Process inputs:
if ( nargin < 5 ) error('Insufficient arguments');end;
if ( nargin < 6 ) alpha=[]; end;
if ( nargin < 7 | isempty(b) ) b=0; end
if ( nargin < 8 | isempty(termParms) ) termParms=[10 1e-4]; end;
if ( nargin < 9 ) clsParms=[]; end;
if ( nargin < 10 | isempty(verb) ) verb=0; end;


% Intialisation: set initial label probabilities to the prior ones.
Pyx=pY;  if ( size(Pyx,2) == 1 ) Pyx(:,2)=1-Pyx(:,1); end;
[N,dim]=size(X);

% Loop the required number of times.
for iter=1:termParms(1);

  %E-Step: compute the predicted point labels and weights.
  % N.B. equiv to Class EM, i.e. all alloc to closest cent
  [pYest,Yest] = max(Pyx,[],2);              % N.B. Yest is col idx, 1,2,3...
  c = (pYest - min(Pyx,[],2)).*wght;         % orginal weight * label certainty
  
  %M-Step: maximise the classification performance usign an svm classifier.
  % BODGE: for now just use the linear adatron algorithm, i.e. ignore the kerFn
  [w,b,alpha,svs]=adatron(X,Yest,c,pen,alpha,b,clsParms,verb);
  
  % Generate probabilistic predictions from the classifier outputs and priors

  % 1. Compute the output prediction statistics
  f=X*w+b; 
  nY=[sum(Yest==1) sum(Yest==2)];
  muY=[sum(f(Yest==1))./nY(1) sum(f(Yest==2))./nY(2)];
  sigmaY=[sum(f(Yest==1).^2)./nY(1) - muY(1)*muY(1) ...
          sum(f(Yest==1).^2)./nY(1) - muY(1)*muY(1) ];
  
  % 2. Compute new posterior predicted labels.
  % 2.1 Compute the conditional probability
  Pfxy = [ exp(-.5*(f-muY(1)).^2./sigmaY(1))./sqrt(2*pi*sigmaY(1)) ...
           exp(-.5*(f-muY(1)).^2./sigmaY(1))./sqrt(2*pi*sigmaY(2))];    
  Pfxy = Pfxy.*pY;                              % 2.2 include the prior over Y
  Pyx  = Pfxy ./ repmat(sum(Pfxy,2),[1,2]);     % 2.3 compute the posterior
end;