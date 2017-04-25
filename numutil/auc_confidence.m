function [epsilion]=auc_confidence(N,delta,rho)
% Compute confidence bound for the auc statistic.
%
% [epsilion]=auc_confidence(N,delta,rho)
% N     - number samples
% delta - probability true value within given bound, ie.confidence level (.05)
% rho   - fraction of positive examples, N_+/N     (.5)
%
% Based upon collorary 2 in:
% Shivani Agarwal, Thore Graepel, Ralf Herbrich, Dan Roth, A Large Deviation
% bound for the Area Under the ROC Curve, NIPS 17, 2005
if ( nargin < 2 ) delta=.05; end;
if ( nargin < 3 ) rho  =.5; end;
epsilion=sqrt(log(2./delta)./(2*rho.*(1-rho)*N))/3;
