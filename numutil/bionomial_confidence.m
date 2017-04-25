function [epsilion]=binomial_confidence(N,delta,p)
% Compute confidence bound for the auc statistic.
%
% [epsilion]=auc_confidence(N,rho,delta)
% N     - number samples
% delta - probability true value within given bound, ie.confidence level (.05)
% p     - value to estimate bound about (.5)
%
% Based on the central limit approximation as outlined on the wikipedia page
if ( nargin < 2 ) delta=.05; end;
if ( nargin < 3 ) p=.5; end;
% find the z-score for this confidence bound
h=.1; xs=[0:h:10]; normcdf = .5 + cumsum(exp(-.5*(xs).^2).*h./sqrt(2*pi)); normcdf=normcdf+min(0,(1-normcdf(end)));
z_delta = xs(find(normcdf>1-delta/2,1));
% comput the bound
epsilion= z_delta*sqrt(p*(1-p)/N);
