function bounds = r2_confidence(N, r, z)
% Compute confidence bound for pearson's correlation coefficient
%
%
% [bounds]=r2_confidence(N)
% N     - number of samples
% returns standard error of a correlation coefficient, ie.r=0 (confidence level .05)
%
% [bounds]=r2_confidence(N,r,z)
% N     - number samples
% r     - pearson's correlation coefficient
% z     - probability true value within given bound, ie.confidence level (.05)
%               important values: z = 1.96 (95%); 2.2414 (97.5%) 2.6467 (99%)
%
% Based upon:
% http://davidmlane.com/hyperstat/B8544.html
if nargin  == 0
    error('Need to specify the number of samples at least!')
end


if (nargin < 2) r = 0; end, % results in zprime = 0;
if (nargin < 3) z = 1.96; end, % z = 1.96 (95%); 2.2414 (97.5%);2.6467 (99%)

stderr = 1/sqrt(N-3);
zprime = atanh(r);

if nargin == 1
    if (N < 3)
        bounds = inf;
        warning('N has to be larger than 3, returning inf.');
    else
        bounds= z*stderr;
    end;
else
    if (N < 3)
        bounds(1) = inf;
        bounds(2) = -inf;
        warning('N has to be larger than 3, returning +/- inf.');
    else
        bounds(1) = zprime + z*stderr;
        bounds(2) = zprime - z*stderr;
    end
end


end