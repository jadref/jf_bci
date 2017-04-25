function D = sphericalInterpolate(electr, loc, perc);
%interpolate matrix for spherical interpolation
%
% D = sphericalInterpolate(electr, loc, perc);
%
% Inputs:
%  electr - [N x 3] old electrode positions
%  loc    - [M x 3] new electrode positions
%  perc   - [1x1] smoothness. the percentage of the value that should be  (.5)
%           given by the direct neighbours. So if we go from 256 to 32 
%           electrodes, we want that 256/32 = 8 electrodes account for perc
%           percent. The larger this value, the more local the interpolation. 

if ( nargin < 3 || isempty(perc) ) perc=.5; end;
if ( size(loc,1)~=3    && size(loc,2)==3 )    loc=loc'; end;
if ( size(electr,1)~=3 && size(electr,2)==3 ) electr=electr'; end;

%calculate the cosine of the angle between the new and old electrodes. If
%the vectros are on top of each other, the result is 1, if they are
%pointing the other way, the result is -1
cosEE = tprod(electr,[-1 1],loc,[-1 2]); % all pairs product matrix

%get values between 1 and 0
D = ((cosEE + 1)/2)';

%power D so that approximately only the ones in the cluster build it.
go = 1;
while(go)
    % calculate the number of elements that contribute for 90% to the
    % interpolation
    nrcontr = contribute(D, perc);
    
    %this should be equalish to the number in the cluster
    if(nrcontr < (size(loc,1)/size(electr,1) ))
        go = 0;
    else
        D = D.*D;
    end
end

%normalise the rows so that they sum to 1
D = repop(D,'./',sum(D,2));

return;

%---------------------------------------------------------------------------
function nrcontr = contribute(D,perc)
%function to determine which elements contribute to the value of the
%interpolated electrode.
%
% nrcontr = contribute(D,perc)
%
% D;    matrix of new_electr x electr
% perc: percentage
% 
% the output is the mean number of elements in the matrix D that contribute
% for perc percent to the value of the new interpolated electodes. This
% function is called to determine a smoothing function.

%sum of the rows is used to normalise
sumv = sum(D,2);

%how many sum to perc% of this value
D = sort(D,2,'descend');
D = cumsum(D,2)./(repmat(sumv,1,size(D,2)));

%determine howmany elements per row are used
contr = logical(D < perc);

%add these elements per row
sumv = sum(contr,2);

%get the mean
nrcontr = mean(sumv);


