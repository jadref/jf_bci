function latlong=xy2latlong(xy);
% convert xy to lat-long, taking care to prevent division by 0
latlong= [sqrt(sum(xy.^2,1)); atan2(xy(2,:),xy(1,:))];
return
