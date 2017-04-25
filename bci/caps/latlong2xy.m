function xy=latlong2xy(latlong)
xy = [latlong(1,:).*cos(latlong(2,:));latlong(1,:).*sin(latlong(2,:))]; % 2d
return
