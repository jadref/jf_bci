function xyz=latlong2xyz(latlong)
xyz= [sin(latlong(1,:)).*cos(latlong(2,:)); sin(latlong(1,:)).*sin(latlong(2,:)); cos(latlong(1,:))]; %3d
return
