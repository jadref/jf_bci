function [pts]=pointsOnSphere(N,method,height)
% generated equally spaced points on the sphere using different methods
%
% [pts]=pointsOnSphere(N,method,height)
if ( nargin < 2 || isempty(method) ) method='golden'; end;
if ( nargin < 3 || isempty(height) ) height=2; end;
pts=[];
step=1; if ( method(end)>'0' && method(end)<'9' ) step=str2num(method(end)); method(end)=[]; end;
switch (method);
 case 'golden';  dlong = pi*(3-sqrt(5));        dz = height/N;
 case 'saff';    s     = sqrt(2*height*pi/N);   dz = height/N; % N.B. s=arc-length round circum btw pts
 case 'equal';   s     = sqrt(2*height*pi/N);   dz = height/N;       dlong = inf; 
 otherwise;  error('Unrecog method');     
end
long = 0;
z    = 1 ;%- dz/2;
r    = sqrt(1-z*z);
for k = 1 : N;
    pts(:,k) = [cos(long)*r, sin(long)*r, z];
    switch (method);
     case 'golden'; 
      z    = z - dz;
      r    = sqrt(1-z*z);
      long = long+dlong;
     case 'saff';    
      z    = z - dz;
      r    = sqrt(1-z*z);
      long = long+s/(r+single(r==0));
     case {'equal','equal2','equal4'};  
      long = long+dlong;
      if ( long>2*pi-dlong/2 ) % step increment
        if ( k==1 ) cPts=1; end;
        z   = z-(dz*cPts);
        r   = sqrt(1-z*z);
        long=0; 
        cPts= round(2*pi*r/s/step)*step; % num pts in this ring
        if ( k+cPts<N-cPts/2 )    dlong = 2*pi ./ cPts; % equally spaced round circle
        else dlong = 2*pi ./ max(1,(N-k));  % too many points, equal spaced last row
        end;
      end; 
    end
end
return;
%--------------------------------
function testCase();
scatPlot(pointsOnSphere(10))