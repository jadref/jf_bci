function []=triplot2(X,Y,Z,C)
if ( nargin < 4 )   
   trisurf(delaunay(X,Y),X,Y,zeros(size(X)),Z);
   view(0,90)
else
   %trisurf(delaunay3(X,Y,Z),X,Y,Z,C);
   trisurf(delaunay(X,Y),X,Y,Z,C);
end;
shading interp
axis off
