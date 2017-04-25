function [X,Y,Z,dZdx,dZdy]=plot2dFn(f,df,xs,ys,zs)
hstat=ishold;
if ( nargin < 3 | isempty(xs) ) xs=[-1.5:.02:1.5]; end;
if ( nargin < 4 | isempty(ys) ) ys=xs; end;
[X,Y]=meshgrid(xs,ys);Z=zeros(size(X));
for i=1:numel(Z); Z(i)=f([X(i);Y(i)]);end;
minZ=min(Z(:));maxZ=max(Z(:));rng=maxZ-minZ;
imagesc(xs,ys,Z,'AlphaData',.5);
hold on;
[c,h]=contour(X,Y,Z,[-1 0 1],'k-','LineWidth',1);clabel(c,h);
if ( nargin > 1 & ~isempty(df) ) 
   [Xd,Yd]=meshgrid(xs(1:3:end),ys(1:3:end));
   dZdx=zeros(size(Xd));dZdy=zeros(size(Yd));
   for i=1:numel(Xd); grad=df([Xd(i);Yd(i)]);dZdx(i)=grad(1);dZdy(i)=grad(2);end;
   hold on; quiver(Xd,Yd,dZdx,dZdy);
end
if ( ~hstat ) hold off; end;
