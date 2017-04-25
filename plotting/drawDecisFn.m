function [fx,cX1,cX2]=drawDecisFn(fn,minX,maxX,c1,c2)
if ( nargin < 4 ) c1='k'; end;
if ( nargin < 5 ) c2='r'; end;
hold on;
[cX1,cX2]=meshgrid(minX(1):(maxX(1)-minX(1))/100:maxX(1),...
                   minX(2):(maxX(2)-minX(2))/100:maxX(2));
fx=reshape(fn([cX1(:),cX2(:) repmat(minX(3:end),[length(cX1(:)),1])]),...
           size(cX1));
contour(cX1,cX2,fx,[0 0],c1);% do a contour plot of the decis fn.
if ( ~isempty(c2) ) contour(cX1,cX2,fx,[-1 -1; 1 1],c2); end;
%contour(cX1,cX2,fx,[-1:.1:1 -1:.1:1],'g');

