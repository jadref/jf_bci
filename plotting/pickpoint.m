function [x] = pickpoint(ah)
if ( nargin < 1 ) ah=gca; end;

% get the mouse-click
fh=get(ah,'parent');
set(fh,'units','normalized'); % N.B. necessary as axes reports norm units!
x=waitclick(fh);
% set(fh,'windowbuttonupfcn','uiresume');uiwait;
% x = get(fh,'currentpoint');
% set(fh,'windowbuttonupfcn',[]);

% convert from window co-ords to data-co-ords
% get pos of axes -- assume's only child of window is plot axes 
set(ah,'units','normalized');
apos = get(ah,'position'); % in [xl,yb,xr,xt] format
% convert position to normalised point in plot window
x=(x-apos(1:2))./[alim(3:4)-alim(1:2)];
% get data limits of the axes, into [xl,yb,xr,yt] format
drng([1 3])=get(ah,'Xlim');drng([2 4])=get(ah,'YLim');
% convert to data units
x=(x.*[drng(3:4)-drng(1:2)])+drng(1:2);
