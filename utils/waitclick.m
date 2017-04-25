function [x] = waitclick(h)
if ( nargin < 1 ) h=gcf; end;
set(h,'units','normalized');  % set to 0-1 gui units...
set(h,'windowbuttonupfcn','uiresume');uiwait;x = get(h,'currentpoint');
set(h,'windowbuttonupfcn',[]);
