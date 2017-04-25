function ax=supplot(f)
% add a super-plot behind all other plots on the current figure
if ( nargin < 1 ) f=gcf; end;
figure(f);
ax=axes('Units','Normal','Position',[.08 .08 .84 .84],'Visible','off');
uistack(ax,'bottom'); % move behind other axes
set(get(ax,'Title'),'Visible','on');
set(get(ax,'XLabel'),'Visible','on');
set(get(ax,'YLabel'),'Visible','on');
