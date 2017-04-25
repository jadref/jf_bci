function sensiblesve(fig, fname)

if nargin < 1, fig = []; end
if nargin < 2
   if ( isnumeric(fig) ) fname='/tmp/matlab'; else fname=fig; fig=[]; end
end
if isempty(fig), fig=gcf; end

% Get info to compute the right new size
pos=get(fig,'position');
sz=pos(3:4)/get(0,'screenpixelsperinch');

% store current state
paperposmode=get(fig,'PaperPositionMode');
papertype   =get(fig,'PaperType');
paperunits  =get(fig,'PaperUnits');
papersz     =get(fig,'PaperSize');
paperpos    =get(fig,'PaperPosition');

% Change so print works nicely
set(fig,'PaperOrientation',orient,...
        'PaperPositionMode','manual',...
        'PaperType','<custom>',...
        'PaperUnits', 'inches',...
        'PaperSize',sz,...
        'PaperPosition',[0 0 sz]);
% Do the print
print(fig,'-dpdf',fname);

% Restore the previous state
set(fig,'PaperPositionMode',paperposmode);
set(fig,'PaperType',papertype);
set(fig,'PaperUnits',paperunits);
set(fig,'PaperSize',papersz);
set(fig,'PaperPosition',paperpos);
