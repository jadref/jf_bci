function saveaseps(fig, fname)

if nargin < 1, fig = []; end
if nargin < 2
   if ( isnumeric(fig) ) fname='untitled'; else fname=fig; fig=[]; end
end
if isempty(fig), fig=gcf; end

% append the pdf if not given.
%if ( fname(max(end-3,1):end)~='.pdf' ) fname=[fname '.pdf']; end;

% get the stuff we need.
if ( fname(1)=='~' && exist('glob')==2 ) fname=glob(fname); end;
% if ( fname(1)~=filesep )
%    curDir = cd();
%    fname = fullfile(curDir,fname);
% end

% store current state
fp.Units            =get(fig,'Units');
fp.PaperPositionMode=get(fig,'PaperPositionMode');
fp.PaperType        =get(fig,'PaperType');
fp.paperUnits       =get(fig,'PaperUnits');
fp.PaperSize        =get(fig,'PaperSize');
fp.PaperPosition    =get(fig,'PaperPosition');

% Change so print works nicely
set(fig,'Units','inches'); pos=get(fig,'position'); % get size in inches
set(fig,'PaperPositionMode','manual',...  % set print size to screen size
        'PaperType','<custom>',...
        'PaperUnits','inches',...
        'PaperSize', pos(3:4),...
        'PaperPosition',[0 0 pos(3:4)]);

% Do the print with matlab
print(fig,'-depsc2',fname);

% Restore the previous state
set(fig,fp);