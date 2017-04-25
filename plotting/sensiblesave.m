function [fname]=sensiblesave(fig, fname)
% save given figure as a eps+pdf pair of files
if nargin < 1, fig = []; end
if nargin < 2
   if ( isnumeric(fig) ) fname='/tmp/matlab'; else fname=fig; fig=[]; end
end
if isempty(fig), fig=gcf; end
[ans fname]=system(['echo ' fname]); fname(fname==10)=[]; % glob expand
set(fig, 'PaperPositionMode', 'auto');
%set(fig, 'XTickMode', 'manual');
%set(fig, 'YTickMode', 'manual');
%set(fig, 'ZTickMode', 'manual');
fname=[fname '.eps'];
if ( exist(fname,'file') ) system(['\rm -f "' fname '"']); end;
print(fig, '-depsc2', '-r300', fname);
% because matlabs pdf generator doesnt get the bounding box right
system(['epstopdf ' fname]);  
if ( exist(fname,'file') ) system(['\rm -f "' fname '"']); end;