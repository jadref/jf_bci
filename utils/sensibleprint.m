function sensibleprint(fig, orient)

if nargin < 1, fig = []; end
if nargin < 2, orient = ''; end

if ischar(fig) | ~ischar(orient), [fig orient] = deal(orient, fig); end
if isempty(fig), fig=gcf; end
if isempty(orient), orient = 'portrait'; end

set(fig, 'paperorientation', orient, 'papertype', 'a4', 'paperunits', 'normalized', 'paperposition', [0.05 0.05 0.9 0.9])

print(fig, '-depsc', '/tmp/matlab.eps')
!lp -d HP9500 /tmp/matlab.eps
