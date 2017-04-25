function sensibleprint(fig, orient, fname)

if nargin < 1, fig = []; end
if nargin < 2, orient = ''; end
if nargin < 3, fname='/tmp/matlab.eps'; end

if isstr(fig) | ~isstr(orient), [fig orient] = deal(orient, fig); end
if isempty(fig), fig=gcf; end
if isempty(orient), orient = 'portrait'; end

set(fig, 'paperorientation', orient, 'papertype', 'a4', 'paperunits', 'normalized', 'paperposition', [0.05 0.05 0.9 0.9])

print(fig, '-depsc', fname)
if ( nargin < 3 ) 
   system(['lp -d lena ' fname]);
end
