function rgb = rainbow(n);

% RAINBOW(n) creates a colormap, ranging from blue via green to red.
% Similar to 'jet', but without the darkening at the ends.
%
% Nico Sneeuw
% Munich, 30/08/94

if nargin == 0, n = size(get(gcf,'colormap'),1); end

m = fix(n/3);
step = 1/m;
ltop = ones(m+1,1);
stop = ones(m,1);
lbot = zeros(m+1,1);
sbot = zeros(m,1);
lup = (0:step:1)';
sup = (step/2:step:1)';
ldown = (1:-step:0)';
sdown = (1-step/2:-step:0)';
if n - 3*m == 2
   rgb = [lbot lup ltop; sup stop sdown; ltop ldown lbot];
elseif n-3*m == 1
   rgb = [sbot sup stop; sup stop sdown; ltop ldown lbot];
else
   rgb = [sbot sup stop; sup stop sdown; stop sdown sbot];
end