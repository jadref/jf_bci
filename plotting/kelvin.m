function c = kelvin(m)

if nargin < 1, m = size(get(gcf,'colormap'),1); end

cl = [
	0.0            1/2   0.1   1.0
   0.25 - 0.125   1/2   1.0   0.9
   0.25 + 0.125   2/3   1.0   0.8
	0.5            2/3   1.0   0.0
];

cu = cl;
cu(:, 3:4) = flipud(cu(:, 3:4));
cu(:, 1) = cu(:, 1) + 0.5;
cu(:, 2) = cu(:, 2) - 0.5;

x = linspace(0, 1, m)';
l = (x < 0.5); u = ~l;
for i = 1:3
	h(l, i) = interp1(cl(:, 1), cl(:, i+1), x(l));
	h(u, i) = interp1(cu(:, 1), cu(:, i+1), x(u));
end
c = hsv2rgb(h);

return
% old version below

pr = [42.8002, -151.571, 209.707, -143.227, 51.4844, -8.9111, 0.718063];
pg = [-195.622, 584.241, -641.933, 313.093, -59.7597, -0.0279417, 1];
pb = [42.8002, -105.23, 93.8536, -35.8914, 6.334, -2.14836, 1];

if nargin < 1, m = size(get(gcf,'colormap'),1); end
c = makemap(m, pr, pg, pb);

function c = makemap(m, varargin)

c = ones(m, 3);
x = linspace(0, 1, m);
for i = 1:length(varargin)
	y = polyval(varargin{i}, x);
	if min(y) < 0, y = y - min(y); end
	if max(y) > 1, y = y / max(y); end
	c(:, i) = y(:);
end
