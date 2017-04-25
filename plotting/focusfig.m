function focusfig(fig)

if nargin < 1, fig = gcf; end
n = get(fig, 'name');
if strcmp(get(fig, 'numbertitle'), 'on')
	if ~isempty(n), n = [': ' n]; end
	n = sprintf('Figure %g%s', fig, n);
end
res = deblank(evalc(['!sendxkey ''' n ''' raise focus input enter']));
if ~isempty(res), fprintf('%s\n', res); end