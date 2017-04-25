function [x,y,fs,trialinfo,summary,opts,info] = readraw_comp3v(z, files, fileindex, varargin)

opts = {
	'duration_msec' 1000
	   'max_trials' inf
}';
error(resolveopts(varargin, opts))

x = [];
y = [];
fs = [];
trialinfo = [];
summary = '';
info=struct('filename',filename);

filename = files{fileindex};
if ~strncmp(lower(fliplr(filename)), 'tam.', 4), return, end

v = load(filename);

[nsamp nchan] = size(v.X);
if ~isfield(v, 'Y'), v.Y = zeros(nsamp, 1); end
fs = v.nfo.fs;

len = round(fs * opts.duration_msec / 1000);
off = 0;

ntrials = 0;
maxntrials = floor(nsamp/len);
x = zeros(maxntrials, nchan, len);
y = zeros(maxntrials, 1);
while 1
	ind = off + [1:len];
	if ind(end) > nsamp, break, end
	change = min(find(diff(v.Y(ind))));
	if isempty(change)
		trial = v.X(ind, :);
		ntrials = ntrials + 1;
		x(ntrials, :, :) = permute(trial, [3 2 1]);
		y(ntrials) = v.Y(ind(1));
		off = ind(end);
	else
		off = off + change;
	end
end
x(ntrials+1:end, :, :) = [];
y(ntrials+1:end) = [];

summary = sprintf('%g-msec sections', opts.duration_msec);
if size(x, 1) > opts.max_trials
	error('not implemented yet')
	summary = sprintf('%d x %s', opts.max_trials, summary);
end
