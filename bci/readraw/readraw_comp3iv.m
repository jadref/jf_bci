function [x,y,fs,trialinfo,summary,opts,info] = readraw_comp3iv(z, files, fileindex, varargin)

opts = {
	          'set' 'both'
	'interval_msec' []
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
switch lower(opts.set)
	case 'both', 0;
	case 'test'
		if isempty(findstr('test', lower(filename))), return, end
		summary = 'test trials, ';
	case 'train'
		if isempty(findstr('train', lower(filename))), return, end
		summary = 'training trials, ';
	otherwise
		error('unrecognized ''set'' option')
end

v = load(filename);

[nsamp nchan] = size(v.cnt);
if ~isfield(v.mrk, 'y'), v.mrk.y = nan * ones(size(v.mrk.pos)); end
if ( any(isnan(v.mrk.y) ) ) % load the true labels
   labFn=[filename(1:find(filename=='/',1,'last')) 'true_labels' filename(find(filename=='_',1,'last'):end)];
   if ( exist(labFn,'file') )
      load(labFn)
      v.mrk.y=true_y;
   end
end

fs = v.nfo.fs;
maxlen = min(diff(v.mrk.pos));

if isempty(opts.interval_msec)
	off = 0;
	len = maxlen;
	opts.interval_msec = round(1000 * [off off+len]/fs);
else
	off = fs * opts.interval_msec(1) / 1000;
	len = fs * diff(opts.interval_msec) / 1000;
end

ntr = numel(v.mrk.pos);
nch = size(v.cnt, 2);
x = zeros([ntr nch len]);
ind = off + [1:len]-1;
for i = 1:ntr
	x(i, :, :) = permute(0.1*double(v.cnt(v.mrk.pos(i)+ind, :)), [3 2 1]);
end
y = v.mrk.y(:);

summary = sprintf('%sinterval %g-%g sec', summary, opts.interval_msec(1)/1000, opts.interval_msec(2)/1000);
