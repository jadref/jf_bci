function [x,y,fs,trialinfo,summary,opts,info] = readraw_comp3iiia(z, files, fileindex, varargin)

opts = {
	'interval' [3.5 7]
	     'set' 'all'
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

load(filename);
fs = HDR.SampleRate;

nt = numel(HDR.TRIG);
nchan = size(s, 2);
sampind = round(opts.interval * fs);
nsamp = diff(sampind) + 1;
sampind = sampind(1):sampind(2);

x = zeros(nt, nchan, nsamp);
y = HDR.Classlabel;
for trial = 1:nt
	off = HDR.TRIG(trial)-1;
	x(trial, :, :) = permute(s(sampind+off, :), [3 2 1]);   
end

cut = [];
switch opts.set
	case 'train', cut = isnan(y);
	case 'test', cut = ~isnan(y);
	case 'all', cut = [];
	otherwise, error('haeh?')
end

% include the true-labels for the testing set if available
labFn=[filename(1:find(filename=='/',1,'last')) 'true_label_' filename(find(filename=='/',1,'last')+1:end-5) '.txt'];
if ( exist(labFn,'file') )
   tstLabs=load('-ascii',labFn);
   y=tstLabs;
end

x(cut, :, :) = [];
y(cut) = [];
summary = sprintf('interval %g--%g sec', opts.interval(1), opts.interval(2));

