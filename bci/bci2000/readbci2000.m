function s = readbci2000(fn, maxnsamples)
% S = READBCI2000(FILENAME   [, MAX_NUMBER_OF_SAMPLES])
% 
% Use S with BCI2000PARAM and BCI2000STATE

s.signal = [];
s.state = [];

verbose = 1;

if nargin < 2, maxnsamples = inf; end
if nargin < 1, fn = ''; end

if isempty(fn), fn = cd; end
if isdir(fn)
	oldd = cd;
	cd(fn)
	[fn d] = uigetfile('*.dat', 'Select BCI2000 file to read');
	if isempty(fn) | ~isstr(fn) | ~isstr(d), fn=''; else fn = fullfile(d, fn); end
	cd(oldd)
end

if isempty(fn), s = []; return, end
fid = fopen(fn);
if fid == -1, error(sprintf('failed to open ''%s''', fn)), end

[s.headline, err] = headline2struct(fgetl(fid)); % e.g.: 'HeaderLen= 4481  SourceCh= 16 StatevectorLen= 11 DataFormat='float32' '
if ~isempty(err), fclose(fid); error(err), end
if ( isfield(s.headline,'DataFormat') ) 
   switch s.headline.DataFormat;
     case {'float32','single'}; s.headline.DataFormat='single'; s.headline.sampleLen=4;
     case 'int32';   s.headline.sampleLen=4;
     case 'int16';   s.headline.sampleLen=2;
    otherwise; warning('Unrecoginised dataformat: %s',s.headline.dataFormat);
   end
else
   s.headline.DataFormat='int16'; % data format used to store the data 
   s.headline.sampleLen =2;       % length of each sample in bytes
end
t = readtext(fid, s.headline.HeaderLen - ftell(fid));
[s.descriptors, s.comments, s.types, err] = parsedescriptors(t);
if ~isempty(err), fclose(fid); error(err), end

nchannels = s.headline.SourceCh;
statelen  = s.headline.StatevectorLen;
sampleType= s.headline.DataFormat;
sampleLen = s.headline.sampleLen;
oneunit = sampleLen * nchannels + statelen; % one sampleLen-byte word per sample per channel, followed after last channel by statelen bytes

pos = ftell(fid);
fseek(fid, 0, 'eof');
nbytes = ftell(fid) - pos;
fseek(fid, pos, 'bof');

nsamples = floor(nbytes / oneunit);
extra = nbytes - nsamples * oneunit;
if extra, warning(sprintf('file ''%s'' has %d extraneous bytes (nbytes = %d, sample length = %d)', fn, extra, nbytes, oneunit)), end

if maxnsamples < nsamples
	nsamples = maxnsamples;
	if verbose
		if nsamples == 0
			fprintf('reading BCI2000 parameter-header only\n');
		else
			fprintf('reading only the first %d samples of the recording\n', nsamples);
		end
	end
end

if verbose & nsamples, fprintf('reserving %d MB...\n', round(nbytes/1024^2)), end
s.signal = repmat(feval(sampleType,0), [nchannels nsamples]); % pre-alloc storage array
s.state  = repmat(uint8(0), [statelen nsamples]);

nc = -1; t = 0; interval = 2 / (24 * 60 * 60);
if verbose & nsamples, fprintf('reading data from %s\n', fn); t = now + interval; end

for isample = 1:nsamples
	s.signal(:, isample) = fread(fid, [nchannels 1], sampleType);
	s.state(:, isample)  = fread(fid, [statelen 1], 'uint8');

	if verbose & ~rem(isample, 10000)
		if now > t
			if nc < 0, nc = fprintf('percent done: '); end
			if nc > 72, fprintf('\n'); nc = fprintf('              '); end
			nc = nc + fprintf('% 3d ', round(100*isample/nsamples));
			t = now + interval;
		end
	end
end
if nc > 0, fprintf('\n'); end
fclose(fid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s,err] = headline2struct(t)

s = [];
err = '';

t = fliplr(deblank(fliplr(deblank(t))));
key = []; val = [];
while ~isempty(t)
	[val t] = strtok(t, char([9:13 32]));
	if val(end)=='='
		key = val;
	elseif isempty(key)
		err = sprintf('headline parse error: stopped at ''%s''', t);
		return
	else
		if all(ismember(val, '0123456789. -+eE')), val = eval(['[' val ']'], 'val'); end
		s = setfield(s, key(1:end-1), val);
	end
end	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t = readtext(fid, nchars)

t = char(fread(fid, [1 nchars], 'uint8'));
cr = char(13); lf = char(10);
t = strrep(t, [cr lf], lf);
t = strrep(t, cr, lf);
t = strrep(t, char(9), ' ');
while any(findstr([t 'xx'], '  ')), t = strrep(t, '  ', ' '); end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s, comments, types, err] = parsedescriptors(t)

s = [];
comments = [];
types = [];
err = '';

section = '';
while ~isempty(t)
	[lin t] = strtok(t, char(10));
	lin = fliplr(deblank(fliplr(deblank(lin))));
	comind = min([findstr([lin '//'], '//') findstr(lin,'% ')]);
	com = fliplr(deblank(fliplr(lin(comind+2:end))));
	if isempty(com), com = ''; end
	lin = lin(1:comind-1);
	if isempty(lin), continue, end
	if strcmp(lin([1 end]), '[]')
		lin([1 end find(isspace(lin))]) = [];
		section = lin;
		continue
	end
	if isempty(section), err = sprintf('no descriptor section defined: stopped at ''%s''', lin); return, end
	
	typeinfo = {};
	switch lower(section)
		case 'statevectordefinition'
			[key val] = strtok(lin);
		case 'parameterdefinition'
			eqind = min(find(lin=='='));
			if isempty(eqind), err = sprintf('no = sign: stopped at ''%s''', lin); return, end
			val = lin(eqind+1:end);
			lin = deblank(lin(1:eqind-1));
			words = {};
			while ~isempty(lin), [words{end+1} lin] = strtok(lin); end
			if length(words) ~= 3, err = sprintf('error parsing ''%s'': expected 3 words', lin); return, end
			key = words{end};
			typeinfo = words(1:end-1);
		otherwise
			err = sprintf('do not know how to parse section ''%s''', section);
			return
	end
	val = deblank(fliplr(deblank(fliplr(val))));
	if all(ismember(val, '0123456789. -+eE')), val = eval(['[' val ']'], 'val'); end
	if isstr(val), words = {}; while ~isempty(val), [words{end+1},val] = strtok(val); end, val = words; end
	key = [section '.' deblank(fliplr(deblank(fliplr(key))))];
	s = addfield(s, key, val);
	comments = addfield(comments, key, com);
	types = addfield(types, key, typeinfo);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = addfield(s, key, val)
if isempty(s), s = struct; end
eval(['s.' key '=val;'])
