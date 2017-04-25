function x = bci2000param(s, fn)
% X = BCI2000PARAM(S, FN)
% 
% Given structure S (the output of TTDREAD), extract the parameter(s) denoted
% by FN.
% 
% If FN is a string, output X is the raw parameter value. If FN is a cell array
% of strings, X is a structure containing all the requested parameters in the
% appropriately named fields. If FN is empty or omitted, all available
% parameters are extracted to a structure. Hence,
% 
% X = BCI2000PARAM(S)
% 
% returns all parameters.
% 
% See also: TTDREAD, TTDSTATE, TTDEXTRACT
% 
% 2004-11-09 Jez Hill

if nargin < 2, fn = {}; end

if ~isstruct(s), s = ttdread(s, 0); end
if ~isstruct(s), return, end

namesin = fieldnames(s.descriptors.ParameterDefinition);
valsin = struct2cell(s.descriptors.ParameterDefinition);
types = struct2cell(s.types.ParameterDefinition);
for i = 1:length(types), types{i} = types{i}{2}; end

if isempty(fn), fn = namesin; end
wascell = iscell(fn);
if ~wascell, fn = {fn}; end
fn = fn(:);
for i = 1:length(fn)
	ni = min(find(strcmp(cellstr(lower(char(namesin))), lower(fn{i}))));
	if isempty(ni), error(sprintf('parameter ''%s'' not found', fn{i})), end
	ind(i) = ni;
end
valsin = valsin(ind);
types = types(ind);

vals = cell(size(fn));
st = []; prevbyte = [];
mult = []; prevnbits = [];
for i = 1:length(fn)
	x = valsin{i};
	t = types{i};
	if iscell(x)
		x = x{1};
	elseif isnumeric(x)
		if strncmp(lower(fliplr(t)), 'tsil', 4)
			len = x(1);
			x = x([1:len]+1);
		elseif strcmp(lower(t), 'matrix')
			siz = x(1:2);
			x = reshape(x([1:prod(siz)]+2), siz);  % TODO: confirm with Thilo: rows then columns in TTD file ??
		elseif( ~isempty(x) )
			x = x(1);
      end
	else
		error(sprintf('don''t know how to handle value of %s', fn{i}))
	end	
	vals{i} = {x};
end

if wascell
	x = [fn vals]';
	x = struct(x{:});
else
	x = vals{1}{1};
end
