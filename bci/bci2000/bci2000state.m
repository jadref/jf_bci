function x = bci2000state(s, fn)
% X = BCI2000STATE(S, FN)
% 
% Given structure S (the output of TTDREAD), extract the state(s) denoted by FN
% 
% If FN is a string, output X is a numeric vector of state values. If FN is a
% cell array of strings, X is a structure containing all the requested states
% in the appropriately named fields. If FN is empty or omitted, all available
% states are extracted to a structure. Hence,
% 
% X = BCI2000STATE(S)
% 
% returns all states.
% 
% See also: TTDREAD, TTDPARAM, TTDEXTRACT
% 
% 2004-11-09 Jez Hill

if nargin < 2, fn = {}; end

namesin = fieldnames(s.descriptors.StateVectorDefinition);
bitaddr = struct2cell(s.descriptors.StateVectorDefinition);

if isempty(fn), fn = namesin; end
wascell = iscell(fn);
if ~wascell, fn = {fn}; end
fn = fn(:);
for i = 1:length(fn)
	ni = min(find(strcmp(cellstr(lower(char(namesin))), lower(fn{i}))));
	if isempty(ni), error(sprintf('state ''%s'' not found', fn{i})), end
	ind(i) = ni;
end
bitaddr = bitaddr(ind);

vals = cell(size(fn));
st = []; prevbyte = [];
mult = []; prevnbits = [];
for i = 1:length(fn)
	nbits = bitaddr{i}(1);
	startbyte = bitaddr{i}(3);
	startbit = bitaddr{i}(4);
	

%%%%%%%%%%%%%%%%%%%	
	
	bit = startbit + [0:nbits-1]';
	byte = startbyte + floor(bit/8);
	bit = rem(bit, 8);
	x = uint8(2 .^ bit); % other endianness of bit addresses within bytes would have been x = uint8(2.^(8-bit));
	if ~isequal(byte, prevbyte)
		st = s.state(byte+1, :);
%		st = double(st); % uncomment if bitand and ~= methods do not exist for uint8
		prevbyte = byte;
	end
	if nbits > 1, x = repmat(x, 1, size(st, 2)); end
	x = bitand(x, st);
	x = (x~=0);

	if nbits > 1
		if ~isequal(nbits, prevnbits)
			mult = 2.^[0:nbits-1]'; % other endianness of state field within given bitfield would have been mult = 2.^[nbits-1:-1:0]';
			mult = repmat(mult, 1, size(st, 2));
			prevnbits = nbits;
		end	
		x = single(x);
		x = x.*mult;
		x = sum(x, 1);
		
%%%%%%%%%%%%%%%%%%%	

		if nbits <= 8
			x = uint8(x);
		elseif nbits <= 16
			x = uint16(x);
		elseif nbits <= 32
			x = uint32(x);
		end
	end
	
	vals{i} = x;
end

if wascell
	x = [fn vals]';
	x = struct(x{:});
else
	x = vals{1};
end
