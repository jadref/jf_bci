function [i,m,b]=float2int(f,nbits,normalise,center)
% [i,m,b]=float2int(f,nbits,normalise,center)
% Function to map float data to integer perserving as much precision as
% possible.  Converted using:
%   i = (f-b)*m
% Inputs:
%  f         -- float matrix to convert
%  nbits     -- number of bits in the output integer, one of {8,16,32,64} (16)
%  normalise -- re-scale input to use the full range of the output int,
%               normalise < 0  => use (data range*-normalise) to re-scale
%               0              => don't normalise
%               normalise > 0  => just use normalise to re-scale
%  center    -- center the data to use the full integer output range? (false)
% Outputs
%  i         -- converted integer
%  m         -- scaling factor used. (use float(i)/m+b to invert conversion)
%  b         -- centering used
if ( nargin<2 || isempty(nbits) ) nbits=16; end;
if ( nargin<3 || isempty(normalise) ) normalise=0; end;
if ( nargin<4 || isempty(center) ) center=0; end;
if ( center )    b=mean([max(f(:)),min(f(:))]); else b=0; end;
if ( normalise==0 )      m = 1;
elseif ( normalise < 0 ) m = -normalise*2.^(nbits-1) / max(abs(f(:)-b)); 
else                     m = normalise*2.^(nbits-1); % just scaling factor
end;
i = zeros(size(f),sprintf('int%d',nbits));
i(:) = (f(:)-b) .* m;
return

%--------------------------------------------------------------------
function testcase()
f=randn(1000,1);
f=sin(0:.1:10*pi*2);
[i8 ,m8 ]=float2int(f,8, -1);
[i16,m16]=float2int(f,16,-1);
[i32,m32]=float2int(f,32,-1);
clf;plot(f,'LineWidth',1);hold on;
range=max(abs(f(:))); b=range*.01;
plot(single(i8)/m8+b,'r');plot(single(i16)/m16+b*2,'g');plot(single(i32)/m32+b*3,'c');
