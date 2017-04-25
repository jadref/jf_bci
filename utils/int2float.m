function [f]=int2float(i,m,b,nbits)
if ( nargin < 2 || isempty(m) ) m=1; end;
if ( nargin < 3 || isempty(b) ) b=0; end;
if ( nargin < 4 || isempty(nbits) ) nbits=64; end;
switch nbits;
 case 32;
  f=single(i)./m+b; 
 case 64;
  f=double(i)./m+b;
 otherwise
  error('Illegal bit size for float, must be 32/64');
end
return;
%--------------------------------------------------------------------
function testcase()
f=randn(1000,1);
f=sin(0:.1:10*pi*2);
[i8 ,m8 ]=float2int(f,8, -1);
[i16,m16]=float2int(f,16,-1);
[i32,m32]=float2int(f,32,-1);
clf;plot(f,'LineWidth',1);hold on;
range=max(abs(f(:))); b=range*.01;
plot(int2float(i8,m8)+b,'r');plot(int2float(i16,m16)+b*2,'g');plot(int2float(i32,m32)+b*3,'c');
