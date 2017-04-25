function [status,epoch,cmrange,battery]=decodestatus(statusinf)
% find indices of negative numbers
bit24i = find(statusinf < 0 | statusinf>0);
% make number positive and preserve bits 0-22
statusinf(bit24i) = bitcmp(abs(statusinf(bit24i))-1,32);
% apparently 24 bits reside in 3 higher bytes, shift right 1 byte
statusinf(bit24i) = bitshift(statusinf(bit24i),-8);
% re-insert the sign bit on its original location, i.e. bit24
statusinf(bit24i) = statusinf(bit24i)+(2^(24-1));

% typecast the data to ensure that the status channel is represented in 32 bits
statusinf = uint32(statusinf);

% decode status channel into its parts
statusinf=bitand(uint32(statusinf+2^(24-1)),2^24-1); % +24 bit number
status   =bitand(statusinf,2^16-1);% actual status info in low-order 16 bits
epoch    =int8(bitget(statusinf,16+1));
cmrange  =int8(bitget(statusinf,20+1));
battery  =int8(bitget(statusinf,22+1));
