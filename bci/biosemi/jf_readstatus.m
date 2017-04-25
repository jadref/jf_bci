function [status,epoch,cmrange,battery]=decodestatus(status)
% decode status channel into its parts
statusinf=bitand(uint32(statusinf+2^(24-1)),2^24-1); % +24 bit number
status   =bitand(statusinf,2^16-1);% actual status info in low-order 16 bits
epoch    =int8(bitget(statusinf,16+1));
cmrange  =int8(bitget(statusinf,20+1));
battery  =int8(bitget(statusinf,22+1));
