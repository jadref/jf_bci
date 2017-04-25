function [A]=readFwdMx(fn,asciip);
% load the produced system matrix
if ( nargin < 2 || isempty(asciip) ) asciip=0; end;
if ( isequal(fn(max(1,end-4):end),'.txt') || asciip ) % ascii format
   fid=fopen(fn,'r');
   [sz]=fscanf(fid,'%d',2);
%    c=fscanf(fid,'%c',1); 
%    if ( c==10 ) sz(2)=1; else fseek(fid,-1,0); [sz(2)]=fscanf(fid,'%d',1); end;
   A=fscanf(fid,'%f',sz(:)');
   fclose(fid);
else % binary format
   fid=fopen(fn,'rb','ieee-le');
   magic=fread(fid,[1 8],'int8=>char');
if( strcmp(magic,';;mbfmat') )
   % new binary format
   biged=fread(fid,1,'uint8');
   if( biged ) 
      fclose(fid); fid=fopen(fn,'rb','ieee-be'); fseek(fid,9,0); 
   end;
   hsize=fread(fid,1,'uint32');
   dsgn =fread(fid,1,'uint8');
   dtype=fread(fid,1,'uint8');
   dsz  =fread(fid,1,'uint8');
elseif( ~isempty(strmatch(';;mbf',magic)) )
   error('Only for matrix formats');
else
   fseek(fid,0,-1); % reset to start
   dsz = 4; 
end
[sz]=fread(fid,2,'int32');
A=fread(fid,[sz(2) sz(1)],sprintf('real*%d',dsz));
end
fclose(fid);
return;
