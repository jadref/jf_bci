function [a]=writeMxInfo(fname,Mx)
% write matrix in mxInfo ascii format
if ( isstr(fname) ) 
   fid=fopen(fname,'w');
   if ( fid<0 ) error(sprintf('Couldnt open file: %s',fname)); end;
elseif ( isnumeric(fname) && fname>0 ) fid=fname;
else error('invalid fid'); 
end

% write type info
if ( isa(Mx,'single') ) fprintf(fid,'s');
elseif ( isa(Mx,'double') ) fprintf(fid,'d');
elseif( isa(Mx,'logical') ) fprintf(fid,'b');
else error('unsupported data type');
end
if ( isreal(Mx) ) fprintf(fid,'r'); else fprintf(fid,'c'); end;
fprintf(fid,'\n');

% write size info
szMx=size(Mx); if(numel(szMx)==2 && szMx(2)==1 ) szMx(2)=[]; end;
fprintf(fid,'['); fprintf(fid,'%dx',szMx(1:end-1)); fprintf(fid,'%d]',szMx(end));
fprintf(fid,'\n');

% write the data itself
fprintf(fid,'[');  fprintf(fid,'%0.20g\t',real(Mx)); fprintf(fid,']');
if ( ~isreal(Mx) ) 
   fprintf(fid,'['); fprintf(fid,'%0.20g\t',imag(Mx));  fprintf(fid,']');
end
fprintf(fid,'\n');

% close the file
if ( isstr(fname) ) fclose(fid); end;
return;

%-----------
function testCase()
X=randn(10,10);
writeMxInfo('X',X)

