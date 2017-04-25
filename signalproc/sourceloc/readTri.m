function [pnt,tri]=readTri(fn)
fid = fopen(fn, 'r');
if ( fid<=0 ) error('Couldnt open %s for reading',fn); end
% read the vertex points
npnt=fscanf(fid, '%d\n',1);
pnt=fscanf(fid,'%d\t%f\t%f\t%f\n',[4,npnt]);
pnt=pnt(2:end,:); % convert to 3 x N
% read the triangles
ntri=fscanf(fid, '%d\n',1);
[tri]=fscanf(fid,'%d\t%d\t%d\t%d\n',[4,ntri]);
tri=tri(2:end,:); % convert to 3 x N
fclose(fid);
return;
