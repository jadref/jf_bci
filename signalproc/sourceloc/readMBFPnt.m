function [pnt]=readMBFPos(fn)
fid = fopen(fn, 'r');
if ( fid<=0 ) error('Couldnt open %s for reading',fn); end
% read the vertex points
npnt=fscanf(fid, '%d\n',1);
pnt=fscanf(fid,'%d\t%f\t%f\t%f\n',[4,npnt]);
pnt=pnt(2:end,:); % convert to 3 x N
return;
