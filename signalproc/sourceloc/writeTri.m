function fn=writeTri(fn,pnt,tri)
% write a tri-mesh to ascii file
%
% fn=writeTri(fn,pnt,tri)
if ( size(pnt,1)>3 ) pnt=pnt'; end;
if ( size(tri,1)>3 ) tri=tri'; end;
fid = fopen(fn, 'wt');
if ( fid<=0 ) error('Couldnt open %s for writing',fn); end
% write the vertex points
fprintf(fid, '%d\n', size(pnt,2));
for i=1:size(pnt,2); fprintf(fid,'%d\t%f\t%f\t%f\n',i,pnt(:,i)); end;
% write the triangles
fprintf(fid, '%d\n', size(tri,2));
for i=1:size(tri,2); fprintf(fid,'%d\t%d\t%d\t%d\n',i,tri(:,i)); end;
fclose(fid);
return;
