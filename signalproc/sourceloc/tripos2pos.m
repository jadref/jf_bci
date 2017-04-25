function [pos]=tripos2pos(triPos,pnt,tri);
% convert from position specified in a mesh to 3d co-ords
for pi=1:size(triPos,2);
   trii   = triPos(1,pi);
   tripnt = pnt(:,tri(:,trii));
   dtripnt= [tripnt(:,2)-tripnt(:,1) tripnt(:,3)-tripnt(:,1)];
   pos(:,pi)=tripnt(:,1)+dtripnt*triPos(2:3,pi);
end
return;