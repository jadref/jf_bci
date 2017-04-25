function [lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,V,tri)
% compute the barycentric co-ords and signed distance between a set of points and a set of triangles
%
% [lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,V,tri)
%
% Maps between 3-d positions an 3-d positions relative to each input triangle specified in
% barycentric co-ords.  Tri spec is:
%                   V_2
%               E_02/ \E_12
%                  /   \
%               V_0-----V_1
%                   E_01
% where, V_{0,1,2} are the vertices, and E_{i,j} the edge *from* vertex i *to* vertex j
% N.B. the positive direction *out-of* the triangles plane is the one where E_01 and E_02
%  for a RHS co-ord system, i.e. where the vertices are spec in *anti-clockwise* order.
%
% Inputs:
%  P -- [3 x nPnt] matrix of test-point positions
%  V -- [3 x nVert] matrix of triangle-mesh vertex positions
%  tri -- [3 x nTri] matrix of indicies into the vertex array V which specify 
%          a triangle.  When specified in *anti-clockwise* order, negative (-ve) 
%          perpendicular distances represent points *behind* the triangle.
%          N.B. if spec in clockwise order then sign of Pnorm is wrong
% Outputs:
%  lambda1 -- [nPnt x nTri] barycentric co-ord for V1, or edge E01
%  lambda2 -- [nPnt x nTri] barycentric co-ord for V2 or edge E02
%  Pnorm   -- [nPnt x nTri] signed perpendicular distance to the triangle.  The sign 
%              indicates if the point is *in-front* or *behind* the triangle.  Where it is 
%              in-front if looking from the point the triangles vertices are in 
%              *anti-clockwise* order, or equivalently if the edges, E_01, E_02 and E_0P form a 
%              right-hand-coordinate system.
%  d2tri   -- [nPnt x nTri] *squared* *planer* distance to the nearest point on the triangle.
%               i.e. squared dist to tri as measured only in the plane of the tri
%             N.B. to get true dis to the tri us: sqrt(d2tri+Pnorm.^2)
nPnt = size(P,2); nVert=size(V,2); nTri=size(tri,2); 
tri = int32(tri); 
P=single(P); V=single(V); % convert to singles to save RAM

% compute barycentric co-ords of the point in the triangle
% math lifted from: http://www.gamedev.net/community/forums/topic.asp?topic_id=481835
V0      = V(:,tri(1,:));    % [3 x nTri]
E01     = V(:,tri(2,:))-V0; % edge from V0->V1 [3 x nTri]
E02     = V(:,tri(3,:))-V0; % edge from V0->V2 [3 x nTri]
E01E02  = tprod(E01,[-1 1],E02,[-1 1],'n'); % i.p. btw. E01 and E02 [nTri]
d2E01   = tprod(E01,[-1 1],[],[-1 1],'n');  % sqrd len E01 [nTri]
d2E02   = tprod(E02,[-1 1],[],[-1 1],'n');  % sqrd len E02 [nTri]
s       = d2E01.*d2E02 - E01E02.*E01E02;    % scale factor = 2*area [nTri]
pinvA1  = repop((d2E02./s)','*',E01)-repop((E01E02./s)','*',E02); % pseudo-inv for LS soln [3 x nTri]
pinvA2  = repop((d2E01./s)','*',E02)-repop((E01E02./s)','*',E01); % pseudo-inv for LS soln [3 x nTri]
pinvA10 = tprod(V0,[-1 1],pinvA1,[-1 1],'n'); % mapping back to V0 contribution [nTri]
pinvA20 = tprod(V0,[-1 1],pinvA2,[-1 1],'n'); % mapping back to V0 contribution [nTri]
lambda1 = repop(tprod(P,[-1 1],pinvA1,[-1 2],'n'),'-',pinvA10'); % [nPnt x nTri]
lambda2 = repop(tprod(P,[-1 1],pinvA2,[-1 2],'n'),'-',pinvA20'); % [nPnt x nTri]

% check the barycentric co-ords, with different orgin locations
% lambda0= 1-lambda1-lambda2;
% E12    = V(:,tri(3,:))-V(:,tri(2,:));
% PP     = repop(V(:,tri(1,:)),'+',repop(E01,'/',s')*lambda1' + repop(E02,'/',s')*lambda2'); mad(P,PP)
% PP     = repop(V(:,tri(2,:)),'+',repop(-E01,'/',s')*lambda0' + repop(E12,'/',s')*lambda2'); mad(P,PP)
% PP     = repop(V(:,tri(3,:)),'+',repop(-E02,'/',s')*lambda0' + repop(-E12,'/',s')*lambda1'); mad(P,PP)
% mad(P,PP)

% Distance, *in the triangles plane*, to the closest point on the triangle
if ( nargout > 3) % only if wanted
d2tri  = zeros(nPnt,nTri,'single'); % assume inside, i.e. dis 0

% Points below size 01
dE01   = sqrt(d2E01);
d2E01n = (d2E02-E01E02.^2./d2E01);    % squared length of proj of E02 onto E01's normal vector
tmp    = lambda1 + repop(lambda2,'*',(E01E02./d2E01)'); % lambda1 inc part lambda2 in this dir [nPnt x nTri]
tmp    = tmp - max(0,min(1,tmp)); % squeeze out the 0-1 (i.e. triangle) portion of the co-ord system
tmp    = repop(tmp.^2,'*',d2E01');
tmp    = tmp + repop(lambda2.^2,'*',d2E01n');    % planer pnt-tri dist
ind    = ( lambda2 < 0 ); % set of points for which this dis is the minimum
d2tri(ind) = tmp(ind); % store
clear tmp ind;

% Points above side 02
dE02   = sqrt(d2E02);
d2E02n = (d2E01-E01E02.^2./d2E02); % proj of E01 onto E02's normal vector
tmp    = lambda2 + repop(lambda1,'*',(E01E02./d2E02)');% lambada01 inc part of lamba02 in this dir [nPnt x nTri]
tmp    = tmp - max(0,min(1,tmp));
tmp    = repop(tmp.^2,'*',d2E02'); % 2 steps to save ram
tmp    = tmp + repop(lambda1.^2,'*',d2E02n'); % planer pnt-tri dist
ind    = ( lambda1 < 0 ); % points on this side
d2tri(ind) = tmp(ind);
clear tmp ind;

% Points outside side 12
d2E12  = d2E01 - 2*E01E02 + d2E02;   % length of side E12 [nTri]
dE12   = sqrt(d2E12);
E10E12 = (-E01E02 + d2E01 );         % proj of E10 onto E12 [nTri]
d2E10n = (d2E01-E10E12.^2./d2E12);   % proj of E10 onto E12's normal vector
lambda0= 1-lambda1-lambda2;          % the other lambda
tmp    = lambda2 + repop(lambda0,'*',(E10E12./d2E12)');%lambada2 inc part of lamba0 in this dir [nPnt x nTri]
tmp    = tmp - max(0,min(1,tmp));% lambda2 with interval between 0 and 1 removed
tmp    = repop(tmp.^2,'*',d2E12'); % split the computation to save ram
tmp    = tmp + repop(lambda0.^2,'*',d2E10n'); % dist to P perpendicular to E12 
ind    = ( lambda0 < 0 ); % set points on this side
d2tri(ind)=tmp(ind); 
clear tmp ind

end

% Perpendicular distance to the triangle
if ( nargout > 2) % only if wanted
norm   =repop(cross(E01,E02),'./',sqrt(s'));    % unit normal to the triangle [3 x nTri]
Pnorm  =repop(P'*norm,'-',tprod(V0,[-1 2],norm,[-1 2],'n')); % pnt-perpendicular distance [nPnt x nTri]
end

return;

%---------------------------------------------------------------------------
function testCases()
% simple tests
V0=[0 0 0;1 0 0;0 1 0]'; V=V0;
tri0=[1 2 3]';           tri=tri0;
P=[.2 .2 0; 0 0 0; .5 0 0; 1 0 0; .5 .5 0; 0 1 0; 0 .5 0;... % pts inside + on edges
   -1 0 0; -1 -1 0; 0 -1 0; .5 -1 0; 1 -1 0; 1.5 0 0; 1 1 0; 0 1.5 0; -1 1 0; 1 .5 0]'; % outside 
Dp=[zeros(1,size(P,2))]';
D2t=[0 0 0 0 0 ...
     0 0 1 2 1 ...
     1 1 .25 .5 .25 ...
     1 0.125]';
clf;scatPlot(V0(:,[tri0;tri0(1,:)]));hold on; scatPlot(P,'r.');text(P(1,:)',P(2,:)',P(3,:),reshape(sprintf('%02d',1:size(P,2)),2,size(P,2))');
[lambda10,lambda20,Pnorm0,d2tri]=pntTriDis(P,V,tri);[single(R0) D0]
[plambda1,plambda2,pPnorm,pd2tri]=pntTriDis([P(1:2,:);ones(1,size(P,2))],V,tri);[single(pd2tri) D2t]
[nlambda1,nlambda2,nPnorm1,nd2tri]=pntTriDis([P(1:2,:);-ones(1,size(P,2))],V,tri);[single(nd2tri) D2t]
[lambda12,lambda22,Pnorm2,d2tri2]=pntTriDis(P*2,V*2,tri);[d2tri2 D2t]

% permute edge order
V=[1 0 0;0 1 0;0 0 0]'; tri=[1 2 3]'; % vertex order only
[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,V,tri);
[D2t d2tri]

V=[0 0 0;1 0 0;0 1 0]'; tri=[2 3 1]'; % edge order only
[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,V,tri);
[D2t d2tri]
% shift the triangles
[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,[V([1:2],:);ones(1,size(V,2))],tri);
[D2t d2tri]


% multiple triangles
V=[0 0 0;1 0 0;0 1 0; 1 1 0]'; % square
tri=[1 2 3;2 4 3]';
D2t=[0 0.18;0 0.5;0 0.125;0 0;0    0;0    0;0  0.125;1    2;2   4.5;...
     1    2;1  1.25;1    1;0.25  0.25;0.5    0;0.25  0.25;1    1;0.125    0];
[lambda11,lambda21,Pnorm1,d2tri1]=pntTriDis(P,V,tri(:,1));
[lambda12,lambda22,Pnorm2,d2tri2]=pntTriDis(P,V,tri(:,2));
[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(P,V,tri);
[d2tri1 d2tri(:,1) d2tri2 d2tri(:,2)]

% full mesh and point cloud
skinfn='../signalproc/sourceloc/rsrc/skin.tri';
% load a cap-file and align it with the MRI coords
[mrifids.cnames mrifids.ll mrifids.xy mrifids.xyz]=readCapInf('fids_xyz.txt','../signalproc/sourceloc/rsrc');
[capfids.cnames capfids.ll capfids.xy capfids.xyz]=readCapInf('fids');
[R t]=rigidAlign(capfids.xyz,mrifids.xyz);
[cap.cnames cap.ll cap.xy cap.xyz]=readCapInf('cap64');
electPos = repop(R*cap.xyz,'+',t);
% map to the skin mesh
[skin.pnt,skin.tri]=readTri(skinfn);

% visually check the alignment
clf;trisurf(skin.tri',skin.pnt(1,:)',skin.pnt(2,:)',skin.pnt(3,:)');
hold on; scatPlot(electPos,'k.','markersize',20);

% map the electrodes to the surface
tic,[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(electPos,skin.pnt,skin.tri);toc
% find the closest point
sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2); % get the closest tri
ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii);
% N.B. the barycentric co-ords can be dir used for interpolation!
projPos = repop(skin.pnt(:,skin.tri(2,mind2Trii)),'*',lambda1(ii)') ...
          + repop(skin.pnt(:,skin.tri(3,mind2Trii)),'.*',lambda2(ii)') ...
          + repop(skin.pnt(:,skin.tri(1,mind2Trii)),'.*',1-lambda2(ii)'-lambda1(ii)');
% plot the distance map
eli=1:size(electPos,2); 15;
clf;trisurf(skin.tri',skin.pnt(1,:)',skin.pnt(2,:)',skin.pnt(3,:)',sqrt(abs(sd2tri(min(eli),:))).*sign(Pnorm(min(eli),:)));
hold on; scatPlot(electPos(:,eli),'k.','markersize',20); 
scatPlot(projPos(:,eli),'g.','markersize',20);


% try with a *lot* of points
brainfn='../signalproc/sourceloc/rsrc/brain.tri';
% compute source locations
[brain.pnt,brain.tri]=readTri(brainfn);
rng(1,1)=min(brain.pnt(1,:));rng(1,2)=max(brain.pnt(1,:));
rng(2,1)=min(brain.pnt(2,:));rng(2,2)=max(brain.pnt(2,:));
rng(3,1)=min(brain.pnt(3,:));rng(3,2)=max(brain.pnt(3,:));

gridSpc=[11 11 11]; % 11mm spacing
clear srcPos
[srcPos(:,:,:,1),srcPos(:,:,:,2),srcPos(:,:,:,3)]=...
  meshgrid(rng(1,1):gridSpc(1):rng(1,2),...
   rng(2,1):gridSpc(2):rng(2,2),...
   rng(2,1):gridSpc(2):rng(2,2));
srcPos=reshape(srcPos,[size(srcPos,1)*size(srcPos,2)*size(srcPos,3) size(srcPos,4)])';
tic,[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(srcPos,brain.pnt,brain.tri);toc
% now convert to signed distance matrix
sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2); % get the closest tri
ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii);
inside = (Pnorm(ii)>0); % inside if inside at the closest point!

% N.B. the barycentric co-ords can be dir used for interpolation!
projPos = repop(brain.pnt(:,brain.tri(2,mind2Trii)),'*',lambda1(ii)') ...
          + repop(brain.pnt(:,brain.tri(3,mind2Trii)),'.*',lambda2(ii)') ...
          + repop(brain.pnt(:,brain.tri(1,mind2Trii)),'.*',1-lambda2(ii)'-lambda1(ii)');

% plot the distance map
eli=1:size(srcPos,2); 
trii=1:size(brain.tri,2);
clf;trisurf(brain.tri(:,trii)',brain.pnt(1,:)',brain.pnt(2,:)',brain.pnt(3,:)',sqrt(abs(sd2tri(min(eli),trii))));
hold on; scatPlot(srcPos(:,eli),'k.','markersize',20); 
scatPlot(projPos(:,eli),'g.','markersize',20);

eli=3792; trii=mind2Trii(eli);
clf;plot(sqrt(sd2tri(eli,:))); hold on; plot(Pnorm(eli,:),'g'); plot(sqrt(d2tri(eli,:)),'r');

% get sub-set tri's within X of point eli
trii=find(sqrt(abs(sd2tri(eli,:)))<120);
hold on; 
trisurf(brain.tri(:,mind2Trii(eli))',brain.pnt(1,:)',brain.pnt(2,:)',brain.pnt(3,:)');%'linewidth',5)

