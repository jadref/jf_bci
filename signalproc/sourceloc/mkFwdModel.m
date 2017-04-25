function [A,inside,mind2tri,electTriPos,electTriProj]=mkFwdModel(srcPos,electPos,varargin);
% compute a forward model using Thom Ostendorps dipoli fn
%
% [A,inside,mind2tri,electTriPos,electTriProj]=mkFwdModel(srcPos,electPos,varargin)
%
% Inputs:
%  srcPos   -- [3 x nSrc] set of source positions
%  electPos -- [3 x nElect] set of electrode positions (optional)   ([])
% Options:
%  brain    -- [str] filename of the brain triangle mesh   ('./rsrc/brain.tri')
%              [struct .pnt [3 x nP], .tri [3 x nTri]] brain triangle mesh spec
%  brainconductivity -- [float] brain conductivity                          (1)
%  skin     -- [str] filename of the scalp triangle mesh    ('./rsrc/skin.tri')
%              [struct .pnt [3 x nP], .tri [3 x nTri]] brain triangle mesh spec
%  skinconductivity -- [float] brain conductivity                          (1)
%  skull    -- [str] filename of the skull triangle mesh   ('./rsrc/skull.tri')
%              [struct .pnt [3 x nP], .tri [3 x nTri]] brain triangle mesh spec
%  skullconductivity -- [float] brain conductivity                      (.0666)
%  iselectTriPos - [bool] flag if electPos is already in skin.tri coord set (0)
%  srcDepth      - [float] max depth of sources below brain surface       (inf)
% Outputs:
%  A        -- [nElect x nSrcInside x 3] matrix of fwd model strengths, for sources inside the brain
%  inside   -- [nSrc x 1 of bool]  indicator set for sources inside brain (only these have valid fwd model)
%  sd2tri   -- [nSrcInside] matrix of distances to the brain surface
%  electTriPos -- [nElect x 3] matrix of positions of electrodes in the skin tri mesh,
%                    in [triIdx mu lambda] form
[mfiled]=fileparts(mfilename('fullpath'));
opts=struct('insidebrain',true,'isolatedsrc',true,...
            'brain',fullfile(mfiled,'rsrc','brain.tri'),'brainconductivity',1,...
            'skin',fullfile(mfiled,'rsrc','skin.tri'),'skinconductivity',1,...
            'skull',fullfile(mfiled,'rsrc','skull.tri'),'skullconductivity',.066666,...
            'normalise',0,'iselectTriPos',0,'tmpdir','/tmp','verb',1,'srcDepth',inf);
opts=parseOpts(opts,varargin);

if( ischar(opts.skull) ) 
   skullfn=opts.skull; 
   if( exist(fullfile(mfiled,skullfn),'file') ) skullfn=fullfile(mfiled,skullfn); end;
   [skull.pnt,skull.tri]=readTri(skullfn);
else
   skull=opts.skull;
   skullfn=writeTri(fullfile(opts.tmpdir,sprintf('tmpSkull_%d.tri',round(rand(1)*10000))),...
                    skull.pnt,skull.tri);
end;
if( ischar(opts.skin) )  
   skinfn =opts.skin;  
   if( exist(fullfile(mfiled,skinfn),'file') ) skinfn=fullfile(mfiled,skinfn); end;
   [skin.pnt,skin.tri]=readTri(skinfn);   
else
   skin  =opts.skin;
   skinfn=writeTri(fullfile(opts.tmpdir,sprintf('tmpSkin_%d.tri',round(rand(1)*10000))),skin.pnt,skin.tri);
end;
if( ischar(opts.brain) ) 
   brainfn=opts.brain; 
   if( exist(fullfile(mfiled,brainfn),'file') ) brainfn=fullfile(mfiled,brainfn); end;
   [brain.pnt,brain.tri]=readTri(brainfn);
else % write the given tri to a file
   brain  =opts.brain;
   brainfn=writeTri(fullfile(opts.tmpdir,sprintf('tmpBrain_%d.tri',round(rand(1)*10000))),brain.pnt,brain.tri);
end;

if ( opts.insidebrain ) % limit source locs to inside brain
   if( opts.verb>0 ) fprintf('Removing sources outside the brain, or tooo deep...'); end;
   % shrink brain by 1% to ensure all points are strictly inside it!
   brainpnt=brain.pnt; cent=mean(brainpnt,2); brainpnt=repop(repop(brainpnt,'-',cent)*.99,'+',cent);
   % alt way
   [lambda1,lambda2,Pnorm,d2tri]=pntTriDis(srcPos,brainpnt,brain.tri);
   % now convert to signed distance matrix
   sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2); % get the closest tri
   ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii); % direct index
   inside = (Pnorm(ii)>0) & sqrt(mind2tri)<opts.srcDepth; % inside if inside at the closest point & 25mm from surf
   osrcPos=srcPos;
   srcPos(:,inside~=1)=[]; % remove outside brain points
   if( opts.verb>0 ) fprintf('done\n'); end
end

% 2) Move electrodes onto the skin and into tri-coords
if ( ~isempty(electPos) ) % only if given
   if ( ~opts.iselectTriPos )
      [lambda1,lambda2,Pnorm,d2tri]=pntTriDis(electPos,skin.pnt,skin.tri); % get pnts in barycentric coords
      sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2);      % get the closest tri
      ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii);            % get dir index into lambda
      electTriPos = cat(1,mind2Trii',lambda1(ii)',lambda2(ii)');           % tri-pos of the electrodes
   else
      electTriPos =electPos;  % comp 3-d co-ords
   end
   % convert from triPos to projected pos
   electProjPos = repop(skin.pnt(:,skin.tri(2,electTriPos(1,:))),'*',electTriPos(2,:)) ...
       + repop(skin.pnt(:,skin.tri(3,electTriPos(1,:))),'.*',electTriPos(3,:)) ...
       + repop(skin.pnt(:,skin.tri(1,electTriPos(1,:))),'.*',1-electTriPos(2,:)-electTriPos(3,:));
end

% check the fit
%clf;trisurf(skin.tri',skin.pnt(1,:),skin.pnt(2,:),skin.pnt(3,:));hold on;for i=1:size(electPos,2);scatPlot(skin.pnt(:,skin.tri(:,electTriPos(2,i))),'linewidth',5),hold on;scatPlot(electPos(:,i),'r.','markersize',10);end; scatPlot(electProjPos,'k.','markersize',20);

% 3) make the forward model.  3 types of material.
%  ./dipoli skin 1, skull .06666,  %brain 1,  electPos,  sourcePos
% This means
%   skin tri, conductivity=1
%   skull tri, conductivity=.066
%   brain tri, conductivity=1, this is the isolated source interface
%   electPos
%   sourcePos -- all within the brain
% 3.1) write the electrode positions to an ascii file to call dipoli
tmpElectSkinFn = ''; % no given electrodes
if ( ~isempty(electPos) ) % only if electrode positions specified
   tmpElectSkinFn = writeTriPos(fullfile(opts.tmpdir,sprintf('tmpelectskinfn_%d.txt',round(rand(1)*10000))),electTriPos);
end
tmpSrcFn   = writeMx(fullfile(opts.tmpdir,sprintf('tmpsrcfn_%d.txt',round(rand(1)*10000))),srcPos);
tmpFwdMxFn = fullfile(opts.tmpdir,sprintf('tmpFwdMx_%d',round(rand(1)*10000)));
% build the command to call
switch ( lower(computer()) )
 case {'mac','maci'}
  if ( opts.isolatedsrc ) srcFlag='%'; else srcFlag=''; end;
  if(~isempty(tmpElectSkinFn)) tmpElectSkinOpt=sprintf('-e %s',tmpElectSkinFn); else tmpElectSkinOpt=''; end
  dipcmd=sprintf('%s -g %s %g -g %s %g -g %c%s %g %s -s %s -t %s',...
                 fullfile(mfiled,'dipoli','dipoli.macppc'),...
                 skinfn,opts.skinconductivity,...
                 skullfn,opts.skullconductivity,...
                 srcFlag,brainfn,opts.brainconductivity,...
                 tmpElectSkinOpt,tmpSrcFn,tmpFwdMxFn);
 case 'glnx86'
   if ( opts.isolatedsrc ) srcFlag='!'; else srcFlag=''; end;
   dipcmd=sprintf('%s << EOF\n%s\n%g\n%s\n%g\n%c%s\n%g\n\n%s\n%s\n\n%s\nEOF\n',...
                  fullfile(mfiled,'dipoli','dipoli.linux'),...
                  skinfn,opts.skinconductivity,...
                  skullfn,opts.skullconductivity,...
                  srcFlag,brainfn,opts.brainconductivity,...
                  tmpElectSkinFn,tmpSrcFn,tmpFwdMxFn);
 otherwise; 
  error('Unsupported system');
end
% call it
if( opts.verb>0 ) 
   fprintf('\n--------\nCalling dipoli with the command :\n%s\n---------------\n',dipcmd);
end
if ( opts.verb>0 )
   status=system(dipcmd);  % verbose call with debug printing
else
   [status,res]=system(dipcmd);
end

if( status~=0 ) error('dipcmd failed!'); end;

% load back the results
A=readMBFmat(tmpFwdMxFn);
A=A'; A=reshape(A,[size(A,1) 3 size(A,2)/3]); % [nElect x nOri x nSrc] reshape int orientations per source

%4) normalise -- to correct for central sources
% apply leadfield normaliziation by substracting for every virtual voxel,
% mean over all electrodes from every row of electrodes in the leadfield
% No! Normalise away the very low gain of central sources by normalising 
% the power in each compenent
if ( opts.normalise ) A=repop(A,'./',sqrt(msum(A.^2,[1 2]))); end;

% clean up tmp files
delete(tmpElectSkinFn,tmpSrcFn,tmpFwdMxFn);
if( ~ischar(opts.brain) ) delete(brainfn); end
if( ~ischar(opts.skull) ) delete(skullfn); end
if( ~ischar(opts.skin) )  delete(skinfn); end
return;

function fn=writeMx(fn,A);
fid=fopen(fn,'w');
if ( fid<=0 ) error('Couldnt open %s for writing',fn); end
fprintf(fid,'%d\n',size(A,2));
for i=1:size(A,2); 
   fprintf(fid,'%i\t',i); 
   if ( isinteger(A) ) fprintf(fid,'%d\t',A(:,i)); 
   else                fprintf(fid,'%f\t',A(:,i)); 
   end;
   fprintf(fid,'\n');
end
fclose(fid);
return;

function fn=writeTriPos(fn,A);
fid=fopen(fn,'w');
if ( fid<=0 ) error('Couldnt open %s for writing',fn); end
fprintf(fid,'%d\n',size(A,2));
for i=1:size(A,2); 
   if ( size(A,1)==1 ) % idx,triIdx
      fprintf(fid,'%i\t%i',i,A(1,i));
   elseif ( size(A,1)==3 ) % tri,lambdamu
      fprintf(fid,'%i\t',A(1,i));fprintf(fid,'\t%d',A(2:end,i)); 
   else error('bad elect pos spec');
   end;
   fprintf(fid,'\n');
end
fclose(fid);
return;


%--------------------------------------------------------------------------------
function testCase()
% make a 3-sphere model
[spheretri.pnt spheretri.tri] = readTri('rsrc/icosahedron162.tri');
% dipoli has another definition of the direction of the surfaces
spheretri.tri = spheretri.tri(end:-1:1,:);
r = [88 92 100];
c = [1 1/80 1];
brain=spheretri; brain.pnt=brain.pnt*r(1);
skull=spheretri; skull.pnt=skull.pnt*r(2);
skin =spheretri; skin.pnt =skin.pnt*r(3);
[fids.cnames fids.ll fids.xy fids.xyz]=readCapInf('sphere_fids'); fids.xyz=fids.xyz*r(3);
%electTriPos = [1:size(skin.tri,2); ones(2,size(skin.tri,2))];%pos2tripos(skin.pnt,skin.pnt,skin.tri);
%electPos    = skin.pnt;
%srcPos      = [0 0 70;0 0 -70]'; % probe dipole position

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

% compute the fwd model
[A,inside,d2tri]=mkFwdModel(srcPos,[],'brain',brain,'brainconductivity',c(1),...
                            'skull',skull,'skullconductivity',c(2),'skin',skin,'skinconductivity',c(3));
% build an annotated object to hold the data and the meta-info
fwdMx.X = A;
fwdMx.di = mkDimInfo(size(fwdMx.X),'ch',[],[],'ori',[],{'x','y','z'},'ch_src',[],[]);
[fwdMx.di(1).extra.pos3d] = num2csl(skin.pnt);     % 3d pos in mri-space
[fwdMx.di(1).info.tri]    = skin.tri;
fwdMx.di(1).info.fids     = fids.xyz;                 % ref-pos to align with cap-co-ords
[fwdMx.di(3).extra.pos3d] = num2csl(srcPos(:,inside>0)); % 3d pos in mri-space of the srcs
[fwdMx.di(3).extra.d2tri] = num2csl(d2tri(inside>0));    % sqrd dis to brain surface
save('~/source/matfiles/signalproc/sourceloc/rsrc/fwdMx_3sphere','-struct','fwdMx');

% plot this matrix
clf; tri=fwdMx.di(1).info.tri; vert=[fwdMx.di(1).extra.pos3d]; trisurf(tri',vert(1,:),vert(2,:),vert(3,:));
hold on; scatPlot(fwdMx.di(1).info.fids,'g.','markersize',40); 

% visualise the results
clf;
subplot(221);trisurf(skin.tri',skin.pnt(1,:),skin.pnt(2,:),skin.pnt(3,:),A(:,1,1));
subplot(222);trisurf(skin.tri',skin.pnt(1,:),skin.pnt(2,:),skin.pnt(3,:),A(:,2,1));
subplot(223);trisurf(skin.tri',skin.pnt(1,:),skin.pnt(2,:),skin.pnt(3,:),A(:,3,1));

% validate against FT call with the same parameters
lf=load('temp/ftlf'); clf; plot([lf(:,3) A(:,3)]);

%-----------------------------------------------------
% try on "real" meshs + caps
brainfn='rsrc/brain.tri';skullfn='rsrc/skull.tri';skinfn='rsrc/skin.tri';
[mrifids.cnames mrifids.ll mrifids.xy mrifids.xyz]=readCapInf('fids_xyz.txt','../../signalproc/sourceloc/rsrc');

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

% load a cap-file and align it with the MRI coords
capFn='cap64';
[capfids.cnames capfids.ll capfids.xy capfids.xyz]=readCapInf('fids');
[R t]=rigidAlign(capfids.xyz,mrifids.xyz);
[cap.cnames cap.ll cap.xy cap.xyz]=readCapInf(capFn);
electPos = repop(R*cap.xyz,'+',t);
electfids= repop(R*capfids.xyz,'+',t);

% map electrodes to the skin mesh
[skin.pnt,skin.tri]=readTri(skinfn);
[lambda1,lambda2,Pnorm,d2tri]=pntTriDis(electPos,skin.pnt,skin.tri); % get pnts in tri coords
sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2); % get the closest tri
ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii);       % get dir index into lambda
electTriPos = cat(1,mind2Trii',lambda1(ii)',lambda2(ii)');         % tri-pos of the electrodes
% build the interpolation matrix
W = zeros(size(skin.pnt,2),size(electPos,2)); % map from fwdMx -> electPos
W(sub2ind(size(W),skin.tri(2,electTriPos(1,:)),1:size(W,2)))=electTriPos(2,:);
W(sub2ind(size(W),skin.tri(3,electTriPos(1,:)),1:size(W,2)))=electTriPos(3,:);
W(sub2ind(size(W),skin.tri(1,electTriPos(1,:)),1:size(W,2)))=1-electTriPos(2,:)-electTriPos(3,:);
electProjPos = skin.pnt*W; % compute the surface interpolated positions
% electProjPos2 = repop(skin.pnt(:,skin.tri(2,electTriPos(1,:))),'*',electTriPos(2,:)) ...
%     + repop(skin.pnt(:,skin.tri(3,electTriPos(1,:))),'.*',electTriPos(3,:)) ...
%     + repop(skin.pnt(:,skin.tri(1,electTriPos(1,:))),'.*',1-electTriPos(2,:)-electTriPos(3,:));


% visually check the alignment
[skin.pnt skin.tri]=readTri(skinfn);
clf;trisurf(skin.tri',skin.pnt(1,:)',skin.pnt(2,:)',skin.pnt(3,:)');
hold on; scatPlot(electPos,'m.','markersize',20); scatPlot(electProjPos,'k.','markersize',40);

[A,inside,d2tri]=mkFwdModel(srcPos,electTriPos,'brain',brainfn,'skull',skullfn,'skin',skinfn,...
                            'iselectTriPos',1,'srcDepth',inf);
% build an annotated object to hold the data and the meta-info
fwdMx.X = A;
fwdMx.di = mkDimInfo(size(fwdMx.X),'ch',[],cap.cnames,'ori',[],{'x','y','z'},...
                     'ch_src',[],[]);
fwdMx.di(1) = addPosInfo(fwdMx.di(1),capFn);
[fwdMx.di(1).extra.pos3d] = num2csl(electProjPos);     % 3d pos in mri-space
% surface triangulation of the electrode co-ords, N.B. only valid for *convex* surfaces!
% used to map data electrodes onto fwd-soln vertices
[fwdMx.di(1).info.tri]    = convhulln(double(electProjPos)')';  
fwdMx.di(1).info.fids     = electfids;                 % ref-pos to align with cap-co-ords
[fwdMx.di(3).extra.pos3d] = num2csl(srcPos(:,inside>0)); % 3d pos in mri-space of the srcs
[fwdMx.di(3).extra.d2tri] = num2csl(d2tri(inside>0)); % sqrd dis to brain surface
save('~/source/matfiles/signalproc/sourceloc/rsrc/fwdMx_cap64','-struct','fwdMx');

% without any electrode positions
[A,inside,d2tri]=mkFwdModel(srcPos,[],'brain',brainfn,'skull',skullfn,'skin',skinfn,'srcDepth',inf);
% build an annotated object to hold the data and the meta-info
fwdMx.X = A;
fwdMx.di = mkDimInfo(size(fwdMx.X),'ch',[],[],'ori',[],{'x','y','z'},'ch_src',[],[]);
[fwdMx.di(1).extra.pos3d] = num2csl(skin.pnt);     % 3d pos in mri-space
[fwdMx.di(1).info.tri]    = skin.tri;
fwdMx.di(1).info.fids     = mrifids.xyz;                 % ref-pos to align with cap-co-ords
[fwdMx.di(3).extra.pos3d] = num2csl(srcPos(:,inside>0)); % 3d pos in mri-space of the srcs
[fwdMx.di(3).extra.d2tri] = num2csl(d2tri(inside>0));    % sqrd dis to brain surface
save('~/source/matfiles/signalproc/sourceloc/rsrc/fwdMx','-struct','fwdMx');


% visualise the solution
clf; jplot(cap.xy,A(:,:,1))
% point in the center of the brain
[a,ai]=min(sum(repop(srcPos,'-',mean(srcPos,1)).^2));
clf; jplot(cap.xy,A(:,:,ai))

% compare with pre-computed solution
AA=readMBFmat('temp/forward/forwardmat11.txt'); % 256cap fwd Mx

[cap.cnames cap.ll cap.xy cap.xyz]=readCapInf('cap256');
% check vs. others
[electProjPos2]=readMBFPnt('temp/electrodes/projmri.pnt');
[electTriPos2] =readMBFTriPos('temp/electrodes/projmri.pos');

clf;scatPlot(electProjPos,'b.');hold on;scatPlot(electProjPos2,'g.');