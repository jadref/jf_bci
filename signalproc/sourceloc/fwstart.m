clear all
close all
clc


% START MAKING OF THE FORWARD MODEL
%
% This code creates the forward model from the segmented surfaces. 
%
% How to make a forward model?
% *Change the configurations as desired (most important are: subject, measured electrode
% positions(elfile))
% *align electrode positions to mri using first part of the code
% *if there is a wrong measurement, fill in the +/- coordinates in the
% repair code and run this part of the code. ?? Moet automatisch kunnen
% *run the rest of the code. NOTE: some parts are not yet translated into
% Matlab, therefore the instructions about Thoms programs should be used as
% mentioned in the code


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%VARIABLE CONFIGURATIONS   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subject='standard';
%test=['test/' subject];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER  CONFIGURATIONS    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paths
% STARTPATH
startpath=['/Volumes/']; %mac
codepath=['/Users/Bram/BCI Code/'];                %svn check-out place
temppath=['/Users/Bram/temp/fwstart/'];

% paths
addpath         (genpath([codepath 'BCI code/toolboxes/utilities/file_io']));
addpath         (genpath([codepath 'BCI code/external_toolboxes/fieldtrip/']));
addpath         (genpath([codepath 'BCI code/external_toolboxes/tcp_udp_ip/']));
addpath         (genpath([codepath 'BCI code/toolboxes/plotting/']));
addpath         (genpath([codepath 'BCI code/external_toolboxes/spm2/']));
addpath         (genpath([codepath 'BCI code/toolboxes/utilities/electrode_positions/conversion/'])); %oa conversion to xyz
addpath         (genpath([codepath 'BCI code/toolboxes/utilities/electrode_positions/read_translate/']));

%variables
voxelsize=11;
%[FileName,PathName,FilterIndex] = uigetfile('*.txt',['elcfile' subject],[startpath 'BCI_data/own_experiments/']) %dit mag nog wat algemener, uiteindelijk komt alles in __concept new structure en dan staat daar per experiment een load aan caps/leadfields
elcfile     =   ([startpath 'BCI_Data/own_experiments/auditory/sequential_selective_attention/subjective_rhythm/Subjective_Rhythm/Biosemi/Caps/weet niet/SR_stan_070207_??_placement.txt']);
%elcfile     =   ([startpath 'BCI_Data/00 Admin/Caps/Denise 190606 placement.txt']);
%elcfile= [PathName FileName];
elcstfile=[startpath 'BCI_Data/equipment_data/cap_layout/biosemi/positions/cap256.txt'] %if we use a 256 electrodes cap; standard electrode file

%labelfile  =   ([startpath 'BCI_Data/equipment_data/cap_layout/biosemi/labels/labels256.mat']); %makkelijker om uit elcstfile te halen/
nfile=([startpath 'BCI_Data/equipment_data/cap_layout/biosemi/neighbours/neighbors256.mat']);

segdir     =   ([startpath 'BCI_Data/subject_data/' subject '/segmentation/']); 
centerfile     =   ([segdir 'analysis/center.mat']); 

fwdir      =   ([startpath 'BCI_Data/subject_data/forwardmodels/' subject '/']);
mri=myload([segdir 'analysis/brain.mat']);
s=size(mri)
clear mri;


%Little correction to make some maps if they ain't there yet.
a = exist([fwdir 'electrodes/']);
b = exist([fwdir 'forward/']);
c = exist([fwdir 'grid/']);
d = exist([temppath]);


while a == 0
    mkdir([fwdir 'electrodes/']);
    a = exist([fwdir 'electrodes/']);
end
while b == 0
    mkdir([fwdir 'forward/']);
    b = exist([fwdir 'forward/']);
end
while c == 0
    mkdir([fwdir 'grid/']);
    c = exist([fwdir 'grid/']);
end
while d == 0
    mkdir([temppath]); %'/temp/']);
    d = exist([temppath]);% '/temp/']);
end


%
c = exist([temppath 'pnttridist']);
d = exist([temppath 'dipoli']);
e = exist([temppath 'mat2asci']);

if c == 0
    disp 'goin to copy c'
    copyfile([codepath 'BCI code/toolboxes/utilities/file_io/thom_io/pnttridist'],[temppath],'f');
end
if d == 0
    disp 'goin to copy d'
    copyfile([codepath 'BCI code/toolboxes/utilities/file_io/thom_io/dipoli'],[temppath],'f');
end
if e == 0
    disp 'goin to copy e'
    copyfile([codepath 'BCI code/toolboxes/utilities/file_io/thom_io/mat2asci'],[temppath],'f');
end

%find bad electrodes and transform electrodes to mri coordinates
[nmr] = findbadelecs(segdir, fwdir, elcfile, elcstfile, nfile); % some tweeking can be done with the values
%what electrodes are used? the projected is better for the ratio_calc?

bad=[];
if nmr~=0
    warning('bad electrodes found!')
    bad=myload([fwdir 'badelecs/badelectrodes.mat'])
end



%USING QTRIPLOT to view electrodes projected on head
vizelecs(segdir,fwdir,bad); %extra option to display the labels of the electrodes
% no point drawing (time consuming & same point are unmarked the second time you call them); but via elecs...



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAKE SENSOR FIELDTRIP FILE  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sens=[];
[cap_cell,cap_struct] = readCap(elcstfile);
sens.label = [cap_cell(:,1)]; %sens.label=myload(labelfile)';
f=fopen([fwdir 'electrodes/projmri.pnt']);
pnt=[];
N=fscanf(f,'%i',1);
for(ii=1:N)
    fscanf(f,'%i',1);
    p=fscanf(f,'%f',3);
    pnt(ii,:)=p;
end
fclose(f);
sens.pnt=pnt;
mysave([fwdir 'electrodes/elec.mat'],sens);

%save a layout structure (for plotting in fieldtrip); 2D projection of head
X=pnt(:,1);
Y=pnt(:,2);
chNum=1:length(X);
Lbl=cap_cell(:,1);
Width(1:length(X),1)=0.12;
Height(1:length(X),1)=0.1;
ToWright=[chNum', X, Y, Width, Height];
name=[fwdir 'electrodes/layout' subject '.lay'];
file=fopen(name,'wt');
for(ii=1:size(ToWright,1))
    fprintf(file,'%6.1f %6.4f %6.4f %6.2f %6.2f %s \n' , ToWright(ii,:), Lbl{ii}); %d?
end
fclose(file);

clear name X Y chNum Lbl Width Height ToWright

%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE VOXEL MATRIX  %
%%%%%%%%%%%%%%%%%%%%%%%%
brain                   =   myload([segdir 'analysis/brain.mat']);
dimgrid=floor(size(brain)/voxelsize);
ind=1:dimgrid(1)*dimgrid(2)*dimgrid(3);
[X,Y,Z]=ind2sub(dimgrid,ind);
grid=[X' Y' Z']*voxelsize;
ind=sub2ind(size(brain),grid(:,1),grid(:,2),grid(:,3));
grid=grid(find(brain(ind)),:);
name=([fwdir 'grid/griduncorrected' num2str(voxelsize) '.txt']);
file=fopen(name,'w');
fprintf(file,'%i\n' , size(grid,1));
for(ii=1:size(grid,1))
    fprintf(file,'%i %i %i %i \n' , ii,grid(ii,:));
end
fclose(file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE DISTANCES GRID-BRAIN  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -copy fwdir/grid/griduncorrected N .txt to folder containing
% programs Thom Oostendorp (temp)
% -also copy segdir/results/brain.tri to this folder
% -run terminal
% -type cd .. two times
% -type "/programThom/pnttridist /programThom/temp/brain.tri /programThom/temp/griduncorrected N .txt>/programThom/temp/dist N .txt"
% -copy programThom/temp/dist N.txt to fwdir/grid/

copyfile(name,temppath,'f');
copyfile([segdir '/results/brain.tri'], temppath,'f');
eval(['!' temppath '/pnttridist ' temppath '/brain.tri ' temppath '/griduncorrected' num2str(voxelsize) '.txt >' temppath '/dist' num2str(voxelsize) '.txt']);
copyfile([temppath '/dist' num2str(voxelsize) '.txt'], [fwdir 'grid/'],'f');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REMOVE GRIDPOINTS THAT ARE OUTSIDE BRAIN  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file=([fwdir 'grid/dist' num2str(voxelsize) '.txt']);
out=readOutOfVolume(file);
grid(out,:)=[];
mysave([fwdir 'grid/grid' num2str(voxelsize) '.mat'],grid);
name=([fwdir 'grid/grid' num2str(voxelsize) '.txt']);
file=fopen(name,'w');
fprintf(file,'%i\n' , size(grid,1));
for(ii=1:size(grid,1))
    fprintf(file,'%i %i %i %i \n' , ii,grid(ii,:));
end
fclose(file);

%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAKE NEIGHBOUT MATRIX  %
%%%%%%%%%%%%%%%%%%%%%%%%%%
grid=myload([fwdir 'grid/grid' num2str(voxelsize) '.mat']);
nm=grid2NBmatrix(grid,voxelsize,6);
mysave([fwdir 'grid/neighbours_6_' num2str(voxelsize) '.mat'],nm);
nm=grid2NBmatrix(grid,voxelsize,18);
mysave([fwdir 'grid/neighbours_18_' num2str(voxelsize) '.mat'],nm);
nm=grid2NBmatrix(grid,voxelsize,26);
mysave([fwdir 'grid/neighbours_26_' num2str(voxelsize) '.mat'],nm);


%%%%%%%%%%%%%%%%%%%%%%%
% MAKE FORWARD MODEL  %
%%%%%%%%%%%%%%%%%%%%%%%
% Make forward matrix using dipoli
% -copy segdir/results/brain.tri
%       segdir/results/skin.tri 
%       segdir/results/skull.tri
%       fwdir/grid/grid N.txt and 
%       fwdir/electrodes/projmri.pos to programThom/temp
% - run dipoli
% - type /programThom/temp/skin.tri
% - type 1
% - type /programThom/temp/skull.tri
% - type 0.066666666667
% - type %/programThom/temp/brain.tri %Don't forget the % sign
% - type 1
% - enter
% - type /programThom/temp/projmri.pos
% - type /programThom/temp/grid N .txt
% - enter
% - type /programThom/temp/forwardmat N .txt   %output dus geen potentiaal
% verdeling gespecificeerd

% -type /programThom/mat2asci
% -choose /programThom/temp/forwardmat N .txt
% -choose /programThom/temp/forward N .txt 
% -copy resulting file as forwarddipoli N .txt to fwdir/forward/

copyfile([segdir '/results/skull.tri'], [temppath],'f');
copyfile([fwdir '/grid/grid' num2str(voxelsize) '.txt'],temppath,'f');
eval(['!' temppath '/dipoli -g ' temppath '/skin.tri 1 -g ' temppath '/skull.tri 0.066666666667 -g %' temppath '/brain.tri 1 -e ' temppath 'projmri.pos -s ' temppath 'grid' num2str(voxelsize) '.txt -t ' temppath '/forwardmat' num2str(voxelsize) '.txt']);
%copyfile(['./temp/dist' num2str(voxelsize) '.txt'], [fwdir 'grid/']);
eval(['!' temppath '/mat2asci ' temppath '/forwardmat' num2str(voxelsize) '.txt ' temppath '/forward' num2str(voxelsize) '.txt']);
copyfile([temppath '/forwardmat' num2str(voxelsize) '.txt'],[fwdir '/forward/'],'f');
copyfile([temppath '/forward' num2str(voxelsize) '.txt'],[fwdir '/forward/forwarddipoli' num2str(voxelsize) '.txt'],'f');

%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE IN MATLAB FORMAT  %
%%%%%%%%%%%%%%%%%%%%%%%%%%
forwarduncor=loadmat([fwdir 'forward/forwarddipoli' num2str(voxelsize) '.txt']); %258x4122 forwarduncor' needed, because loadmat transposes matrix
forwarduncor=forwarduncor'; %this ordering worked-out; loadmat is in the qtriplot dir
forwarduncor=forwarduncor(:,1:256); %first 256, because they are sorted before&256 electrodes
mysave([fwdir 'forward/forwardunnormal' num2str(voxelsize) '.mat'],forwarduncor');

%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY NORMALISATION  %
%%%%%%%%%%%%%%%%%%%%%%%%
% apply leadfield normaliziation by substracting for every virtual voxel, mean over all electrodes from every row of
% electrodes in the leadfield
forward=forwarduncor-mean(forwarduncor,2)*ones(1,size(forwarduncor,2));
mysave([fwdir 'forward/forward' num2str(voxelsize) '.mat'],forward);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAKE FIELDTRIP STRUCTURE  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid=[];
grid.dim=floor(s/voxelsize);
grid.xgrid=voxelsize:voxelsize:voxelsize*grid.dim(1);
grid.ygrid=voxelsize:voxelsize:voxelsize*grid.dim(2);
grid.zgrid=voxelsize:voxelsize:voxelsize*grid.dim(3);
ind=1:grid.dim(1)*grid.dim(2)*grid.dim(3);
[X,Y,Z]=ind2sub(grid.dim,ind);
grid.pos=[X' Y' Z']*voxelsize;
grid.inside=[];
griddef=myload([fwdir 'grid/grid' num2str(voxelsize) '.mat']);
for(ii=1:size(griddef,1))
    index=find(sum(grid.pos==(ones(size(grid.pos,1),1)*griddef(ii,:)),2)==3); 
    grid.inside(length(grid.inside)+1)=index;
end
grid.outside=1:grid.dim(1)*grid.dim(2)*grid.dim(3);
grid.outside(grid.inside)=[];
grid.leadfield={};
for(voxel=1:size(grid.pos,1))
    if(find(grid.inside==voxel)) %when the voxel is inside the brain, an index is searched wich is used for the leadfield matrix.
        index=find(sum(griddef==ones(size(griddef,1),1)*grid.pos(voxel,:),2)==3); 
        grid.leadfield{voxel}=forward(index*3-2:index*3,:)'; %(index-1)*3+(1:3); every voxel has a leadfield
    else
        grid.leadfield{voxel}=[NaN];
    end
end
grid.cfg=[];
mysave([fwdir 'forward/forwardft' num2str(voxelsize) '.mat'],grid);



%check if the left-ear in the mri and the left ear plotted over the voxels
%are the same


%{
%%%delete the files copied to cd
if (a==0) && (strcmp(cd,testcd)==1)
    delete pnt2elec
    disp 'pnt2elec deleted'
    if (b==0) && (strcmp(cd,testcd)==1)
        delete elec2pnt
        disp 'elec2pnt deleted'
    end
elseif (b==0) && (strcmp(cd,testcd)==1)
    delete elec2pnt
    disp 'elec2pnt deleted'
elseif strcmp(cd,testcd)
    disp 'both allready existed'
else disp 'something went wrong'
end
%}
