% START ANALYSIS
% This code does beamformer analyis as described in Rianne's thesis. 

clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%VARIABLE CONFIGURATIONS   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBJECT
subject     =   ('desain');
foldername=('TFT_peter_130606');
datasetp=('TFTp_peter_130606_');
dataseta=('TFTanf_peter_130606_');
throwout=[254 238 239 240 236 241];%real bad electrodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER  CONFIGURATIONS    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startpath=['/Volumes/'];                %mac

eegdirp      =   ([startpath 'BCI_Data/__old_dir/Frequency Tagging/Tactile Frequency Tagging/' foldername '/perception/']); 
datasetnamep     =  ([eegdirp 'data/' datasetp]);
eegdira      =   ([startpath 'BCI_Data/__old_dir/Frequency Tagging/Tactile Frequency Tagging/'  foldername '/attentionNoFeedback/']); 
datasetnamea     =  ([eegdira 'data/' dataseta]);
fwdir      =   ([startpath 'BCI_Data/subject_data/forwardmodels/' subject '/']);
if(strcmp(subject,'desain'))
anamri=    ([startpath 'BCI_Data/subject_data/' subject '/segmentation (rianne)/oudMRIVERSLAG/analysis/readmri.mat']);
brainfile=([startpath 'BCI_Data/subject_data/' subject '/segmentation (rianne)/oudMRIVERSLAG/analysis/brain.mat']);
else
anamri=    ([startpath 'BCI_Data/subject_data' subject '/segmentation/analysis/readmri.mat']);
brainfile= ([startpath 'BCI_Data/subject_data/' subject '/segmentation/analysis/brain.mat']);
end
nfile=([startpath 'BCI_data/__old_dir/00 Admin/Caps/neighbormatrix.mat']);


% PATHS
addpath         (genpath([startpath 'BCI code/toolboxes/utilities/file_io']));
addpath         (genpath([startpath 'BCI code/toolboxes/eeg_analysis/beamforming/beamformer/'])); 
%addpath         (genpath([startpath 'BCI code/fieldtrip/fieldtrip-20060529/']));
addpath         (genpath([startpath 'BCI code/external_toolboxes/fieldtrip/']));
addpath         (genpath([startpath 'BCI code/external_toolboxes/spm2/']));
%addpath         (genpath([startpath 'BCI code/__concept_new_structure/toolboxes/eeg_analysis/beamforming/utilities/vol3d/']));

voxelsize=11;
nrexp=2;
mri=myload(anamri);
s=size(mri.anatomy);
clear mri;
bdfpa=[1 3;2 4];%datasets, first row perception, second row attention
ppmarkerspa=[9 10 11 12 101;3 38 4 39 101];%preprocess markers, first row perception, second row attention
tlmarkersp=[9 12 101;10 11 101];%use for covariance matrix perception
tlmarkersa=[3 38 101;4 39 101];%use for covariance matrix attention


return
%Little correction to make some maps if they ain't there yet.
for exp=1:nrexp
    if exp==1
        eegdir=eegdirp;
    else
        eegdir=eegdira;
    end
    %a = exist([eegdir 'datasets/']);
    b = exist([eegdir 'analysis/setsBas/vox/']);
    %c = exist([fwdir 'grid/']);
    %d = exist('./temp/');


    % while a == 0
    %     mkdir([eegdir 'datasets/']);
    %     a = exist([eegdir 'datasets/']);
    % end
    while b == 0
        mkdir([eegdir 'analysis/setsBas/vox/']);
        b = exist([eegdir 'analysis/setsBas/vox/']);
    end
    % while c == 0
    %     mkdir([fwdir 'grid/']);
    %     c = exist([fwdir 'grid/']);
    % end
    % while d == 0
    %     mkdir('./temp/');
    %     d = exist('./temp/');
    % end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     PREPROCESSING        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%wat doet dit pre processen?? en waarom is het nu al nodig om eeg data te
%gebruiken

for exp=1:2
    if exp==1
        type='p';
        datasetname=datasetnamep;
        eegdir=eegdirp;
    else
        type='a';
         datasetname=datasetnamea;
          eegdir=eegdira;
    end
    ppbdf=bdfpa(exp,:);
    ppmarkers=ppmarkerspa(exp,:);
    for(b=1:length(ppbdf))
        bdf=ppbdf(b);
        dataset     =  ([datasetname num2str(bdf) '.bdf']);
        for(n=1:length(ppmarkers))
            marker=ppmarkers(n);
            if(marker==101)
                version=3;% 3 parts of 1.5 second starting 1.5 second from marker
            else
                version=2;%parts of 1.5 second
            end
            data=preprocessingle(marker,dataset,version);
            mysave([eegdir 'analysis/preprocsingle/'  , num2str(marker) '_' num2str(bdf),'.mat'],data);
        end
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     MAKE BEAMFILTER      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sens=myload([fwdir 'electrodes/elec.mat']);
cfgtl = [];
cfgtl.covariance='yes';
cfgtl.blc='yes';
cfgbe = [];
cfgbe.rianne=1; %??
cfgbe.grid=myload([fwdir 'forward/forwardft' num2str(voxelsize) '.mat']);
cfgbe.grid=removefromleadfield(cfgbe.grid,throwout);
cfgbe.grid=averageftgrid(cfgbe.grid);
cfgbe.method = 'lcmv';
cfgbe.projectnoise = 'yes';
cfgbe.sens=sens;
cfgbe.elec=sens;
cfgbe.vol=0;
cfgbe.keepfilter='yes';
cfgbe.keepleadfield='yes';
for exp=1:2
    if exp==1
        eegdir=eegdirp;
    else
        eegdir=eegdira;
    end
    tlbdf=bdfpa(exp,:);
    for(ii=1:length(tlbdf))
        data=myload([eegdir 'analysis/preprocsingle/101'  '_' num2str(tlbdf(ii)) '.mat']);
        data=removechannels(data);
        data=blcRianne(data);
        [data, badv, badr, nv, nr]=removebadelectrodesnew(data,sens,0.05,nfile);
        mysave([eegdir 'analysis/preprocsingle/removedbadandchan101' '_' num2str(tlbdf(ii)) '.mat'],data);
    end
end
for exp=1:2
    if exp==1
        eegdir=eegdirp;
        tlmarkersset=tlmarkersp;
    else
        eegdir=eegdira;
        tlmarkersset=tlmarkersa;
    end
    tlbdf=bdfpa(exp,:);
    for(jj=1:length(tlbdf)) %??
        bdfnow=tlbdf(jj);
        for(set=1:size(tlmarkersset,1))
            tlmarkers=tlmarkersset(set,:);
            N=inf;
            for(ii=1:length(tlmarkers))
                if(~(tlmarkers(ii)==101))
                    data=myload([eegdir 'analysis/preprocsingle/' num2str(tlmarkers(ii)) '_' num2str(bdfnow) '.mat']);
                    data=removechannels(data);
                    data=blcRianne(data);
                    [data, badv, badr, nv, nr]=removebadelectrodesnew(data,sens,0.05,nfile);%nog op hele set
                    mysave([eegdir 'analysis/preprocsingle/removedbadandchan' num2str(tlmarkers(ii)) '_' num2str(bdfnow) '.mat'],data);
                    if(N>length(data.trial))
                        N=length(data.trial);
                    end
                end
            end
            data=myload([eegdir 'analysis/preprocsingle/removedbadandchan' num2str(tlmarkers(1)) '_' num2str(bdfnow) '.mat']);
            data.trial=[];
            data.time=[];
            for(ii=1:length(tlmarkers))
                if(~(tlmarkers(ii)==101))
                    plusdata=myload([eegdir 'analysis/preprocsingle/removedbadandchan' num2str(tlmarkers(ii)) '_' num2str(bdfnow) '.mat']);
                    data.trial=[data.trial,plusdata.trial(1:N)];
                    data.time=[data.time,plusdata.time(1:N)];
                else
                    plusdata1=myload([eegdirp 'analysis/preprocsingle/removedbadandchan101_1' '.mat']);
                    plusdata2=myload([eegdirp 'analysis/preprocsingle/removedbadandchan101_3' '.mat']);
                    plusdata3=myload([eegdira 'analysis/preprocsingle/removedbadandchan101_2.mat']);
                    plusdata4=myload([eegdira 'analysis/preprocsingle/removedbadandchan101_4.mat']);
                    plusdata=plusdata1;
                    plusdata.trial=[];
                    plusdata.time=[];
                    plusdata.trial=[plusdata1.trial,plusdata2.trial,plusdata3.trial,plusdata4.trial];
                    plusdata.time=[plusdata1.time,plusdata2.time,plusdata3.time,plusdata4.time];
                    plusdata.trial=plusdata.trial(1:N);
                    plusdata.time=plusdata.time(1:N);
                    data.trial=[data.trial,plusdata.trial];
                    data.time=[data.time,plusdata.time];
                end
            end
            data=blcRianne(data);
            data=permremovebad(data,throwout); 
            data=avgrefRianne(data);
            filename=[];
            for(jj=1:length(tlmarkers))
                filename=[filename num2str(tlmarkers(jj)) '_'];
            end
            mysave([eegdir 'analysis/' 'preprocsingle/all' ,filename num2str(bdfnow) '.mat'],data);
            dataTL = timelockanalysis(cfgtl, data);
            mysave([eegdir 'analysis/' 'timelock/' ,filename num2str(bdfnow) '.mat'],dataTL);
            dataSource  = sourceanalysis(cfgbe,dataTL);
            mysave([eegdir 'analysis/beamtime/'  filename num2str(bdfnow) '.mat'],dataSource);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     FILTER THE DATA      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
makedatasets([9 12 101],eegdirp, nfile, fwdir, 1, throwout, voxelsize,myload([eegdirp 'analysis/beamtime/9_12_101_1.mat']) );
makedatasets([10 11 101],eegdirp, nfile, fwdir, 1, throwout, voxelsize,myload([eegdirp 'analysis/beamtime/10_11_101_1.mat']) );
makedatasets([3 38 101],eegdira, nfile, fwdir, 2, throwout, voxelsize,myload([eegdira 'analysis/beamtime/3_38_101_2.mat']) );
makedatasets([4 39 101],eegdira, nfile, fwdir, 2, throwout, voxelsize,myload([eegdira 'analysis/beamtime/4_39_101_2.mat']) );
makedatasets([9 12 101],eegdirp, nfile, fwdir, 3, throwout, voxelsize,myload([eegdirp 'analysis/beamtime/9_12_101_3.mat']) );
makedatasets([10 11 101],eegdirp, nfile, fwdir, 3, throwout, voxelsize,myload([eegdirp 'analysis/beamtime/10_11_101_3.mat']) );
makedatasets([3 38 101],eegdira, nfile, fwdir, 4, throwout, voxelsize,myload([eegdira 'analysis/beamtime/3_38_101_4.mat']) );
makedatasets([4 39 101],eegdira, nfile, fwdir, 4, throwout, voxelsize,myload([eegdira 'analysis/beamtime/4_39_101_4.mat']) );

%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VOXEL AND ELEC ANALYSIS FOR SESSION 1 AND 2 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load voxel grid
grid=myload([fwdir 'grid/grid' num2str(voxelsize) '.mat']);
%find voxels
voxelr=findvoxel([123 80 204],grid);
voxell=findvoxel([43 80 204],grid);
%set standard CS for Bas
CS=[];
CS.subject=subject;
CS.Fs=256;
data=myload([eegdirp 'analysis (Rianne)/erp_ft_template.mat' ]); %waar komt het vandaan? alle data van alle trials in class A en B... is nodig zodat hierop PCA gedaan kan worden
CS.t=data.time;
clear data;
%perception 9-12:
eegdir=eegdirp;
stim1name='9_1';
stim2name='12_1';
fixname='101_1';
beamfilter='_912101';
l1='l-26';
l2='r-20';
CS.markers=[9 12];
CSe=CS;
CS.description=['TFT_P_peter_130606_beam_session1'];CSe.description=['TFT_P_peter_130606_elc_session1'];
%stim1name=[stim1name beamfilter];
%stim2name=[stim2name beamfilter];
%fixname=[fixname beamfilter];
doAnalysisVox([stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],eegdir,CS,voxelsize,s,anamri,brainfile,grid,0,voxell,voxelr);
makevoxelplots(voxell,voxelr,eegdir,[stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],l1,l2,CS.t);
doAnalysisElc(stim1name,stim2name,fixname,eegdir,CSe,eegdirp,l1,l2,startpath,throwout,voxell,voxelr,beamfilter,fwdir);
makeelcplots(voxell,voxelr,eegdir,stim1name,stim2name,fixname,l1,l2,CS.t,beamfilter,fwdir,eegdirp,startpath);
%perception10-11:
eegdir=eegdirp;
stim1name='10_1';
stim2name='11_1';
fixname='101_1';
beamfilter='_1011101';
l1='l-20';
l2='r-26';
CS.markers=[10 11];
CSe=CS;
CS.description=['TFT_P_peter_130606_beam_session1'];
CSe.description=['TFT_P_peter_130606_elc_session1'];
doAnalysisVox([stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],eegdir,CS,voxelsize,s,anamri,brainfile,grid,0,voxell,voxelr);
makevoxelplots(voxell,voxelr,eegdir,[stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],l1,l2,CS.t);
doAnalysisElc(stim1name,stim2name,fixname,eegdir,CSe,eegdirp,l1,l2,startpath,throwout,voxell,voxelr,beamfilter,fwdir);
makeelcplots(voxell,voxelr,eegdir,stim1name,stim2name,fixname,l1,l2,CS.t,beamfilter,fwdir,eegdirp,startpath);
%attention38-3:
eegdir=eegdira;
stim1name='38_2';
stim2name='3_2';
fixname='101_2';
beamfilter='_338101';
l1='L-26/r-20';
l2='l-26/R-20';
CS.markers=[38 3];
CSe=CS;
CS.description=['TFT_A_peter_130606_beam_session2'];
CSe.description=['TFT_A_peter_130606_elc_session2'];
doAnalysisVox([stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],eegdir,CS,voxelsize,s,anamri,brainfile,grid,1,voxell,voxelr);
makevoxelplots(voxell,voxelr,eegdir,[stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],l1,l2,CS.t);
doAnalysisElc(stim1name,stim2name,fixname,eegdir,CSe,eegdirp,l1,l2,startpath,throwout,voxell,voxelr,beamfilter,fwdir);
makeelcplots(voxell,voxelr,eegdir,stim1name,stim2name,fixname,l1,l2,CS.t,beamfilter,fwdir,eegdirp,startpath);
%attention39-4:
eegdir=eegdira;
stim1name='39_2';
stim2name='4_2';
fixname='101_2';
beamfilter='_439101';
l1='L-20/r-26';
l2='l-20/R-26';
CS.markers=[39 4];
CSe=CS;
CS.description=['TFT_A_peter_130606_beam_session2'];
CSe.description=['TFT_A_peter_130606_elc_session2'];
doAnalysisVox([stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],eegdir,CS,voxelsize,s,anamri,brainfile,grid,1,voxell,voxelr);
makevoxelplots(voxell,voxelr,eegdir,[stim1name beamfilter],[stim2name beamfilter],[fixname beamfilter],l1,l2,CS.t);
doAnalysisElc(stim1name,stim2name,fixname,eegdir,CSe,eegdirp,l1,l2,startpath,throwout,voxell,voxelr,beamfilter,fwdir);
makeelcplots(voxell,voxelr,eegdir,stim1name,stim2name,fixname,l1,l2,CS.t,beamfilter,fwdir,eegdirp,startpath);

%%%%%%%%%%%%%%%%%%
% CLASSIFICATION %
%%%%%%%%%%%%%%%%%%
addpath         (genpath([startpath 'BCI code/beamformer/programsBas/FeatRianne2']));
filenameNB=[startpath 'BCI code/beamformer/programsBas/FeatRianne2/V2/_private/neighbormatrix.mat'];
%elc
neighbmat=myload([startpath 'BCI code/beamformer/programsBas/FeatRianne2/V2/_private/neighbormatrix_250.mat']);
save(filenameNB,'testneighbmat');
FeatFrame('E9121.cfg');
FeatFrame('E3382.cfg');
FeatFrame('E10111.cfg');
FeatFrame('E4392.cfg');
%voxel
neighbmat=myload([startpath 'BCI code/beamformer/programsBas/FeatRianne2/V2/_private/neighbours_6_11.mat']);
save(filenameNB,'testneighbmat');
FeatFrame('B9121.cfg');
FeatFrame('B10111.cfg');
FeatFrame('B3382.cfg');
FeatFrame('B4392.cfg');





