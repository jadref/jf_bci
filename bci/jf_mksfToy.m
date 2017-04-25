function [z]=jf_mksfToy(varargin);
% 
%  [z]=jf_mksfToy(varargin)
% Options:
%  Y       -- [N x L] or [N x 1] matrix of classes for each epoch
%  sources -- [nSrc x nSig] cell array of nSig signal type generated at each of nSrc source locations
%               default= ({{{'sin' 15}; {'coloredNoise' 1}}})
%               which is 2 source location, 1st with pure sin, 2nd with white noise and 3rd with nothing
%  srcPos -- [2 x nSrc]  or [3 x nSrc] position of the sources
%  y2mix -- [nSrc x nSig x L] mapping from classes(L) to weight over source locations and signals
%           OR
%           [nSrc x nSig x N] mapping epochs to from weight over sources-positions and types for each epoch
%  N     -- number of epoch to generate                              (100)
%  nCh   -- number of channels/detectors                             (10)
%  chPos -- [2 x nCh] or [ 3 x nCh ] position of the detectors
%  nSamp -- number of samples                                        (3*fs)
%  fs    -- sampling rate                                            (128)
%  amp/period/periodStd/phaseStd -- [nCh x M] set of source parameters
%  summary - [str] descriptive string
opts=struct('Y',[],'y2mix',[],'N',100,'nCh',10,'fs',128,'nSamp',[], ...            
            'sources',{{{'sin' 15}; {'coloredNoise' 1}}},...
            'srcPos',[],'srcNames',[],'chPos',[],'chNames',[],'epsilon_0',[],'summary','','verb',0);
opts=parseOpts(opts,varargin);


src_loc=opts.srcPos; if(isempty(src_loc))  src_loc=size(opts.sources,1); end;
if ( numel(src_loc)==1 ) nSrc=src_loc; else nSrc=size(src_loc,2); end;
nSig=size(opts.sources,2);
elect_loc=opts.chPos;if(isempty(elect_loc))elect_loc=opts.nCh; end;
if ( numel(elect_loc)==1 ) nCh=elect_loc; else nCh=size(elect_loc,2); end;
if ( isempty(opts.nSamp) ) opts.nSamp=opts.fs*3; end;

Y=opts.Y;if ( isempty(Y) ) Y=sign(randn(opts.N,1)); end;
%if ( isempty(Y) ) % if Y unspec then 1 per source
%  M=2; % binary
%  Yl =ceil(rand(opts.N,1)*M); Y=lab2ind(Yl,unique(Yl),[],[],0);
%else
if ( numel(Y)==max(size(Y)) || ~all(Y(:)==-1 | Y(:)==0 | Y(:)==1) ) % vector of labels
  Yl=Y;
  [Y,key]=lab2ind(Y,[],[],[],0); 
else
  if ( size(Y,2)==1 ) [ans,Yl]=max([Y -Y],[],2); % binary is special case
  else                [ans,Yl]=max(Y,[],2);
  end
end
if ( ~isempty(Y) ) N=size(Y,1); end;
y2mix=opts.y2mix;
if ( isempty(y2mix) )
  if ( isempty(Y) )
    y2mix=ones(nSrc,nSig,opts.N); 
  else    
    y2mix=ones(nSrc,nSig,size(Y,2)); 
    for i=1:size(Y,2); y2mix(1,:,i)=y2mix(1,:,i)*(size(Y,2)-(i-1))./size(Y,2); end; % scaled
  end
end;
mix = y2mix;
if ( ~isempty(Yl) && size(y2mix,3)==numel(unique(Yl)) ) % map from labels to epochs
   Yi=Yl; for li=1:numel(key); Yi(Yl==key(li))=li; end; % convert to indices
   mix =y2mix(:,:,Yi);
end

% call inner code to do the actual work
[X,A,S,src,elect]=mksfToy(opts.sources,mix,opts.nSamp,src_loc,elect_loc,'epsilon_0',opts.epsilon_0);

X=single(X);
Cnames=opts.chNames;
if ( isempty(Cnames) ) Cnames=split(',',sprintf('%d,',1:size(X,1)));Cnames=Cnames(1:size(X,1)); end
di=mkDimInfo(size(X),'ch',[],Cnames,...
             'time','ms',(0:opts.nSamp-1)*1000/opts.fs,...
             'epoch','',[]);
if ( size(elect,1)==2 ) 
  [di(1).extra.pos2d] =num2csl(elect,1);
else
  [di(1).extra.pos3d] =num2csl(elect,1);
  [di(1).extra.pos2d] =num2csl(xyz2xy(elect),1);
end
if ( ~isempty(Yl) ) [di(3).extra.marker]=num2csl(Yl,2); end;
di(2).info.fs = opts.fs;

iseeg = true(size(X,1),1); iseeg(1:size(opts.sources,1)-1)=false; 
[z.di(1).extra.iseeg]=num2csl(iseeg);

% Rec the source matrix annotation
Sdi = di([1:end end]); Sdi(end-1)=mkDimInfo(size(S,4),1,'source');

z=jf_import('toy/sf','default','sftoy',X,di,'summary',opts.summary,'Y',Yl,'Ydi','epoch');
if ( isfield(z,'Y') ) 
  z.foldIdxs = gennFold(z.Y,10,'perm',0);
end
info = struct('sources',{opts.sources},'A',A,'S',S,'Sdi',Sdi,'srcPos',src,'y2mix',y2mix);
z=jf_addprep(z,'mksfToy','spatially filtered toy dataset',opts,info);
return;
%-----------------------------------------------------
function testCase()

nCh=10; N=100; fs=128; nSamp=3*fs;  
L=2; % 2 types label, L/R, diff freq, amp change with label
Yl=ceil(rand(N,1)*L);        % True labels
Y =double(repop(Yl,'==',1:L));% indicator matrix

% Pure source without noise
y2mix=cat(3,[sqrt(2);1],[1;sqrt(2)],[1;1]); % ch x source x labels
z=jf_mksfToy(Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'sources',{{'sin' fs/16};{'sin' fs/16};{'none' 0}});

% 3-d version
[nm latlong xy xyz]=readCapInf('cap16');%'1010');%'cap32');
chPos=xyz;
y2mix=cat(3,[sqrt(2);1],[1;sqrt(2)],[1;1]); % ch x source x labels
z=jf_mksfToy(Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nSamp',3*fs,...
             'sources',{{'sin' fs/16};{'sin' fs/16};{'none' 0}},'srcPos',10,...
             'chPos',chPos,'chNames',nm);


% Pure source with phase jitter
y2mix=cat(3,[sqrt(2);1],[1;sqrt(2)],[1;1]); % ch x source x labels
z=jf_mksfToy(Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'sources',{{'sin' fs/16 0 pi/2};{'sin' fs/24 0 pi/2};{'none' 0}});

% Pure single freq source with phase jitter
% power switch in 1st source btw ch with labels
y2mix=cat(3,[sqrt(2) 3;1 3],[1 3;sqrt(2) 3],[1 3;1 3]); % ch x source x labels
z=jf_mksfToy(Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'sources',{{'sin' fs/16 0 pi/2};{'sin' fs/16 0 pi/2};{'none' 0}});

% Lots of internal sources to really mix things up
d=10; fs=128;
nAmp=1; phaseStd=pi/2; periodStd=1;
elect_loc = [+.5 .2;... % Internal signals
             -.5 .2;... % Internal signals
             0 .05;...    % Internal Noise sources
             0 .35;...
             .7*cos(linspace(0,pi,6))' .7*sin(linspace(0,pi,6))';...
             cos(linspace(0,pi,d))' sin(linspace(0,pi,d))']'; % electrodes
sources = { {'sin' fs/15 0 periodStd phaseStd}; ...
            {'sin' fs/24 0 periodStd phaseStd}; ... % N.B. RMS = Amp.^2/2
            {'coloredNoise' 1}; {'coloredNoise' 1}; {'coloredNoise' 1};...
            {'coloredNoise' 1}; {'coloredNoise' 1}; {'coloredNoise' 1};...
            {'coloredNoise' 1}; {'coloredNoise' 1}; ...
            {'none' 1} };
y2mix=cat(3,[1;sqrt(2);nAmp],[sqrt(2);1;nAmp]); % ch x source x label
z=jf_mksfToy(Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nCh',size(elect_loc,2),...
             'nSamp',3*fs,'sources',sources,'pos2d',elect_loc);


% Visualise
clf;image3ddi(z.X,z.di,1,'disptype','image','clim',[],'colorbar','nw');packplots('sizes','equal');
clf;image3d(z.X(:,:,1:4),1,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','clim',[],'colorbar','nw')

muX = cat(3,mean(z.X(:,:,z.Y(:,1)>0),3),mean(z.X(:,:,z.Y(:,2)>0),3));
clf;image3d(muX,1,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','clim',[],'colorbar','nw')

A=dv2auc(z.Y(:,1)*2-1,z.X,strmatch('epoch',{w.di.name}));
clf;image3ddi(A,z.di,1,'disptype','plot','clim',[0 1],'colorbar','nw')

% Spectral visulation
% Welch
w=jf_welchpsd(z,'width_ms',500);
muW=cat(3,mean(w.X(:,:,w.Y(:,1)>0),3),mean(w.X(:,:,w.Y(:,2)>0),3));
clf;image3ddi(muW,w.di,1,'disptype','image','clim','minmax','colorbar','nw');packplots('sizes','equal');

A=dv2auc(w.Y(:,1),w.X,strmatch('epoch',{w.di.name}));
clf;image3ddi(A,w.di,1,'disptype','plot','clim',[0 1],'colorbar','nw');packplots('sizes','equal');

% spectrogram
s=jf_spectrogram(z,'width_ms',500);
muS=cat(4,mean(s.X(:,:,:,s.Y(:,1)>0),3),mean(s.X(:,:,:,s.Y(:,2)>0),3));
clf;image3ddi(muS,s.di,1,'disptype','image','clim','minmax','colorbar','nw');packplots('sizes','equal');

A=dv2auc(s.Y(:,1),s.X,strmatch('epoch',{s.di.name}));
clf;image3ddi(A,s.di,1,'disptype','image','clim',[0 1],'colorbar','nw');packplots('sizes','equal');

