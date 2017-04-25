function z=jf_mks2nsfToy(varargin);
opts=struct('nCh',10,'N',300,'fs',128,'freq',16,'nSamp',[],'L',2,'d',10,'nAmp',5, ...
            'nAmp2',[],'nSpect',1,'phaseStd',pi/2,'freqStd',1,'f2Amp',[],'verb',0);
opts=parseOpts(opts,varargin);
if ( isempty(opts.nSamp) ) opts.nSamp=opts.fs*3; end;
if ( isempty(opts.nAmp2) ) opts.nAmp2=opts.nAmp; end;
if ( isempty(opts.f2Amp) ) 
   if ( numel(opts.freq)==1 ) opts.f2Amp=0; else opts.f2Amp=1; end
end
% Make a toy data-set
nCh=opts.nCh; N=opts.N; fs=opts.fs; nSamp=opts.nSamp;  L=opts.L; 
Yl = ceil(rand(N,1)*L); % True labels
Y  = lab2ind(Yl);       % indicator N x L

% Lots of internal sources to really mix things up
d=opts.d; fs=opts.fs;
nAmp=opts.nAmp; nAmp2=opts.nAmp2;  nSpect = opts.nSpect; f2Amp=opts.f2Amp;
phaseStd=opts.phaseStd; freqStd=opts.freqStd; % phase and spectral jitter
src_loc = [+.5 .2; -.5 .2;... % Internal signals
             0 .05;...    % Internal Noise sources
             0 .35;...
             .7*cos(linspace(0,pi,6))' .7*sin(linspace(0,pi,6))']'; % internal sources
elect_loc=[cos(linspace(0,pi,d))' sin(linspace(0,pi,d))']'; % electrodes
sources={{'sin' fs/opts.freq(1) 0 freqStd phaseStd} {'sin' fs/opts.freq(min(end,2)) 0 freqStd phaseStd} ...
         {'coloredNoise' nSpect}; 
         {'sin' fs/opts.freq(1) 0 freqStd phaseStd} {'sin' fs/opts.freq(min(end,2)) 0 freqStd phaseStd} ...
         {'coloredNoise' nSpect}; 
         {'coloredNoise' nSpect} {} {}; {'coloredNoise' nSpect} {} {}; ...
         {'coloredNoise' nSpect} {} {}; {'coloredNoise' nSpect} {} {}; ... 
         {'coloredNoise' nSpect} {} {}; {'coloredNoise' nSpect} {} {}; ...
         {'coloredNoise' nSpect} {} {}; {'coloredNoise' nSpect} {} {};};
% Translate from labels to source weightings: ch x src x lab
y2mix=cat(3,[1       1*f2Amp        nAmp2;  sqrt(2) sqrt(2)*f2Amp  nAmp2;  nAmp  0  0],... % Y==-1
            [sqrt(2) sqrt(2)*f2Amp  nAmp2;  1       1*f2Amp        nAmp2;  nAmp  0  0]);   % Y==1  
summary=sprintf('N=%d, L=%d, nAmp=%4f',numel(sources),size(Y,2),nAmp);
z=jf_mksfToy('Y',Y,'y2mix',y2mix,'fs',fs,'N',size(Y,1),'nCh',size(elect_loc,2),...
             'nSamp',nSamp,'sources',sources,'srcPos',src_loc,'chPos',elect_loc,'summary',summary);

% % mark internal sources as non-eeg
% iseeg = false(size(z.X,1),1); iseeg(end-d+1:end)=true; 
% [z.di(1).extra.iseeg]=num2csl(iseeg);

% % remove the internal sources we can't see
% z=jf_reject(z,'dim','ch','idx',~iseeg,'summary','true sources');

% Generate the outer folds
z.foldIdxs = gennFold(z.Y,10,'perm',0);
jf_disp(z)
return;
%-------------------------------------------------------------------
function testCase()

% Make nice visulation plot
z=jf_mks2nsfToy('N',10,'nSamp',30,'nAmp',.1);

ypi=find(z.Y(:,1)>0,1); yni=find(z.Y(:,1)<0,1);
S=sum(z.prep(2).info.S(:,:,[ypi yni],:),4);
D=z.X(:,:,[ypi yni]);
SD=cat(1,S(~[z.di(1).extra.iseeg],:,:),D([z.di(1).extra.iseeg],:,:));
clf;image3ddi(SD,z.di,1,'disptype','mcplot','zlabel','Class','colorbar','nw','ticklabs','sw');
packplots('sizes','equal','interplotgap',.002,'plotsposition',[.07 .09 .93 ...
                    .87])
saveaspdf('sftoyProblem');
