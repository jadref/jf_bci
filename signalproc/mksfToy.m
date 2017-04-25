function [X,A,S,src,dest]=mksfToy(sources,mix,T,src_loc,elect_loc,varargin)
% make a spatial filtering toy problem
%
% [X,A,S,src,dest]=mksfToy(sources,mix,T,src_loc,elect_loc,sources[,options])
%
% Inputs:
%  sources   -- [ d x M ] cell array of source descriptions, d=#source locations, M=#source signals
%    {function_name params} -- 
%       where function name is the name of the function to call and params
%       are its parameters -- such that it can be called as:
%       function_name(T,params{:}).  Builtin fns are:
%        coloredNoise(T,spect) -- specific type of coloured noise
%        sin(T,period,phase,periodStd,phaseStd) -- sinusoid period, phase
%                           and given jitter about these values
%        N.B. unit amp Sin has RMS=.5!
%        none(T)               -- no source at this pos/source
%               N.B. if d < num src_loc then rest electrodes use same
%               signal as final electrode and if M'th sig is empty we use
%               the same as for the previous source
%        saw(T,period)
%        square(T,period)
%  mix       -- [ d x M x N ] list of src_loc x src_signal x epoch weights
%  T         -- [ 1 x 1 ] number of samples per trial
%  src_loc   -- [ 2 x M ] or [ 3 x M ] positions of the sources 
%               [ 1 x 1 ] this number of equally spaced sources
%  elect_loc -- [ 2 x d ] or [ 3 x d ] positions of the electrodes 
%               [ 1 x 1 ] this number of equally spaced electrodes
%            (10 semi-circle on circumference circle radius 1 + 2 at +/1 .25,0)
% Options:
%  epsilon_0 -- [1x1] value of permittivity for signal attenuation (.1)
% Outputs:
%  X         -- [ d x T x N ] generated epochs
%  A         -- [ nSrc x d ] forward source mixing matrix
%  S         -- [ nSrc x T x N x M] generated sources, per electrode,sample,epoch,source
%  src       -- [ 2 x nSrc ] or [ 3 x nSrc ] set of source positions
%  dest      -- [ 2 x d ] or [ 3 x d ] set of electrode positions
%
% Examples:
% N=1000; % number of examples to generate
% sources = { {'sin' 5}; {'none' 1} };  % 1 source, rest detectors
% y2mix=cat(3,[1],[.5]); % ch x source x label
% mix =y2mix(:,:,(randn(N,1)>0)+1);         % ch x source x N
% [X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);
%
% sources = { {'prod' {'exp' log(2)./30} {'sin' 8}}}; % exp decay sin
% 
%  % Pure signal, diff freqs
%sources = { {'sin' 5} {'coloredNoise' 1};   % 5-sample sin with background noise @ pos1
%            {'sin' 8} {'coloredNoise' 1};   % 8-sample sin with background noise @ pos2
%            {'none' 1} {}}; % rest just detectors
%y2mix=cat(3,[1;.5],[.5;1]); % ch x source x label, inc source 1 in class 1, inc source 2 in class 2
%mix  =y2mix(:,:,(randn(N,1)>0)+1);  % ch x source x N
%[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);
opts=struct('epsilon_0',1e-1);
opts=parseOpts(opts,varargin);
if ( nargin < 4 || isempty(src_loc) ) src_loc=size(sources,1); end;
if ( nargin < 5 ) elect_loc=[]; end;
if ( numel(src_loc)==1 ) nsrc=src_loc; else nsrc=size(src_loc,2); end;
if ( nsrc > size(sources,1) ) % replicate last entry
   sources(end+1:nsrc,:) = repmat(sources(end,:),[nsrc-size(sources,1),1]);
   mix(end+1:nsrc,:,:)   = repmat(mix(end,:,:),  [nsrc-size(mix,1),ones(1,ndims(mix)-1)]);
end
% generate the mixed sources
[X,S]=mixSig(sources,mix,T);

% get the fwd mx
[A,src,dest]=toyFwdMx(src_loc,elect_loc,opts.epsilon_0);
% map through the forward matrix to the output
X = tprod(single(X),[-1 2 3],single(A),[-1 1]);
return

%---------------------------------------------------------------------------
function testCase()
N=100; L=2;
fs = 128; T=3*fs;

Yl = (randn(N,1)>0)+1;     % True labels + 1
Y  = double([Yl==1 Yl==2]);% indicator N x L

% Single signal
sources = { {'sin' 5}; {'none' 1} };  % 1 source_loc with 1 src_sig, rest detectors
y2mix=cat(3,[1],[.5]); % src_loc x src_sig x label
mix =y2mix(:,:,Yl);    % src_loc x src_sig x N
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);

% Pure signal, diff freqs
sources = { {'sin' 5};   % inc amplitude label=2
            {'sin' 8};   % dec amplitude label=1
            {'none' 1}}; % rest just detectors
y2mix=cat(3,[1;.5],[.5;1]); % src_loc x src_sig x label
mix  =y2mix(:,:,Yl);  % src_loc x src_sig x N
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);

% A Mix of signal types..
sources = { {'sin' 15};
            {'saw' 18};
            {'square',20};
            {'coloredNoise' 1};
            {'coloredNoise' [1 0]};
            {'coloredNoise' 1./(1:50)}
            {'none' 1}
            }
mix=ones(1,1,N);
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);

% A lot of internal sources to confuse things
d=10;
elect_loc = [+.5 .2;... % Internal source 1
             -.5 .2;... % Internal source 2
             0 0;...
             0 .4;...
             .7*cos(linspace(0,pi,6))' .7*sin(linspace(0,pi,6))';...
             cos(linspace(0,pi,d))' sin(linspace(0,pi,d))']';
sources = { {'sin' fs/15}; {'sin' fs/24}; ...
            {'coloredNoise' 1}; {'coloredNoise' 1}; {'coloredNoise' 1};...
            {'coloredNoise' 1}; {'coloredNoise' 1}; {'coloredNoise' 1};...
            {'coloredNoise' 1}; {'coloredNoise' 1}; ...
            {'none' 1} };
y2mix=cat(3,[1;.5;nAmp],[.5;1;nAmp]); % src_loc x src_sig x label
mix  =y2mix(:,:,Yl);  % src_loc x src_sig x N
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T,elect_loc);


% With some noise
sources = { {'sin' 5} {'coloredNoise' 1};
            {'sin' 8} {'coloredNoise' 1};
            {'none' 1} {}}; % rest just detectors
y2mix=cat(3,[1 1;.5 1],[.5 1;1 1]); % src_loc x src_sig x label
mix  =y2mix(:,:,Yl);                % src_loc x src_sig x N
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);

% With some noise + phase jitter
freqStd=0; phaseStd=pi/2; nAmp=5;
sources = { {'sin' 5 0 freqStd phaseStd} {'coloredNoise' 1};
            {'sin' 8 0 freqStd phaseStd} {'coloredNoise' 1};
            {'none' 1} {}}; % rest just detectors
y2mix=cat(3,[1 nAmp;.5 nAmp],[.5 nAmp;1 nAmp]); % src_loc x src_sig x label
mix  =y2mix(:,:,Yl);                % ch x src_sig x N
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T);

dv2auc(Y(:,1)*2-1,sum(X.^2,2),3), % AUC for each channels power

iseeg = true(size(X,1),1); iseeg(1:2)=false; 
trnInd= ones(size(X,3),1); trnInd(1:floor(end*.25))=-1;
[sf,d]=csp(X(iseeg,:,trnInd(:,1)>0),Y(trnInd(:,1)>0,1)*2-1,[-1 1]);
sfX    = tprod(X(iseeg,:,:),[-1 2 3],sf,[-1 1]); % map to csp space, [sf x time x epoch]
dv2auc((Y(:,1)*2-1).*double(trnInd(:,1)<0),sum(sfX.^2,2),3)


% Visualise the result
clf;image3d(S(:,:,1:10),1,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','clim','minmax','colorbar','nw')

clf;mimage(shiftdim(X(1,:,Y(:,1)>0)),shiftdim(X(1,:,Y(:,2)>0)),shiftdim(X(2,:,Y(:,1)>0)),shiftdim(X(2,:,Y(:,2)>0)),'clim','minmax','title',{'S=1 Y=1' 'S=1 Y=2' 'S=2 Y=1' 'S=2 Y=2'},'xlabel','epoch','ylabel','time');

clf;image3d(X(:,:,1:2),1,'plotPos',elect_loc,'disptype','plot','clim',[],'colorbar','nw')

muX=cat(3,mean(X(:,:,Y(:,1)>0),3),mean(X(:,:,Y(:,2)>0),3));
clf;image3d(muX,1,'plotPos',elect_loc,'disptype','plot','clim',[])

sX=spectrogram(X,2);
musX = cat(4,mean(sX(:,:,:,Y(:,1)>0),4),mean(sX(:,:,:,Y(:,2)>0),4));
clf;image3d(sX(:,:,:,1),1,'plotPos',elect_loc,'disptype','image','clim',[],'xlabel','ch','ylabel','freq','zlabel','time');packplots('sizes','equal');


% 3-d version;
[nm latlong xy xyz]=readCapInf('cap16');%'1010');%'cap32');
elect_loc=xyz;
[X,A,S,src_loc,elect_loc]=mksfToy(sources,mix,T,[],elect_loc);
