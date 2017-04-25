function [X,S]=mixSig(sources,mix,T,nCh,verb)
% mix a given set of signal types to produce mixed outputs
%
% [X,S]=mixSig(sources,mix,T,d)
%
% Inputs:
%  sources   -- [ d x M ] cell array of source descriptions, d=#source locations, M=#source signals
%               per source can be:
%    {function_name params} -- 
%       where function name is the name of the function to call and params
%       are its parameters -- such that it can be called as:
%       function_name(T,params{:}).  Builtin fns are:
%        coloredNoise(T,spect) -- specific type of coloured noise
%        sin(T,period,phase,periodStd,phaseStd) -- sinusoid period, phase
%                           and given jitter about these values
%        N.B. unit amp Sin has RMS=.5!
%        none(T)               -- no source at this pos/source
%               N.B. if d < num elect_loc then rest electrodes use same
%               signal as final electrode and if M'th sig is empty we use
%               the same as for the previous source
%        saw(T,period)
%        square(T,period)
%  mix       -- [ d x M x N ] list of [src_loc x src_sig x epoch] weights
%  T         -- [ 1 x 1 ] number of samples per trial
% Outputs:
%  X         -- [ d x T x N ] generated epochs
%  S         -- [ d x T x N x M] generated sources, per electrode,sample,epoch,source
if ( nargin < 4 || isempty(nCh) ) nCh=size(sources,1); end;
if ( nargin < 5 || isempty(verb) ) verb=1; end;
% set the output size
M=size(sources,2); N=size(mix,3);

% Compute the true sources
S = zeros([nCh,T,N,M],'single'); % ch x time x epoch x source
if ( verb>0 ) fprintf('mixSig:'); end;
for di=1:nCh; % for each electrode
   for tri=1:N; % for each epoch
      validD = min(size(sources,1),di); % last valid source entry
      maxvalY= find(~cellfun('isempty',sources(validD,:)),1,'last');
      for si=find(mix(min(di,end),:,tri)); % for each (non-zero) source
         sigType = sources{validD,min(maxvalY,si)};
         if ( isnumeric(sigType) ) sigType={sigType}; end;
         S(di,:,tri,si) = mix(min(di,end),min(si,end),tri)*mkSig(T,sigType{:});
       end
       if ( verb>0 ) textprogressbar((di-1)*N+tri,N*nCh); end;
   end
end
if ( verb>0 ) fprintf('\n'); end;
% Compute the mixed output
X = single(sum(S,4));
return

%---------------------------------------------------------------------------
function testCase()
N=100; L=2;
fs = 128; T=3*fs;

Yl = (randn(N,1)>0)+1;     % True labels + 1
Y  = double([Yl==1 Yl==2]);% indicator N x L

% Single signal
sources = { {'sin' 5}; {'none' 1} };  % 1 source, rest detectors
y2mix=cat(3,[1],[.5]);      % ch x source x label
mix =y2mix(:,:,Yl);         % ch x source x N
[X,S]=mixSig(sources,mix,T);
[X,A,S,elect_loc]=mksfToy(sources,mix,T);

% Pure signal, diff freqs
sources = { {'sin' 5};   % inc amplitude label=2
            {'sin' 8};   % dec amplitude label=1
            {'none' 1}}; % rest just detectors
y2mix=cat(3,[1;.5],[.5;1]); % ch x source x label
mix  =y2mix(:,:,Yl);  % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T);

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
[X,A,S,elect_loc]=mksfToy(sources,mix,T);

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
y2mix=cat(3,[1;.5;nAmp],[.5;1;nAmp]); % ch x source x label
mix  =y2mix(:,:,Yl);  % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T,elect_loc);


% With some noise
sources = { {'sin' 5} {'coloredNoise' 1};
            {'sin' 8} {'coloredNoise' 1};
            {'none' 1} {}}; % rest just detectors
y2mix=cat(3,[1 1;.5 1],[.5 1;1 1]); % ch x source x label
mix  =y2mix(:,:,Yl);                % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T);

% With some noise + phase jitter
freqStd=0; phaseStd=pi/2; nAmp=5;
sources = { {'sin' 5 0 freqStd phaseStd} {'coloredNoise' 1};
            {'sin' 8 0 freqStd phaseStd} {'coloredNoise' 1};
            {'none' 1} {}}; % rest just detectors
y2mix=cat(3,[1 nAmp;.5 nAmp],[.5 nAmp;1 nAmp]); % ch x source x label
mix  =y2mix(:,:,Yl);                % ch x source x N
[X,A,S,elect_loc]=mksfToy(sources,mix,T);

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

