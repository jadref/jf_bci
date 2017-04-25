function [W,dip]=invSrcLoc(X,A,srcPos,electPos,varargin)
% Compute inverse solution for the given data and fwd model (using fieldtrips forwinv codes)
%
% [W,dip]=invSrcLoc(X,A,srcPos,electPos,varargin)
%
% Inputs:
%  X -- [nElect x time x ... ] data matrix
%  A -- [nElect x nOrient x nSrc] lead-field matrix from orientated sources -> electrodes
%  srcPos   -- [ 3 x nSrc   ] source positions
%  electPos -- [ 3 x nElect ] electrode positions
% Options:
%  method -- [str] type of inverse method to used 'lcmv'
%  inside -- size(A) logical vector of srcPostions which are inside the brain
%  brain,skin,skull -- [str] file names of the brain/skin/skull tri-meshs
%                   OR [struct] structs containing the tri-meshs
%  isCov -- [bool] flag that X is already a covariance matrix
% Outputs:
%  W  -- [nElect x nOrient x nSrc ] inverse matrix
%  dip-- fieldtrip dipole fit structure
[cwd]=fileparts(mfilename('fullpath'));
opts=struct('method','lcmv','inside',[],...
            'brain',fullfile(cwd,'rsrc','brain.tri'),...
            'skin',fullfile(cwd,'rsrc','skin.tri'),...
            'skull',fullfile(cwd,'rsrc','skull.tri'),...
            'isCov',0);
%lcmvopts=struct('lambda','.1%');
[opts,varargin]=parseOpts(opts,varargin);

inside=opts.inside; if ( isempty(inside) ) inside=true(size(A,3),1); end;
outside=~inside;

nElect=size(A,1); nSrc=size(A,3);
if ( size(X,1)~=nElect )
   error('X doesnt match nElect from fwdMatrix');
end
if ( (~isempty(srcPos) && size(srcPos,2)~=nSrc) || (~isempty(electPos) && size(electPos,2)~=nElect) )
   error('fwdMx doesnt match electrode/source positions');
end

% compute a base-line-corrected covariance matrix
if ( ~opts.isCov )              Cxx = mcov(X,1)./size(X,2)./size(X,3); 
elseif ( size(X,1)==size(X,2) ) Cxx=X; X=zeros(size(Cxx,1),1); % dummy data
else error('X isnt a valid covariance matrix'); 
end;

% build dummy src/elect positions
if ( isempty(srcPos) ) srcPos=zeros(3,nSrc); end;
if ( isempty(electPos) ) electPos=zeros(3,nElect); end;

% ***N.B. src,vol are uncessary **if** we have pre-computed leadfield!
% N.B. inputs to the different methods. 
% src -- source information
%  src.pos -- positions of the srcs
%  src.inside -- logical vector of positions inside the skull
%  src.outside -- 
%  src.leadfield -- the pre-computed leadfield
for i=1:size(A,3); AA{i}=squeeze(A(:,:,i)); end; % convert to ft's cell array format
src=struct('pos',srcPos','inside',find(inside),'outside',find(outside),'leadfield',{AA});
% elect -- electrode information 
%  elect.pos -- electrode positions
elect=struct('pos',electPos');
% vol -- volume conductor information
%  vol.skin
%  vol.brain
%  vol.source
%  vol.mat -- pre-computed lead field?
vol=struct('mat',A,'skin',opts.skin,'brain',opts.brain,'skull',opts.skull);

switch(opts.method)
 case 'lcmv';
  dip = beamformer_lcmv(src, elect, vol, X, Cxx, 'keepfilter','yes',varargin{:});
 
 case 'pcc';
  dip = beamformer_pcc(src, elect, vol, X, Cxx, 'keepfilter','yes', varargin{:}, 'refdip', cfg.refdip, 'refchan', refchanindx, 'supchan', supchanindx);

 case 'mne';
  dip = minimumnormestimate(src, elect, vol, X, 'keepfilter','yes', varargin{:});

case 'loreta';
 dip = loreta(             src, elect, vol, X, 'keepfilter','yes', varargin{:});

case 'rv';
 dip = residualvariance(   src, elect, vol, X, 'keepfilter','yes', varargin{:});

case 'music';
 if hascovariance
    dip = music(src, elect, vol, X, 'cov', Cxx, 'keepfilter','yes', varargin{:});
 else
    dip = music(src, elect, vol, X,             'keepfilter','yes', varargin{:});
 end
 
otherwise;
    error(sprintf('method ''%s'' is unsupported for source reconstruction in the time domain', opts.method));
 end

% extract the inverse filter
W = zeros(3,nElect,nSrc); % % [ ori x elect x src ]
W(:,:,inside)=cat(3,dip.filter{:}); 
W = permute(W,[2 1 3]); % [elect x ori x src]
return
%--------------------------------------------------------------------------
function testCase();

addtopath('~/source/mmmcode/BCI_code/external_toolboxes/fieldtrip','.','private (you thought)','public');

% now compute the inverse
fwdMx=load('temp/fwdMx_64ch');
A=fwdMx.A;
A=repop(A,'./',sqrt(msum(A.^2,[1]))); % normalise fwdMx gains
clf;image3d(A,2,'xlabel','electrode','ylabel','orientation','zlabel','source')

A2= tprod(A,[-1 2 3],[],[-1 2 3]); % [1 x nOri x nSrc]
clf;imagesc(shiftdim(A2(:,:,1:end/3))); xlabel('source'); ylabel('orientation')

% N.B. the lcmv is **very** unstable because the inverse of ft'*ft is unstable
si=70;AA=tprod(A(:,:,si+(0:9)),[1 -2 3],[],[2 -2 3]);          % cov
si=70;for i=0:9; iAA(:,:,i+1)=pinv(A(:,:,si+i)*A(:,:,si+i)'); end; % inv
clf; for i=1:size(AA,3); 
   subplot(2,size(AA,3),i); imagesc(AA(:,:,i)); subplot(2,size(AA,3),i+size(AA,3)); imagesc(iAA(:,:,i));
end
set(findobj('type','axes'),'clim',[-1 1]*max(abs([AA(:);iAA(:)])));
saveaspdf('fwdMx+pinvfwdMx');

% inv with a perfect, i.e.uniform, source
[W,dip]=invSrcLoc(eye(size(A,1)),A,[],[],'iscov',1);
clf;image3d(W,2,'xlabel','electrode','ylabel','orientation','zlabel','source')
W2 = tprod(W,[-1 2 3],[],[-1 2 3]); % [1 x nOri x nSrc]
clf;imagesc(shiftdim(W2(:,:,1:end/3)));  xlabel('source'); ylabel('orientation')


clf;plot(shiftdim(A2)');
T=1000; nAmp=1;
N=randn(size(A,1),T); S=zeros(size(N)); 
X=S+nAmp*N;
[W,dip]=invSrcLoc(X,A,[],[]);%fwdMx.srcPos(:,fwdMx.inside>0),fwdMx.electPos);
W2 = tprod(W,[-1 2 3],[],[-1 2 3]); % [1 x nOri x nSrc]
clf;plot(shiftdim(W2)');

% apply the inversion to the signal
Sest = tprod(X,[-1 3],W,[-1 1 2]); % [ori x nSrc x time]

[cap.cnames cap.ll cap.xy cap.xyz]=readCapInf('cap64');
clf; jplot(cap.xy,[A(:,:,1) W(:,:,1)]);
% point in the center of the brain
ai=1281; %[a,ai]=min(sum(repop(srcPos,'-',mean(srcPos,1)).^2));
ai=1357; % point near C1
ai=1217; % point near c3
clf;scatPlot(srcPos,'.'); hold on; scatPlot(srcPos(:,[ai]),'r.','markersize',40); % plot src pos
clf; jplot(cap.xy,[A(:,:,ai) W(:,:,ai)]); % plot the fwdMx


% try with a signal embedded in the noise
% find the strongest response source location
fwdSrcPow=(msum(A.^2,[1 2]));
clf;plot(shiftdim(fwdSrcPow))
nAmp=1; T=1000;
sig=mkSig(T,'sin',40); % simu-source
N = randn(size(A,1),1000);     % back-ground noise
S = tprod(fwdMx.A(:,3,ai),[1 -2 -3],sig,[2 -2 -3],'n'); % projected signal
clf;mcplot(S'); % plot the source gain
X = S + nAmp*N;
clf;mcplot(X'); % plot the recieved signals
[W,dip]=invSrcLoc(X,A,[],[]);
W2 = tprod(W,[-1 2 3],[],[-1 2 3]); % [1 x nOri x nSrc]
clf;mcplot(shiftdim(W2)');

clf;jplot(cap.xy,[A(:,:,ai) W(:,:,ai)]);
% apply to the stuff
Sest = tprod(X,[-1 3],W,[-1 1 2]); % [ori x nSrc x time]
clf;mcplot(squeeze(Sest(:,ai,:))');

% plot a group of sources arround the true source
neari = find(sum(repop(srcPos,'-',srcPos(:,ai)).^2)<22.^2);
clf;mcplot(reshape(Sest(:,neari,:),[],size(Sest,3))');

SestPow = tprod(Sest(1,:,:),[-1 1 -3],[],[-1 1 -3]);
clf;plot(neari,SestPow(neari))
clf;
for i=1:numel(neari); 
   scatPlot(srcPos(:,neari(i)),'b.','markersize',SestPow(neari(i))./std(SestPow(neari))*3); hold on; 
end;