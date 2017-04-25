function [X,emg,WE,W,E]=rmEMG(X,varargin)
% use CSP/CCA to remove EMG from the data
%
%  [X,emg,WE,W,E]=rmEMG(X,varargin)
%
% Automatically detect and remove emg artifacts, using the property that EMG has a 
% relatively large power but a very low auto-correlation, i.e. it has a flat spectrum, 
% which means it has very low power in the low frequencies compared to it's total power
%
% N.B. Uses *ALL* input channels - restrict to only BRAIN channels before use....
%
% Inputs:
%  X  - [n-d] input data
% Options:
%  dim - [2x1] dimensions over which to run the emg-removal  ([1 2])
%          dim(1) - channel dimension
%          dim(2) - time dimension (used to compute the auto-covariance function)
%  tau_samp - time offset to use for computation of the auto-covariance (1)
%  tol           - tolerance for detection of zero eigenvalues          (1e-7)
%  minCorr       - [1x1] minumum correlation value for detection of emg feature  (.2)
%  corrStdThresh - [1x1] threshold measured in std deviations for detection of   (2.5)
%                        anomylously small correlations, which is indicative of an emg
%                        channel
%
% Outputs:
%  X - [n-d] input with emg removed
%  emg - [n-d] estimated EMG signal
%  WE  - spatial filter used to get the EMG signal as: emg=WE*X;
%  W   - whitener
%  E   - mapping from white to EMG space
opts=struct('dim',[1 2],'tau_samp',1,'tol',1e-7,'minCorr',.2,'corrStdThresh',2.5);
opts=parseOpts(opts,varargin);

% get dims to use
dim=opts.dim; if ( isempty(dim) ) dim=[1 2]; end;

% get time point to use
taus=opts.tau_samp;

% get the lag 0 and lag tau covariance matrix
taudim=[dim; setdiff(1:ndims(X),dim)];
cov=taucov(X,taudim,[0;taus(:)],'real','3d');

% do the CSP, i.e. solve generalised eigenvalue problem...
% compute whitener
[Uw,Dw]=eig(cov(:,:,1));Dw=diag(Dw);
si=~(isinf(Dw) | isnan(Dw) | abs(Dw)<max(Dw)*opts.tol | Dw<opts.tol);
R=1./sqrt(abs(Dw));
W = repop(Uw(:,si),'*',(1./sqrt(Dw(si)))'); % whitening matrix
iW= repop(Uw(:,si),'*',sqrt(Dw(si))');      % inverse whitening matrix
% apply to laged covariance
tcov=cov(:,:,2);
tcov=W'*tcov*W; % whitened lag tau covariance
% compute decomposition of this matrix now
[Ue,De]=eig(tcov);De=diag(De);
%iPow   = abs(Ue)*Dw(si);
si=~(isinf(De) | isnan(De) | abs(De)<max(abs(De))*opts.tol); % valid eigs
% identify the EMG components, either small value, or small rel to rest
siemg=false(size(si));
for i=1:3; % iterative outlier detector
  mude = mean(abs(De(si & ~siemg))); stdde=std(abs(De(si & ~siemg)));
  siemg= si & (abs(De)<=opts.minCorr | abs(De)<=mude-opts.corrStdThresh*stdde);
end;
% get whitened space the mapping to the emg components
E = Ue(:,siemg);
if ( any(siemg) )  E = repop(E,'*',sqrt(abs(De(siemg)))'); end
% apply the inverse whitening to get back to an input space version
WE= iW*E;
% ensure it has unit norm
WE= repop(WE,'/',sqrt(sum(WE.^2)));
% deflate the data with this set of spatial filters
emg=[];
if ( ~isempty(WE) ) 
  % estimate the emg signal(s)
  emg = tprod(X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],WE,[-dim(1) dim(1)]);
  % deflate data with these signals
  X = X - tprod(emg,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],WE,[dim(1) -dim(1)]);
end

return;
%--------------------------------------------------------------------------------------
function testcase()
rawX=cumsum(randn(10,1000),2); % sim EEG: rand signal with 1/f spectrum, i.e. lots low-freq power
emg =rand(1,size(X,2))*10; % sim EMG: pure random noise signal
Aemg=linspace(-1,1,10)'; % spatial filter for the EMG
X   =rawX+Aemg*emg; % simulated mixed signal with EEG + EMG
[Xa,emga,WE]=rmEMG(X);
clf;plot([emg;emga]')
clf;for i=1:size(rawX,1); subplot(10,1,i); plot([rawX(i,:);X(i,:);Xa(i,:)]'); legend('raw','raw+emg','raw-emg');end;clickplots
clf;plot([sum(rawX.*X,2) sum(rawX.*Xa,2)])

corr_rawX =(rawX*X')./sqrt(sum(rawX.^2,2)*sum(X.^2,2)');
corr_rawXa=(rawX*Xa')./sqrt(sum(rawX.^2,2)*sum(Xa.^2,2)');
clf;mimage(corr_rawX,corr_rawXa,'colorbar',1);colormap ikelvin
