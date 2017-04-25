function [z]=jf_rmEMG(z,varargin)
% use CSP/CCA to remove EMG from the data
%
% Automatically detect and remove emg artifacts, using the property that EMG has a 
% relatively large power but a very low auto-correlation, i.e. it has a flat spectrum, 
% which means it has very low power in the low frequencies compared to it's total power
%
% N.B. Uses *ALL* input channels - restrict to only BRAIN channels before use....
%
% Options:
%  dim - [2x1] dimensions over which to run the emg-removal  ({'ch' 'time'})
%          dim(1) - channel dimension
%          dim(2) - time dimension (used to compute the auto-covariance function)
%  tau_ms/tau_samp - time offset to use for computation of the auto-covariance (1000/60)
%  tol           - tolerance for detection of zero eigenvalues         (1e-7)
%  minCorr       - [1x1] minumum correlation value for detection of emg feature  (.2)
%  corrStdThresh - [1x1] threshold measured in std deviations for detection of   (2.5)
%                        anomylously small correlations, which is indicative of an emg
%                        channel
%  fs    - [1x1] sample rate of the data (used with tau_ms)
%  detectmethod - empty or 'corr' or 'freq': detection method for EMG components ([])
%  freqbands - [2x2] frequency ranges for eeg/emg ave power computation in format
%                    [eeg_low  emg_low;
%                     eeg_high emg_high]
%  freqRatio - frequency ratio to use in freq detection method (5)
% Outputs:
%  z.prep(end).info.WnE -- the spatial filter used to remove the EMG components
% 
opts=struct('dim',{{'ch' 'time'}},'tau_ms',[],'tau_samp',[],'fs',[],'tol',1e-7,...
				'minCorr',.2,'corrStdThresh',2,'detectmethod','corr','freqbands',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

% get dims to use
dim=n2d(z,opts.dim); if ( isempty(dim) ) dim=[1 2 3]; end;

% get time point to use
fs=opts.fs;   if( isempty(fs) ) fs  =getSampRate(z); end;
taus=opts.tau_samp;
if ( isempty(taus) && ~isempty(opts.tau_ms) )
  taus=opts.tau_ms;
  taus=round(taus/1000*fs);
else
  taus=1; % no other spec, then use 1 sample
end

% get the lag 0 and lag tau covariance matrix
taudim=[dim; setdiff((1:ndims(z.X))',dim)];
if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
  idx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
  cov=taucov(z.X(idx{:}),taudim,[0;taus(:)],'real','3d');
else
  cov=taucov(z.X,taudim,[0;taus(:)],'real','3d');
end
% do the CSP, i.e. solve generalised eigenvalue problem...
% compute whitener
[Uw,Dw]=eig(cov(:,:,1));Dw=diag(Dw);
siw=~(isinf(Dw) | isnan(Dw) | abs(Dw)<max(Dw)*opts.tol | Dw<opts.tol);
R=1./sqrt(abs(Dw));
W = repop(Uw(:,siw),'*',(1./sqrt(Dw(siw)))'); % whitening matrix
iW= repop(Uw(:,siw),'*',sqrt(Dw(siw))');      % inverse whitening matrix
% apply to lagged covariance
tcov=cov(:,:,2);
tcov=W'*tcov*W; % whitened lag tau covariance
% compute decomposotion of this matrix now
[Ue,De]=eig(tcov);De=diag(De);
%iPow   = abs(Ue)*Dw(si);
sie=~(isinf(De) | isnan(De) | abs(De)<max(abs(De))*opts.tol); % valid eigs

siemg=false(size(sie));
switch lower(opts.detectmethod);
  case 'corr'; % remove based on ratio of power at short-times to long-times
	 % identify the EMG components, either small value, or small rel to rest
	 for i=1:3; % iterative outlier detector
		mude = mean(abs(De(sie & ~siemg))); stdde=std(abs(De(sie & ~siemg)));
		siemg= sie & (abs(De)<=opts.minCorr | abs(De)<=mude-opts.corrStdThresh*stdde);
	 end;

  case 'freq'; % remove based on ratio of power in the two frequency ranges
    comp = tprod(z.X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],W*Ue,[-dim(1) dim(1)]);
    [comp specopts] = welchpsd(comp,dim(2),'fs',fs,'nwindows',1);
	 comp = mean(comp(:,:,:),3); % average over epochs and other extra dimensions
	 freqs=fftBins([],diff(z.di(dim(2)).vals([1 end]))/1000,fs,1);

    EMGfreqs=freqs>=opts.freqbands(3) & freqs<=opts.freqbands(4); 
	 EEGfreqs=freqs>=opts.freqbands(1) & freqs<=opts.freqbands(2); 
    EMGpow=mean(comp(:,EMGfreqs),2);
    EEGpow=mean(comp(:,EEGfreqs),2);

	 siemg(EMGpow*opts.freqRatio >= EEGpow)=true;	 		 
end
% get whitened space the mapping to the emg components
E = Ue(:,siemg);
%if ( any(siemg) )  E = repop(E,'*',sqrt(abs(De(siemg)))'); end
% apply the inverse whitening to get back to an input space version
WE= W*E;

% compute the spatial filter which removes the EMG components
% spatial filter to remove the identified components by
% whiten ---> remove EMG components ---> unwhiten
WnE= W*Ue(:,~siemg & sie)*Ue(:,~siemg & sie)'*iW'; 

% deflate the data with this set of spatial filters
emg=[];
if ( ~isempty(WE) ) 
  % estimate the emg signal(s)
  emg = tprod(z.X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(z.X)],WE,[-dim(1) dim(1)]);  
  % spatial filter to remove the EMG components
  z.X = tprod(z.X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(z.X)],WnE,[-dim(1) dim(1)]);
end

% update the meta-info
summary = sprintf('over %s',z.di(dim(1)).name);
summary = [summary sprintf(' rm %d est emgs',size(WE,2))];
info=struct('WnE',WnE,'W',W,'E',E,'WE',WE,'emg',emg,'Ue',Ue,'De',De,'Uw',Uw,'Dw',Dw);
z=jf_addprep(z,mfilename,summary,opts,info);
return;
%--------------------------------------------------------------------------------------
function testcase()
expt    = 'own_experiments/motor_imagery/movement_detection/trial_based/offline';
subjects= {'Jorgen'};
sessions= {{'20110114'}};
labels  ={'trn'}; 
oz=jf_load(expt,subjects{1},[labels{1} '_pp'],sessions{1}{1});

z=oz;
z=jf_retain(z,'dim','epoch','idx',[1:100]); % make smaller
zemg=jf_rmEMG(z);
% check if the removal has removed the bits we care about
figure(1);clf;jf_plotERP(jf_welchpsd(z));figure(2);clf;jf_plotERP(jf_welchpsd(zemg));

% try with the frequency based detection method
