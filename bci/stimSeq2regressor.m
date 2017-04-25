function [Y,Ydi,target,targetSeq]=stimSeq2regressor(z,varargin);
% convert a stimulus Sequence to a regression target
%
%
% Options:
%  dim - dimensions along which stimulus applies
%  stimCode
%  zeroLab - treat stimulus=0 as a true stimulus event
%  noPartIRF - [bool]
%  interpType -
%  centerp   - [bool] zero mean the stimulus sequence
%  compressSymbStim - [bool] compress stimType and stimCode into 1 dimension (1)
opts=struct('dim',{{'time' -1}},'stimCode',1,'compressSymbStim',1,'zeroLab',0,'noPartIRF',0,...
				'interpType',[],'centerp',0,'stimPostProc',[],...
			  'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% get the stimSeq
dim=opts.dim; dim=n2d(z,opts.dim);
[stimSeq,stimTime_ms,target,targetSeq,targetKey,epdim]=jf_getstimSeq(z,'dim',dim(end),varargin{:});

% apply any post-processing we want
if ( ~isempty(opts.stimPostProc) )
   dss = diff(stimSeq,1);
   dts = diff(targetSeq,1);
   switch ( opts.stimPostProc ) 
     case 're';               stimSeq(2:end,:)=dss(:,:)>0;    targetSeq(2:end,:)=dts(:,:)>0;    % rising edges only
                              stimSeq(1,:)=0;                 targetSeq(1,:)=0;   % fix startup/end effects

     case 'fe';               stimSeq(2:end,:)=dss(:,:)<0;    targetSeq(2:end,:)=dts(:,:)<0;  % falling edges only
                              stimSeq(1,:)=0;                 targetSeq(1,:)=0;   % fix startup/end effects

     case {'rfe','fre'};      stimSeq(2:end,:)=dss(:,:)~=0;   targetSeq(2:end,:)=dts(:,:)~=0; % rising or falling edges
                              stimSeq(1,:)=0;                 targetSeq(1,:)=0;   % fix startup/end effects

     case 'diff';             stimSeq(2:end,:)=dss;           targetSeq(2:end,:)=dts(:,:);    % gradient of the stimulus    
                              stimSeq(1,:)=0;                 targetSeq(1,:)=0;   % fix startup/end effects

     case 'diffdiff';         stimSeq(2:end-1,:)=diff(stimSeq,2,1);  stimSeq([1 end],:)=0; %2nd gradient of the stimulus
                              targetSeq(2:end-1,:)=diff(targetSeq,2,1); targetSeq([1 end],:)=0;

     case 'max';              stimSeq(2:end-1,:)  = dss(1:end-1,:)>0  & dss(2:end,:)<=0; % local max
                              targetSeq(2:end-1,:)= dts(1:end-1,:)>0  & dts(2:end,:)<=0; 
                              stimSeq([1 end],:)=0; targetSeq([1 end],:)=0; % fix startup/end effects

     case 'min';              stimSeq(2:end-1,:)  = dss(1:end-1,:)<=0 & dss(2:end,:)>0;  % local min
                              targetSeq(2:end-1,:)= dts(1:end-1,:)<=0 & dts(2:end,:)>0;  
                              stimSeq([1 end],:)=0; targetSeq([1 end],:)=0; % fix startup/end effects

     case 'minmax';           stimSeq(2:end-1,:)  = (dss(1:end-1,:)>0 & dss(2:end,:)<=0) | (dss(1:end-1,:)<=0 & dss(2:end,:)>0);
                              targetSeq(2:end-1,:)= (dts(1:end-1,:)>0 & dts(2:end,:)<=0) | (dts(1:end-1,:)<=0 & dts(2:end,:)>0);
                              stimSeq([1 end],:)=0; targetSeq([1 end],:)=0; % fix startup/end effects

     otherwise; warning('Unrecognised post-proc type');
   end
end

stimKey=[]; symKey=[];
% combine these 2 to get the per-symbol stimulus matrix
if ( isfield(z.di(dim(end)).info,'markerdict') && ~isempty(z.di(dim(end)).info.markerdict)) 
  symKey=z.di(dim(end)).info.markerdict.label; 
else
  symKey=1:size(stimSeq,2);
end;
stimKey=[]; % no key to convert code to stimulus
if ( opts.stimCode ) % stimSeq is a stimCode, convert to true stimulus sequence for each stimType
  stimKey=unique(stimSeq(:)); 
  if ( numel(stimKey)>20 ) 
     warning('%d different stimulus values detected..... treating as continuous',numel(stimKey));
     stimKey=[];
  end
  if ( ~opts.zeroLab ) stimKey(stimKey<1 & stimKey>-1)=[]; end;  
end
sampTimes=z.di(dim(1)).vals(:)'; % expect [1 x nSamp]
% check if the data has been windowed along this dimension....
if ( n2d(z,'window',0,0)>1 ) % use the pre-windowing sample times
  warning('Data was windowed, attempting to compensate');
  sampTimes=z.prep(m2p(z,'jf_windowData')).info.odi.vals(:)'; % [1 x nSamp]
end
%[nSamp x nSeq x nSym x nStim]
Y = zeros([numel(sampTimes),size(stimSeq,3),size(stimSeq,2),numel(stimKey)],'single');
Ydi=[];
idx=[]; stimTimesi=[];
for si=1:size(stimSeq,3);
  % get sample idx of these times
  ostimTimesi=stimTimesi; stimTimesi = stimTime_ms(:,min(end,si));
  if ( isempty(idx) || ~all(stimTimesi==ostimTimesi) ) % cache index computation
	 % get sample idx of these times
	 if ( opts.noPartIRF )
		validStim = sampTimes(1+max([0;taus(:)])) <= stimTimesi & ...
						stimTimesi <= sampTimes(end+min([0;taus(:)]));
	 else
		validStim = sampTimes(1) <= stimTimesi & stimTimesi <= sampTimes(end);
	 end
	 % mark invalid any stimuli closer together than the sample rate
	 tooClose = diff(stimTimesi)<median(diff(sampTimes));
	 if ( any(tooClose) ) validStim(tooClose)=false; end;
	 [ans,idx] = min(abs(repop(sampTimes,'-',stimTimesi(validStim))),[],2); 
  end
  % put the markers in the right places %[nSamp x nSeq x nSym x nStim]
  nStims=max(1,numel(stimKey));
  for stimi=1:nStims;
	 if ( isempty(stimKey) ) % use stimSeq directly
		stimInd = stimSeq(validStim,:,si);
	 else % convert from code to indicator for each stimulus type
      stimInd = stimSeq(validStim,:,si)==stimKey(stimi);
	 end
    if ( opts.centerp>0 ) % center the modulator, for the stimulus events only
		stimInd=repop(stimInd,'-',mean(stimInd,1));
    end    
    if ( isempty(opts.interpType) || isequal(opts.interpType,0) || strcmp(opts.interpType,'none') ) 
		Y(idx,si,:,stimi)=stimInd;
    else
		if ( islogical(idx) ) idx=find(idx); end;			 
		for ei=1:numel(idx)-1;
		  samps=idx(ei):idx(ei+1);
		  switch (opts.interpType) 
			 case {1,'constant','const'}; % piecewise constant
				Y(samps,si,:,stimi)=repmat(stimInd(ei,:),numel(samps),1);
			 case {2,'linear','lin'}; % linear interpolant
				alpha = linspace(0,1,numel(samps));
				Y(samps,si,:,stimi)=(1-alpha(:))*stimInd(ei,:)+alpha(:)*stimInd(min(end,ei+1),:);
			 otherwise; error('Unrecognised interplotation method');
		  end
		end
		srng=idx(end):min(idx(end)+mean(diff(find(idx))),size(Y,1));
		Y(srng,si,:,stimi)=repmat(stimInd(end,:),numel(srng),1);
    end
  end
end
% reshape into shape that taucov wants, i.e. with 'channels'=symbols+stimuli first
if ( opts.compressSymbStim ) 
  Y=permute(Y(:,:,:),[3 1 2]); %[ (ch_y*stimi) x samp x nSeq ]
end
if ( opts.verb>0 ) fprintf('done.\n'); end  

% extract the dim info of Ydi if available
if ( ~isempty(Ydi) )
  if ( iscell(Ydi) ) Ydi=mkDimInfo(size(Y),Ydi); end;
  if ( isempty(dim)  || numel(dim)==2 )
	 dim=[dim n2d(z.di,{Ydi(2:end-1).name},0,0)]; % matched dims as sequential along in X
	 dim(dim==0)=[]; % non-match is OK = new dims
  end
end
if( isempty(Ydi) )
  yname='symb'; yvals=symKey;
  if ( opts.compressSymbStim )
  if ( numel(stimKey)>1 ) 
    yname='symb_stim';
    yvals=repmat(symKey(:),1,numel(stimKey));
  end
  if ( numel(dim)>=2 )
    if ( n2d(z,'window',0,0)>1 )
      Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(1)).name,[],[],...
                      z.di(n2d(z,'window')).name,[],[],...
                      z.di(dim(2)).name,[],[],'stim');
     else
       Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(1)).name,[],[],...
                       z.di(dim(2)).name,[],[],'stim');
    end
  else
    Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(1)).name,[],[],...
                    'stim');
  end
  else
	 if ( numel(dim)==1 )
		Ydi = mkDimInfo(size(Y),z.di(dim(1)).name,[],[],yname,[],yvals,'stim',[],stimKey);
	 else
		Ydi = mkDimInfo(size(Y),z.di(dim(1)).name,[],[],z.di(dim(2)).name,[],[],yname,[],yvals,'stim',[],stimKey);
	 end
  end
end
