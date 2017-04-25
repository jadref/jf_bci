function [stimSeq,stimTime_ms,target,targetSeq,targetKey,dim]=jf_getstimSeq(z,varargin)
% extract the used stimulus sequence from an input problem
%
%  [stimSeq,stimTime_ms,target,targetSeq,targetKey,dim]=jf_getstimSeq(z,varargin)
%
% Outputs:
%   stimSeq     - [ seqLen x nClass (x nSeq)] stimulus sequence used for each class
%   stimTime_ms - [ seqLen x 1 ] time each stimulus occured in milliseconds
%   target      - [ nSeq x 1 ] target class for each sequence
%   targetSeq   - [ seqLen x nSeq ] stimulus sequence for each sequence for the target class
%   targetKey   - [ nClass x 1 ] name for each class
%   dim         - [ 2 x 1 ] dimensions of X used for this
opts=struct('dim',-1);
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim,0,0);
if ( dim==0 && n2d(z,'letter',0,0)>0 ) dim=n2d(z,'letter'); end;

% setup the new per-window class labels
stimTime_ms=[];
if ( isfield(z.di(dim).info,'stimSeq') ) % fixed stimSeq for all sequences
   stimSeq     = z.di(dim).info.stimSeq; % [nEvent x nSym]
	if ( isfield(z.di(dim).info,'stimTime_ms') )
     stimTime_ms = z.di(dim).info.stimTime_ms; % [nEvent x 1]
	end
    
elseif ( isfield(z.di(dim).extra,'stimSeq') ) % normal stim-sequ stuff
	stimSeq(:,:,1)     = z.di(n2d(z,dim)).extra(1).stimSeq;  % [nEvent x nSym x nSeq]
	for ei=1:numel(z.di(n2d(z,dim)).extra);
		 ss=z.di(n2d(z,dim)).extra(ei).stimSeq;
		 stimSeq(1:size(ss,1),:,ei) = ss; % allow for different numbers of events in different sequences
	    if ( isfield(z.di(dim).extra,'stimTime_ms') )
		     tmp{ei}=z.di(n2d(z,dim)).extra(ei).stimTime_ms(:);
       end
	end
	if ( isfield(z.di(dim).extra,'stimTime_ms') )
      % put the times into the times array, and mark up when we have different numbers of events
      stimTime_ms=zeros(size(stimSeq,1),1,size(stimSeq,3));
      for ei=1:numel(tmp);
         stimTime_ms(1:numel(tmp{ei}),1,ei)=tmp{ei};
         stimTime_ms(numel(tmp{ei})+1:end,1,ei)=NaN;
      end
   end
    
elseif ( isfield(z.di(dim).extra,'flipgrid') )
  dim         = n2d(z.di,'letter'); % per-letter deconv
  % Note: can have diff numbers of events for different letters... padd to longest length
  stimSeq     = [];
  stimTime_ms = [];
  for li=1:numel(z.di(dim(end)).extra);
     fli=z.di(dim(end)).extra(li).flipgrid; %[nSym x nEvent]
     if ( ~isempty(stimSeq) && size(fli,2)>size(stimSeq,1) ) % increased number of events
        stimTime_ms(end+1:size(fli,2),:)=NaN; % mark missing events
     end
     stimSeq(1:size(fli,2),1:size(fli,1),li)=fli';%[nEvent x nSym x nLet]
     if ( isfield(z.di(dim(end)).extra,'flashi_ms') )
        stimTime_ms(1:size(fli,2),1,li) = z.di(dim(end)).extra(li).flashi_ms(1:size(fli,2)); % [nEvent x 1 x nLet]
     elseif ( isfield(z.di(dim(end)).extra,'stimTime_ms') )
        stimTime_ms(1:size(fli,2),1,li) = z.di(dim(end)).extra(li).stimTime_ms(1:size(fli,2)); % [nEvent x 1 x nLet]
     end
     stimTime_ms(size(fli,2)+1:end,1,li)=NaN; % mark missing events     
  end
  % stimSeq     = cat(3,z.di(dim(end)).extra.flipgrid); %[nSym x nEvent x nLet](nLet=nSeq)
  % stimSeq     = permute(stimSeq,[2 1 3]); % permute to expected: [nEvent x nSym x nSeq]
  %stimTime_ms=reshape(stimTime_ms,[size(stimSeq,1),1,size(stimSeq,3)]); % expect: [nEvent x 1 x nSeq]
end
if ( isempty(stimTime_ms) ) % time not given, assume every sample
  stimTime_ms = z.di(n2d(z,'time')).vals(:); % every sample [nEvent x 1]
end

if ( nargout>2 ) 
  % select the target stims
  target=[];
  targetSeq=[];
  targetKey=[];

  if ( isfield(z.di(n2d(z,dim)).extra,'target') )
    target = cat(1,z.di(n2d(z,dim)).extra.target); % [nEpoch x 1]
	 if ( iscell(target) ) % try to convert into index into stimSeq
		if( isfield(z.di(n2d(z,dim)).info,'markerdict') ) % use marker dict
		  markerdict=z.di(n2d(z,dim)).info.markerdict;
		  targetIdx=zeros(size(target));
		  for ei=1:numel(target);
			 targetIdx(ei) = find(strcmp(target{ei},markerdict.label));
		  end
		  target=targetIdx;
		  targetKey=1:numel(markerdict.label);
		else
		  error('dont know how to deal with this dataset')
		end
	 end

  elseif ( isfield(z,'Y') ) % assume z.Y is already setup to index directly to the correct stimSeq
	 if ( size(z.Y,1)==size(stimSeq,1) ) % assume true-flash set Y=[nEvent x nEp x nSp]
		target=zeros(size(z.Y,2),1);
		for ei=1:size(z.Y,2); 
		  [ans,target(ei)] = max(z.Y(:,ei)'*stimSeq(:,:,ei)); 
		end
		targetKey=1:size(stimSeq,2);

	 % assume indicator set [nEp x nSp] or [nSp x nEp]
	 elseif ( size(z.Y,1)==size(stimSeq,3) || size(z.Y,2)==size(stimSeq,3) ) 
		target = z.Y;
		if ( isfield(z,'Ydi') && strcmp(z.Ydi(1).name,'subProb') ) % Y=[nSp x ...] convert => [nEp x nSp]
		  target=target';
		end
		if ( all(target(:)==-1 | target(:)==0 | target(:)==1 ) )  % assume indicator set [nEp x nSp]
		  if ( all(sum(z.Y>0,2)<=1) ) 
			 uY = eye(size(z.Y,2)); % 1vR encoding
		  else % unknown encoding
			 uY = unique(z.Y,'rows'); % [1xnSp]
		  end
		  target=zeros(size(z.Y,2),1);
		  for ei=1:size(z.Y,1); 
			 [ans,target(ei)] = max(z.Y(ei,:)*uY'); 
		  end
		end; 
		targetKey=unique(target);
	 end
  end

  if ( isfield(z.di(n2d(z,dim)).extra,'tgtSeq') )
     targetSeq = zeros(size(stimSeq,1),size(z.X,n2d(z,dim))); % [ nEvent x epoch ]
     for ei=1:numel(z.di(n2d(z,dim)).extra);
        tsei=z.di(n2d(z,dim)).extra(ei).tgtSeq;
        targetSeq(1:min(end,numel(tsei)),ei) = tsei;
     end
  end
  if ( isempty(targetSeq) ) 
     if ( ~isempty(target) ) 
        % get the target sequence from the key
        targetSeq = zeros(size(stimSeq,1),size(z.X,n2d(z,dim))); % [ nEvent x epoch ]
        for si=1:size(z.X,n2d(z,dim));
           targetSeq(:,si) = stimSeq(:,target(si),min(end,si));
        end
     else % only 1 target....
        target=ones(size(stimSeq,3),1);
        targetSeq=stimSeq;
     end
  else
     % HuH! got targetSeq but not target ID?
     % search for the target ID using the targetSeq
     tstgt=[];
     for si=1:size(z.X,n2d(z,dim));
        tgtSSip = single(targetSeq(:,si))'*stimSeq(:,:,min(end,si)); % similarity btw tgtSeq and each stimSeq
        [ans,tgtidx] = max(tgtSSip);
        tstgt(si) = tgtidx;
     end
     if ( isempty(target) ) target=tstgt;
     else 
        if ( ~isequal(target(:),tstgt(:)) ) % cross check the 2 ways of getting the target sequence
           warning('Marker target and target sequence information disagree!!!');
        end
     end
  end
end
return;
