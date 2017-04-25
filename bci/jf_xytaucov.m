function [z]=jf_xytaucov(z,varargin);
% compute covariance matrices -- N.B. *NOT* centered
%
%  [z]=jf_xytaucov(z,varargin);
%
% Options:
%  dim -- spec of the dimensions to compute covariances matrices
%         dim(1)=dimension to compute covariance over
%         dim(2)=dimension to sum along + compute delayed covariance, i.e. time
%         dim(3:end) - sequence dimension - also y2s contains a per-element of these dimensions different stimulus sequences
%  type   -- 'str' type of covariance to compute,                              ('X,XY')
%            one of:  'X,X'  data with itself only                 cov(tau)=[XX']
%                     'X,Y' data with labels only                  cov(tau)=[XY']
%                     'X,XY' data with data and labels             cov(tau)=[XX';XY']
%                     'XY,XY' data and labels with data and labels cov(tau)=[XX' XY';XY' YY']
%  outType-- type of covariance to compute, one-of  ('real')
%             'real' - pure real, 'complex' - complex, 'imag2cart' - complex converted to real
%  taus_samp/taus_ms-- [nTau x 1] set of sample offsets to compute cross covariance for
%         OR
%         [1x1] and <0, then use all values from 0:-taus
%  shape-- shape of output covariance to compute   ('3d')
%         one-of; '3d' - [d x d x nTau], '2d' - [d*nTau x d*nTau]
%  y2s/Y - [size(z.X,dim(2)) x nClass x nStim] mapping between conditions and stimulus sequences
%              OR
%         [size(z.X,dim(2)) x .... x nClass x nStim] mapping per dim(2:end) between conditions and sequences
%              OR
%         [size(z.X,dim(2))+irflen_samp x nClass x nStim] mapping which contains stim-events *before* start of the data..
%              OR
%         [struct] another data struct with similar shape as z which contains the targets for each
%                  epoch
%       N.B. If not given then y2s is extracted from either:
%             z.di(n2d(z,dim(2))).info -- if use same stimulus sequence/times for all epochs 
%            OR
%             z.di(n2d(z,dim(2))).extra(ei) -- if use a unique stimulus sequence/times for each epoch
%            In either case: 
%                    stimSeq -- [nEvents x nClass] gives the stimulus sequence, i.e. for each time at which a stimulus
%                                     event could happen it gives the stimulus type (e.g. flash/non-flash) 
%                                     for each condition (e.g. letter in the matrix speller), and 
%                    stimTime_ms -- [nEvents] gives the times of each stimulus event
%  y2sdi/Ydi- [struct] dimension info struct describing y2s
%  stimCode -- [bool] do we treat information in stimulus sequence as a 'code' for what *type*    (1)
%                     of stimulus happened at each event time?
%  zeroLab -- [bool] do we treat non-stimulus (stimStrength <1 and >-1) as a special type of stimulus?
%  interpType-- 'str' do we interpolate values between stimulus events? ('none')
%                  one-of: 'none'=no-interp, 'const'=piecewise constant, 'linear'=piecewise-linear
%  centerp -- [int] do we center stim-mapping (y2s) before deconvolving?  (false)
%                  N.B. this is useful as then we shouldn't respond to any DC offsets in the data
%                       also is then more about diff between stimulated and non-stimulated samples
%                       = sort of like a moving baseline
%  normalize -- one of 'none','mean','unbiased'                             ('noner')
%                postfix letter says how to deal with data from outside analysis window
%                'XXXXw'=pad with wrapped, 'XXXX0'=pad with 0, 'XXXXr'=pad with time-reversed
%  noPartIRF -- [bool] restrict cross-covariance to only time points for which we have data for
%               all the desired time lags
%  wght      -- [size(X,dim(2)) x 1] OR [size(Y)] 
%                 weighting over time-points **of X (by default)** when computing the cov-matrix. ([])
%                 such that cov(:,:,tau) = \sum_t wght(t) * x(:,t) y(:,t-tau)'
%                 N.B. 3d representation of the data does not correctly support weighting!
%  wghtX/wghtY-- [bool] if true then wght applys to X (or Y)                            (true,false)
%  bias    -- [bool] flag if we compute the bias response also            (false)
opts=struct('dim',{{'ch' 'time' -1}},'type','X,XY','outType','real','wght',[],...
				'taus_samp',[],'taus_ms',[],'irflen_samp',[],'irflen_ms',[],...
            'Y',[],'y2s',[],'y2sdi',[],'Ydi',[],...
				'stimCode',1,'zeroLab',0,'noPartIRF',0,...
				'interpType',[],'centerp',0,'bias',0,...
            'shape','3d','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% convert from names to dims
dim=n2d(z.di,opts.dim); nd=max(ndims(z.X),numel(z.di)-1);

taus=opts.taus_samp; if ( ~isempty(opts.irflen_samp) ) taus=opts.irflen_samp; end;
if ( isempty(taus) && (~isempty(opts.taus_ms) || ~isempty(opts.irflen_ms)) )
  taus = opts.taus_ms; if ( isempty(taus) ) taus=opts.irflen_ms; end;
  fs=getSampRate(z);
  taus = round(taus / 1000 * fs);
  if ( numel(unique(taus))<numel(taus) )
    warning('some duplicated taps removed');
    taus=unique(taus);
  end
end
if ( isempty(taus) ) taus=0; end;
if ( numel(taus)==1 && isnumeric(taus) && taus<0 ) taus=0:-taus; end;

%--------------------------------------------------------------------------------------------------
% extract the design matrix information we need (if not given)
Y=opts.y2s; if( isempty(Y) && ~isempty(opts.Y) ) Y=opts.Y; end;
Ydi=opts.y2sdi; if(isempty(Ydi) && ~isempty(opts.Ydi) ) Ydi=opts.Ydi; end; 
stimKey=[]; symKey=[];
% center the modulator, for the stimulus events only
if ( isnumeric(Y) && opts.centerp>0 ) 
  Y=repop(Y,'-',mean(Y,1));

elseif ( isstruct(Y) ) % data object to extract info from
  Ydi=Y.di;
  Y  =Y.X;

elseif ( isequal(Y,'Y') ) % use the Y for this object itself
  Ydi=z.Ydi;
  Y  =z.Y;

elseif ( isempty(Y) )
  if ( opts.verb>0 ) 
    fprintf('Building the modulator stream from the given stimSeq...');
  end

  % get the stimSeq
  [Y,Ydi]=stimSeq2regressor(z,'dim',dim(2:end),'stimCode',opts.stimCode,'zeroLab',opts.zeroLab,...
									 'noPartIRF',opts.noPartIRF,'interpType',opts.interpType,'centerp',opts.centerp);
  %[stimSeq,stimTime_ms,target,targetSeq,targetKey]=jf_getstimSeq(z,'dim',dim(end));
  
  %% % combine these 2 to get the per-symbol stimulus matrix
  %% if ( isfield(z.di(dim(end)).info,'markerdict') && ~isempty(z.di(dim(end)).info.markerdict)) 
  %% 	 symKey=z.di(dim(end)).info.markerdict.label; 
  %% else
  %% 	 symKey=1:size(stimSeq,2);
  %% end;
  %% stimKey=[]; % no key to convert code to stimulus
  %% if ( opts.stimCode ) % stimSeq is a stimCode, convert to true stimulus sequence for each stimType
  %% 	 stimKey=unique(stimSeq(:)); 
  %% 	 if ( ~opts.zeroLab ) stimKey(stimKey<1 & stimKey>-1)=[]; end;  
  %% end
  %% sampTimes=z.di(dim(2)).vals(:)'; % expect [1 x nSamp]
  %% % check if the data has been windowed along this dimension....
  %% if ( n2d(z,'window',0,0)>1 ) % use the pre-windowing sample times
  %% 	  warning('Data was windowed, attempting to compensate');
  %% 	  sampTimes=z.prep(m2p(z,'jf_windowData')).info.odi.vals(:)'; % [1 x nSamp]
  %% end
  %% %[nSamp x nSeq x nSym x nStim]
  %% Y = zeros([numel(sampTimes),size(stimSeq,3),size(stimSeq,2),numel(stimKey)],'single');
  %% idx=[]; stimTimesi=[];
  %% for si=1:size(stimSeq,3);
  %%   % get sample idx of these times
  %% 	 ostimTimesi=stimTimesi; stimTimesi = stimTime_ms(:,min(end,si));
  %%   if ( isempty(idx) || ~all(stimTimesi==ostimTimesi) ) % cache index computation
  %% 		% get sample idx of these times
  %% 		if ( opts.noPartIRF )
  %% 		  validStim = sampTimes(1+max([0;taus(:)])) <= stimTimesi & ...
  %% 						  stimTimesi <= sampTimes(end+min([0;taus(:)]));
  %% 		else
  %% 		  validStim = sampTimes(1) <= stimTimesi & stimTimesi <= sampTimes(end);
  %% 		end
  %% 		% mark invalid any stimuli closer together than the sample rate
  %% 		tooClose = diff(stimTimesi)<median(diff(sampTimes));
  %% 		if ( any(tooClose) ) validStim(tooClose)=false; end;
  %% 		[ans,idx] = min(abs(repop(sampTimes,'-',stimTimesi(validStim))),[],2); 
  %%   end
  %%   % put the markers in the right places %[nSamp x nSeq x nSym x nStim]
  %% 	 if ( isempty(stimKey) ) % use stimSeq directly
  %% 		Y(idx,si,:,1)=stimSeq(validStim,:,si);
  %% 	 else % convert from code to indicator for each stimulus type
  %% 		for stimi=1:numel(stimKey);
  %%       stimInd = stimSeq(validStim,:,si)==stimKey(stimi);
  %%       if ( opts.centerp>0 ) % center the modulator, for the stimulus events only
  %% 			 stimInd=repop(stimInd,'-',mean(stimInd,1));
  %%       end    
  %%       if ( ~opts.interpType || strcmp(opts.interpType,'none') ) 
  %% 			 Y(idx,si,:,stimi)=stimInd;
  %%       else
  %% 			 if ( islogical(idx) ) idx=find(idx); end;			 
  %% 			 for ei=1:numel(idx)-1;
  %% 				samps=idx(ei):idx(ei+1)-1;
  %% 				switch (opts.interpType) 
  %% 				  case {1,'constant'}; % piecewise constant
  %% 					 Y(samps,si,:,stimi)=stimInd(ei,:);
  %% 				  case {2,'linear'}; % linear interpolant
  %% 					 alpha = samps/numel(samps);
  %% 					 Y(samps,si,:,stimi)=alpha(:)*stimInd(ei,:)+(1-alpha(:))*stimInd(min(end,ei+1),:);
  %% 				  otherwise; error('Unrecognised interplotation method');
  %% 				end
  %% 			 end
  %% 			 srng=idx(end):min(idx(end)+mean(diff(find(idx))),size(Y,1));
  %% 			 Y(srng,si,:,stimi)=repmat(stimInd(end,:),numel(srng),1);
  %%       end
  %% 		end
  %% 	 end
  %% end
  % % reshape into shape that taucov wants, i.e. with 'channels'=symbols+stimuli first
  % Y=permute(Y(:,:,:),[3 1 2]); %[ (ch_y*stimi) x samp x nSeq ]
  % if ( opts.verb>0 ) fprintf('done.\n'); end  
end

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
   if ( numel(stimKey)>1 ) 
      yname='symb_stim';
      yvals=repmat(symKey(:),1,numel(stimKey));
   end
  if ( numel(dim)>=3 )
     if ( n2d(z,'window',0,0)>1 )
        Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(2)).name,[],[],...
                        z.di(n2d(z,'window')).name,[],[],...
                        z.di(dim(3)).name,[],[],'stim');
     else
        Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(2)).name,[],[],...
                        z.di(dim(3)).name,[],[],'stim');
    end
  else
    Ydi = mkDimInfo(size(Y),yname,[],yvals,z.di(dim(2)).name,[],[],...
                    'stim');
  end
end

%set the weighting over time-points if wanted
wght=opts.wght;
if ( ~isempty(wght) && any(strcmpi(wght,{'valid_ms','stimTime_ms'})) )% xxx says time-range to use
	fn=wght;
	wght=[];
  if( isfield(z.di(dim(end)).extra,fn) )
	 wght   = false(size(z.X,dim(2)),size(z.X,dim(end)));
	 for ei=1:numel(z.di(dim(end)).extra);
		valid_ms=z.di(dim(end)).extra(ei).(fn);		
		[ans,vstartIdx] = min(abs(z.di(dim(2)).vals-valid_ms(1))); % start at time==0
		[ans,vendIdx]   = min(abs(z.di(dim(2)).vals-valid_ms(end))); % end time for this epoch
		wght(vstartIdx:vendIdx,ei)=true;
	 end
  end
end

%--------------------------------------------------------------------------------------------
% call taucov to do the real work..
xxcov=[]; xycov=[]; yycov=[];
if ( any(strcmp(opts.type,{'XX','X,XY','XY,XY'})) ) % Need XX, [ch_x(+b) x ch_x x taus x ...]
  xxcov=xytaucov(z.X,[],dim(1:2),taus,'type',opts.outType,'shape',opts.shape,...
					  'bias',opts.bias,'wght',wght,varargin{:});
end
if ( any(strcmp(opts.type,{'XY','X,XY','XY,XY'})) ) % Need XY, [ch_x(+b) x ch_y x taus x ...]
  xycov=xytaucov(z.X,Y, dim(1:2),taus,'type',opts.outType,'shape',opts.shape,...
					  'bias',opts.bias,'wght',wght,varargin{:});
end
if ( any(strcmp(opts.type,{'YY','XY,XY'})) )        % Need YY, [ch_y(+b) x ch_y x taus x ...]
  yycov=xytaucov(Y,[],  dim(1:2),taus,'type',opts.outType,'shape',opts.shape,...
					  'bias',opts.bias,'wght',wght,varargin{:});	  
end
switch opts.type;
  case 'XX';   z.X  =xxcov; % [ch_x x ch_x x taus x ...]
  case 'XY';   z.X  =xycov; % [ch_x x ch_y x taus x ...]
  case 'YY';   z.X  =yycov; % [ch_y x ch_y x taus x ...]
  case 'X,XY'; z.X  =cat(dim(2),xxcov,xycov); % [ch(+b) x [ch_x;ch_y] x taus x ...]
  case 'XY,XY'; %[[ch_x;ch_y(+b)] x [ch_x;ch_y] x taus x ....]
	 z.X  =cat(dim(1),cat(dim(2),xxcov,xycov),... 
				  cat(dim(2),permute(xycov,[2 1 3:ndims(xycov)]),...
						repmat(yycov,[1 1 1 size(xxcov,4)/size(yycov,4)]))); 
  otherwise; error('Unrecognised type to compute');
end

clear xxcov xycov yycov;

%--------------------------------------------------------------------------------------------------
% update the meta-info
if ( strcmpi(opts.shape,'2d') )
  newDs=[setdiff(1:dim(1),dim(2)) dim(1) setdiff(dim(1)+1:nd,dim(2))];  % time -> new space*taus
else
  newDs=[setdiff(1:dim(1),dim(2)) dim(1) dim(2) setdiff(dim(1)+1:nd,dim(2))];
end
odi=z.di;
z.di=z.di([newDs end]);
nchD=find(newDs==dim(1),1,'first');
xdi=odi(nchD);
% add the y-info to the vals info
if ( isstruct(Ydi) ) 
  yvals=Ydi(1).vals; 
else
  yvals=1:size(Y,1); 
end;
ydi=mkDimInfo(numel(yvals),1,'y',[],yvals);

switch opts.type;
 case 'XX';   z.di(nchD)=xdi; z.di(nchD+1)=xdi; z.di(nchD+1).name=[xdi.name '_2'];
 case 'YY';   z.di(nchD)=ydi; z.di(nchD+1)=ydi; z.di(nchD+1).name=[ydi.name '_2'];
 case 'XY';   z.di(nchD)=xdi; z.di(nchD+1)=ydi; 
 case 'X,XY'; 
	z.di(nchD)  =xdi; 
	z.di(nchD+1)=xdi; z.di(nchD+1).name=[xdi.name '_' ydi.name '2'];
	if ( iscell(xdi.vals) )
	  if ( iscell(ydi.vals) ) z.di(nchD+1).vals = {xdi.vals{:} ydi.vals{:}}; 
	  else                    z.di(nchD+1).vals = {xdi.vals{:} num2cell(ydi.vals)};
	  end
	else
	  z.di(nchD+1).vals = cat(2,xdi.vals,ydi.vals);
	end
 case 'XY,XY';
	xydi=xdi; xydi.name=[xdi.name '_' ydi.name];
	if ( iscell(xdi.vals) )
	  if ( iscell(ydi.vals) ) xydi.vals = {xdi.vals{:} ydi.vals{:}}; 
	  else                    xydi.vals = {xdi.vals{:} num2cell(ydi.vals)};
	  end
	else
	  xydi.vals = cat(2,xdi.vals,ydi.vals);
	end
	z.di(nchD)=xydi; z.di(nchD+1)=xydi; z.di(nchD+1).name=[xydi.name '_2'];	
end
[z.d(nchD+1).extra.Y]=deal([zeros(size(z.X,nchD),1);ones(numel(yvals),1)]); % add flag for Y cov stuff
if (opts.bias) 
	if ( iscell(z.di(nchD).vals) )        z.di(nchD).vals{size(z.X,nchD)}='bias'; 
	elseif( isnumeric(z.di(nchD).vals) )  z.di(nchD).vals(size(z.X,nchD))=-1; 
	end
end
if ( strcmp(opts.outType,'imag2cart') && ~isreal(z.X) )   
  z.di(nchD).vals = repmat(z.di(nchD).vals,[1 2]);
  z.di(nchD+1).vals=repmat(z.di(nchD+1).vals,[1 2]);
end
if ( strcmp(opts.shape,'2d') )
  z.di(nchD).vals = repmat(z.di(nchD).vals,[1 numel(taus)]);
  z.di(nchD+1).vals=repmat(z.di(nchD+1).vals,[1 numel(taus)]);
else
  ntauD=find(newDs==dim(2),1,'first');
  if ( numel(opts.taus_ms)~=numel(taus) )
    z.di(ntauD)=mkDimInfo(numel(taus),1,'tau','samp',taus);
  else        
    z.di(ntauD)=mkDimInfo(numel(taus),1,'tau','ms',opts.taus_ms);
  end
end
if ( ~isempty(z.di(end).units) ) z.di(end).units=[z.di(end).units '^2'];end

% update the Y and Ydi to match the removal of dimensions from X if needed.
oY=z.Y; z.Y=[]; oYdi=[];
validY=true;
if ( strcmp(opts.Y,'Y') ) validY=false; end; % they are now not valid
if ( isfield(z,'Ydi') ) % check if invalid even if not used
   oYdi=z.Ydi;
   if( any(n2d(oYdi,odi(dim(2)).name,0,0)) ) % now invalid labels
      validY=n2d(oYdi,odi(dim(2)).name,0,0); % invalid dims of Y
   end
end
if( ~isequal(validY,true) )
   z.Y=[]; z.Ydi=[];
   if ( isfield(z,'foldIdxs') ) % set new folding up
      fsz=size(z.foldIdxs); 
      if( all(fsz(1:end-2)==1) ) 
         if ( validY>0 ) 
            Y2=oY; Y2(isnan(Y2))=0; Y2=sum(Y2.*Y2,validY); % new Y is sum squared old Y
            szY=size(Y2); Y2=reshape(Y2,szY(setdiff(1:end,validY)));
            z.Y       =Y2;
            z.Ydi     =oYdi(setdiff(1:numel(oYdi),validY));
            z.foldIdxs=reshape(z.foldIdxs,fsz(setdiff(1:end,validY)));
         else
            z.foldIdxs=reshape(z.foldIdxs,fsz(end-1:end)); % keep folding
            z.Y       =ones(fsz(end-1),1); % fake-Y
            z.Ydi     =mkDimInfo(size(z.Y),odi(dim(3)).name,[],[],'subProb',[],[]); % fake-meta-info
         end
      else
         warning('xytaucov invalidated Y and folding... removed');
      end
   end
end

summary=sprintf('over %ss',odi(dim(1)).name);
if( ~strcmp(opts.outType,'real') ) summary = [opts.outType ' ' summary]; end;
if(numel(dim)>1) summary=[summary ' x (' sprintf('%s ',odi(dim(2:end)).name) ')'];end 
summary=[summary sprintf(' %d taus',numel(taus))];
info=struct('oY',oY,'oYdi',oYdi);
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function testCase()
z=jf_mkoverlapToy('nCh',10,'nEpoch',100,'nSamp',128,'irflen',64,...
                  'nAmp',.1,'nSrc',10,'sAmp',1);   
zt=jf_xytaucov(z,'taus_samp',[0:3]);
zt=jf_xytaucov(z,'taus_samp',[0:3],'type','XX');
zt=jf_xytaucov(z,'taus_samp',[0:3],'type','XY');
zt=jf_xytaucov(z,'taus_samp',[0:3],'type','X,XY');
zt=jf_xytaucov(z,'taus_samp',[0:3],'type','YY');
zt=jf_xytaucov(z,'taus_samp',[0:3],'type','XY,XY');
zt=jf_xytaucov(z,'Y',z,'taus_samp',[0:3],'type','XY,XY','bias',0);
zt=jf_xytaucov(z,'Y',z,'taus_samp',[0:3],'type','XY','bias',1);

