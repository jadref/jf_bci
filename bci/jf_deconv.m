function z=jf_deconv(z,varargin);
% apply devoncolution to the data
%
% Options:
%  irflen_samp/irflen_ms - [int] length of the impluse-response-function estimate in samples/milliseconds ([size(X,dim(1))/10])
%  dim - [str] dimension along which to deconv ('time')
%        dim(1)  - time dimension
%        dim(2:end) - sequence dimension - also y2s contains a per-element of these dimensions different stimulus sequences
%  alg  - [str] deconvolution algorithm to use, one-of 'ave','ls'   ('ave')
%  y2s  - [size(z.X,dim) x nClass x nStim] mapping between conditions and stimulus sequences
%              OR
%         [size(z.X,dim) x .... x nClass x nStim] mapping per dim(2:end) between conditions and sequences
%              OR
%         [size(z.X,dim)+irflen_samp x nClass x nStim] mapping which contains stim-events *before* start of the data..
%       N.B. If not given then y2s is extracted from either:
%             z.di(n2d(z,dim(2))).info -- if use same stimulus sequence/times for all epochs 
%            OR
%             z.di(n2d(z,dim(2))).extra(ei) -- if use a unique stimulus sequence/times for each epoch
%            In either case: 
%                    stimSeq -- [nEvents x nClass] gives the stimulus sequence, i.e. for each time at which a stimulus
%                                     event could happen it gives the stimulus type (e.g. flash/non-flash) 
%                                     for each condition (e.g. letter in the matrix speller), and 
%                    stimTime_ms -- [nEvents] gives the times of each stimulus event
%  y2sdi- [struct] dimension info struct describing y2s
%  zeroLab -- [bool] do we treat non-stimulus (stimStrength <1 and >-1) as a special type of stimulus?
%  latchStim -- [bool] do we treat samples between stimulus events as  (0)
%                  having the same stimulus, i.e. stim-remains-on until told it turns off
%  centerp -- [int] do we center stim-mapping (y2s) before deconvolving?  (false)
%                  N.B. this is useful as then we shouldn't respond to any DC offsets in the data
%                       also is then more about diff between stimulated and non-stimulated samples
%                       = sort of like a moving baseline
%  noPartIRF -- [bool] restrict cross-covariance to only time points for which we have data for
%               all the desired time lags
%  verb - [int] verbosity level
opts=struct('irflen_samp',[],'irflen_ms',[],'dim',{{'time' -1}},'stimDim','stim','alg','ave',...
            'y2s',[],'y2sdi',[],'zeroLab',0,'latchStim',0,'centerp',0,'noPartIRF',0,...
            'verb',1,'summary',[],'win',[],'subIdx',[]);
opts=parseOpts(opts,varargin);
if ( ischar(opts.dim) ) opts.dim={opts.dim}; end; 
%if ( numel(opts.dim)<2 ) if ( iscell(opts.dim) ) opts.dim{2}=-1; else opts.dim(2)=-1; end; end;
dim=n2d(z.di,opts.dim);
stimDim=[];

irflen = opts.irflen_samp;
if ( isempty(irflen) ) 
  if ( ~isempty(opts.irflen_ms) )
   % extract the sampling rate
   fs = getSampRate(z,'dim',dim(1));
   ms2samp = fs/1000; samp2ms = 1000/fs; 
   irflen = floor(opts.irflen_ms*ms2samp);
  else
    irflen = round(size(z.X,dim(1))/10); % default to 1/10 of the total time
  end
end
if ( irflen > size(z.X,dim(1)) )
  warning('irflen_samp greater than size of X! reduced');
  irflen=size(z.X,dim(1));
end

% extract the dim info of y2sdi if available
if ( ~isempty(opts.y2sdi) )
  if ( iscell(opts.y2sdi) ) opts.y2sdi=mkDimInfo(size(opts.y2s),opts.y2sdi); end;
   dim=n2d(z.di,{opts.y2sdi(1:end-1).name},0,0); % matched dims as sequential along in X
   dim(dim==0)=[]; % non-match is OK = new dims
   stimDim=n2d(opts.y2sdi,opts.stimDim,0,0); stimDim(stimDim==0)=[];
end
% extract the design matrix information we need (if not given)
y2s=opts.y2s; y2sdi=opts.y2sdi; stimKey=[];
if ( ~isempty(y2s) && opts.centerp>0 ) % center the modulator, for the stimulus events only
  y2s=repop(y2s,'-',mean(y2s,1));
elseif ( isempty(y2s) )
  if ( opts.verb>0 ) 
    fprintf('Building the modulator stream from the given stimSeq...');
  end
  % get the stimSeq
  [stimSeq,stimTime_ms]=jf_getstimSeq(z,'dim',dim(end));
  
  % combine these 2 to get the per-symbol stimulus matrix
  stimKey=unique(stimSeq(:)); 
  if ( ~opts.zeroLab ) stimKey(stimKey<1 & stimKey>-1)=[]; end;
  %[nSamp x nSeq x nSym x nStim]
  y2s = zeros([size(z.X,dim(1)),size(stimSeq,3),size(stimSeq,2),numel(stimKey)],'single');
  idx=[]; stimTimesi=[];
  for si=1:size(stimSeq,3);
    % get sample idx of these times
	 ostimTimesi=stimTimesi; stimTimesi = stimTime_ms(:,min(end,si));
    if ( isempty(idx) || ~all(stimTimesi==ostimTimesi) ) % cache index computation
		% limit to stim in the time-range of the data
		if ( opts.noPartIRF )
		  validStim = z.di(dim(1)).vals(1) <= stimTimesi & stimTimesi <= z.di(dim(1)).vals(end-irflen);
		else
		  validStim = z.di(dim(1)).vals(1) <= stimTimesi & stimTimesi <= z.di(dim(1)).vals(end);
		end
		% mark invalid any stimuli closer together than the sample rate
		tooClose = diff(stimTimesi)<median(diff(z.di(dim(1)).vals));
		if ( any(tooClose) ) validStim(tooClose)=false; end;
      idx=subsrefDimInfo(z.di(dim(1)),'vals',stimTimesi(validStim),'valmatch','nearest'); 
    end
    % put the markers in the right places %[nSamp x nSeq x nSym x nStim]
    for stimi=1:numel(stimKey);
      stimCode = stimSeq(validStim,:,si)==stimKey(stimi);
      if ( opts.centerp>0 ) % center the modulator, for the stimulus events only
        stimCode=repop(stimCode,'-',mean(stimCode,1));
      end    
      if ( ~opts.latchStim )        
        y2s(idx{:},si,:,stimi)=stimCode;
      else
        if ( islogical(idx{1}) ) idx{1}=find(idx{1}); end;
        for ei=1:numel(idx{1})-1;
          for sampi=idx{1}(ei):idx{1}(ei+1)-1;
            y2s(sampi,si,:,stimi)=stimCode(ei,:);
          end
        end
        srng=idx{1}(end):min(idx{1}(end)+mean(diff(find(idx{1}))),size(y2s,1));
        y2s(srng,si,:,stimi)=repmat(stimCode(end,:),numel(srng),1);
      end
    end
  end
  if ( opts.verb>0 ) fprintf('done.\n'); end
end

if( isempty(y2sdi) )
  if ( numel(dim)==2 ) 
    y2sdi = mkDimInfo(size(y2s),z.di(dim(1)).name,[],[],...
                      z.di(dim(2)).name,[],[],...
                      'symb',[],[],'stim',[],stimKey,'stim');
  else
    y2sdi = mkDimInfo(size(y2s),z.di(dim(1)).name,[],[],...
                      'symb',[],[],'stim',[],stimKey,'stim');
  end
end

% Pad y2s with enough extra samples for the final sets of responses
szy2s=size(y2s);
if ( isa(y2s,'logical') ) % deal with logical inputs
   y2s = cat(1,y2s,false([min(size(z.X,n2d(z.di,'time'))-szy2s(1),irflen),szy2s(2:end)]));
else
   y2s = cat(1,y2s,zeros([min(size(z.X,n2d(z.di,'time'))-szy2s(1),irflen),szy2s(2:end)],class(y2s)));
end

% call the function that does the work
z.X = deconv(z.X,y2s,irflen,dim,opts.alg,stimDim,opts.verb,opts.win);

z.di = [z.di([1:end-1]); y2sdi(numel(dim)+1:end-1); z.di(end)];
if ( isnumeric(z.di(dim(1)).vals) ) % get the new index values
  z.di(dim(1)).vals = (1:size(z.X,dim(1)))*median(diff(z.di(dim(1)).vals));
else
  z.di(dim(1)).vals = z.di(dim(1)).vals(1:size(z.X,dim(1)));
end
if ( numel(z.di(dim(1)).extra)>size(z.X,dim(1)) ) 
  z.di(dim(1)).extra= z.di(dim(1)).extra(1:size(z.X,dim(1))); 
end;
info=struct('y2s',y2s);%info=struct('M',M); if ( strcmp(opts.alg,'ls') ) info.pinvM=pinvM; end;
summary = sprintf('%s over %s',opts.alg,z.di(n2d(z.di,opts.dim(1))).name);% sprintf(' x %s',z.di(n2d(z.di,opts.dim(2:end))).name)]);
if ( ~isempty(opts.summary) ) summary=[summary ' (' opts.summary ')']; end;
z = jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------
function testCase()
addtopath('~/projects/bci','deconv','noise-tagging');
[X,Y,M,y2s]=mkoverlapToy('nEpoch',1000,'nAmp',.1);

di=mkDimInfo(size(X),'ch',[],[],'time','ms',[],'epoch');
di(3).info.markeri=struct('mod',num2cell(y2s,1),'name',{'1' '2'});
z=jf_import('overlap','test','toy',X,di,'Y',Y,'info',struct('y2s',y2s,'M',M));
oz=z;

z=jf_deconv(oz,'irflen_samp',32,'alg','ave');
image3d(shiftdim(z.X(:,:,1:30,:)),3,'colorbar',[],'disptype','mcplot')

% deconv with deconv for the non-stim sequence
z=jf_deconv(oz,'irflen_samp',32,'alg','ave','zeroLab',1);

muX2=cat(3,mean(z.X(:,:,z.Y(:,1)>0,:),3),mean(z.X(:,:,z.Y(:,1)<0,:),3));
image3d(muX2(:,:,:),3,'colorbar',[],'disptype','plott')
