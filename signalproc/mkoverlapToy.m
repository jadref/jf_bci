function [X,Yl,M,y2s,xtrue,ptrue,opts]=mkoverlapToy(varargin)
% make a toy signal produced as a set of overlapping responses
%
% [X,Yl,M,y2s,xtrue,ptrue,opts]=mkoverlapToy(varargin)
%
% Options:
%  nCh       -- [1x1] number of channels
%  nSrc      -- [1x1] number of source locations
%  nEpoch    -- [1x1] number of epochs
%  nClass    -- [1x1] number of classes
%  nSamp     -- [1x1] number of samples for each epoch
%  irflen    -- [1x1] number of samples for the stimulus response
%  sAmp      -- [1x1] or [nCh x 1] signal strength (for each channel)
%  nAmp      -- [1x1] or [nCh x 1] noise strength (for each channel)
%  y2s       -- [nSamp x nClass] per class stimilus matrix
%  sigType   -- IRF signal type as parameters to mkSig
%  noiseType -- noise signal as parameters to mkSig
%  Y  -- [nEpoch x 1] the labels for the simulated data
% Outputs:
%  X  -- [nCh x nSamp x nEpoch] the simulated data
%  Y  -- [nEpoch x 1] the labels for the simulated data
%  M  -- [nSamp x irflen x nClass] the mapping from labels to stimulus types at each time
%  y2s  -- [nSamp+1 x nClass] per class stimilus matrix
%  xtrue -- [nSamp x nClass] the true responses without noise
%  ptrue -- [irflen x 1] the true stimulus response
opts = struct('nCh',1,'nSrc',1,'nEpoch',300,'sAmp',1,'nAmp',1,...
              'noiseType',{{'coloredNoise',1}},'isi',[],'Y',[]);
mkconvOpts=struct('nClass',2,'nSamp',100,'irflen',32,'y2s',[],'sigType',{{'prod' {'exp' .2} {'sin' 5}}},'unitnorm',1);
[opts,mkconvOpts]=parseOpts({opts,mkconvOpts},varargin);
if ( opts.nCh==1 && numel(opts.sAmp)>1 ) opts.nCh=numel(opts.sAmp); end;
if ( ~iscell(mkconvOpts.sigType) ) mkconvOpts.sigType={mkconvOpts.sigType}; end;

% convolve the IRF with the stim sequence to make the convolved signal
[M,y2s,xtrue,ptrue,mkconvOpts]=mkConvSig(mkconvOpts);

% source shape is given by the mixed signal xtrue
sources=num2cell(xtrue,1); 
for i=1:numel(sources); if(~iscell(sources{i}))sources{i}={sources{i}}; end;end; % convert to cell
sources{end+1}=opts.noiseType;   % add the noise source 1ch with source with nClass+1 sigs to mix
y2mix=zeros(1,size(sources,2),size(xtrue,2)); % ch x source x label
y2mix(:,end,:)=opts.nAmp;
sAmp=opts.sAmp; sAmp(end+1:size(xtrue,2))=sAmp(end);
y2mix(:,1:end-1,:)=repop(eye(size(xtrue,2)),'*',sAmp(1:size(xtrue,2))); % 1 source for each label with sAmp size

% mix in the noise, and project to the electrode space if wanted
Yl = opts.Y;
if ( isempty(Yl) ) Yl   = ceil(rand(opts.nEpoch,1)*size(xtrue,2)); end; % True labels for each epoch
mix  = y2mix(:,:,Yl);
if ( opts.nCh > 1 ) % make some virtual channels
  if ( opts.nSrc > 1 ) % make 2nd source signal with only the *noise* component active
    sources(2,:)=sources(1,:); % copy source with signal+noise components
    for si=1:size(sources,2)-1; sources{2,si}={}; end; % clear all but the noise component
    mix = cat(1,mix,mix(end,:,:));
    mix(end,1:end-1,:)=0; % ensure mark other sources as inactive
  end;
  [X,A,S,src,dest]=mksfToy(sources,mix,mkconvOpts.nSamp,opts.nSrc,opts.nCh);
else % just mix the signals together
  [X,S]= mixSig(sources,mix,mkconvOpts.nSamp);
end

return;
%--------------------------------------------------------------------------
function []=testCase()
[X,Yl,M,y2s,xtrue,ptrue,opts]=mkoverlapToy(); Y=lab2ind(Yl,[-1 1],[],[],0);

w=2;h=4;
figure(1);
clf;
subplot(h,w,1+[0:3-1]*w);
imagesc(shiftdim(X(1,:,Y(:,1)>0))');ylabel('epochs');title('Class 1');
subplot(h,w,2+[0:3-1]*w);
imagesc(shiftdim(X(1,:,Y(:,2)>0))');ylabel('epochs');title('Class 2');
subplot(h,2,1+[4-1  ]*w);
plot(y2s(:,1),'b');hold on; plot(xtrue(:,1),'g');plot(mean(shiftdim(X(1,:,Y(:,1)>0)),2),'r');
xlabel('Samples');
subplot(h,2,2+[4-1  ]*w);
plot(y2s(:,2),'b');hold on; plot(xtrue(:,2),'g');plot(mean(shiftdim(X(1,:,Y(:,2)>0)),2),'r');
xlabel('Samples');
legend('true bit sequ','true response','mean response');

return;

