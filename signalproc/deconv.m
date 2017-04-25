function [X2]=deconv(X,y2s,irflen,dim,alg,stimDimy2s,verb,win)
% deconvolve the input X using the set of responses y2s
%
% [X]=deconv(X,y2s,irflen,dim,alg,stimdim,verb,win)
% Inputs:
%  X        -- [n-d] data matrix, T=size(X,dim(1))
%  y2s      -- [T x nStim] mapping between conditions and stimulus sequences
%              OR
%              [T x size(X,dim(2:end)) x nStim] mapping per dim(2:end) between conditions and sequences
%              OR
%              [T+irflen-1 x .... x nStim ] mapping which contians info about what happened 
%                   up to irflen-1 *before* the start of the data 
%  irflen -- [1x1] size in samples of Impluse Response to estimate
%  dim      -- [nd x 1] dimensions of X
%                   dim(1)     - dimension of X to deconvolve along
%                   dim(2:end) - sequential dim, i.e. we have a different y2s for each of these entries
%  alg      -- [str] algorithm to use, oneof- {'ls','ave'}                   ('ave')
%                tave, iave, nave -- as for ave but normalise the de-conv matrix by:
%                     t=sum over time, i=sum over tau, n=total sum, 
%                     z=#times this number is used in convolution
%  stimdim  -- [m x 1] dimensions of y2s which contain the stimulus info, i.e.
%                      which define different types of stimulus which should occur together ([])
%  verb     -- [int] verbosity level                                             (0)
% Outputs:
%  X        -- [size(X) x size(y2s,setdiff(1:end,dim(2:end)] with size(X,dim(1))=irflen
%                set of deconvolved time-series
%                with new dimensions for the new bits in y2s
%  M        -- [T x irflen x ... x nStim] deconvolution matrices per stim type
if( nargin<5 || isempty(alg) ) alg='ave'; end;
if( nargin<6 ) stimDimy2s=[]; end;
if( nargin<7 || isempty(verb) ) verb=0; end;
if( nargin<8 ) win=1; end;
if( isempty(strmatch(alg,{'ls','ave'})) && isempty(strmatch(alg(2:end),{'ls','ave'})) )
  error('Unrecognised decov alg: %s\n',alg); 
end;
tDimX=dim(1); tDimy2s=1;                     % time dimension, analiate this dim
stepDimX  =[];
if ( numel(dim)>1 ) % is there a step dim? 
  if( size(y2s,2)==size(X,dim(2)) ) 
    stepDimX  =dim(2:end)'; % step dim, i.e. solve independently for each of these
  elseif (size(y2s,2)==1)  % no step dim, reshape y2s if needed to match
    tmp=size(y2s);
    y2s=reshape(y2s,[tmp(1) tmp(numel(dim)+1:end) 1]);
  else
    warning('step dim in y2s and X dont agree.  Assuming this is a stim-dim');
  end
end
stepDimy2s=setdiff(2:ndims(y2s),stimDimy2s); % y2s can have more step-dims than X for diff stimTypes
% if ( isempty(stimDimy2s) ) 
%   stimDimy2s=numel(dim)+1:ndims(y2s);          % stim dimensions of 
% end
newstepDimX2   = (ndims(X)+stepDimy2s(numel(stepDimX)+1:end)-numel(stepDimX)-1);
stepDimX2 =[stepDimX newstepDimX2]; % where y2s step dims go in the new X2
newstimDimX2   = ndims(X)+stimDimy2s-numel(stepDimX)-1;
stimDimX2 = newstimDimX2;               % where stim go in the new X2

% convert from per-lab stimulus events into a deconvolution matrix
szy2s=size(y2s);
if ( szy2s(1)<size(X,dim(1)) ) % ensure size is right, 0 pad after with non-stimulus events
   if ( isa(y2s,'logical') ) 
      y2s= cat(1,y2s,false([max(0,size(X,dim(1))-size(y2s,1)),szy2s(2:end)]));
   else
      y2s= cat(1,y2s,zeros([max(0,size(X,dim(1))-size(y2s,1)),szy2s(2:end)],class(y2s)));
   end
 end
 
szy2s=size(y2s);szy2s(end+1:max([stepDimy2s(:);stimDimy2s(:)]))=1;
% index to select bits of y2s
y2sidx={}; for di=1:numel(szy2s); y2sidx{di}=int32(1:szy2s(di)); end;
if ( szy2s(1)>=size(X,dim(1))+irflen-1 ) % y2s has irflen-1 points of pre-data stimumulus info
  %irflen th sample in y2s is 1st sample of the data
  y2sidx{1}=int32(repop((1:size(X,dim(1)))','-',(0:irflen-1)))+irflen-1; % sel sub-sets of y2s  
else
  y2sidx{1}=int32(repop((1:size(X,dim(1)))','-',(0:irflen-1))); % sel sub-sets of y2s
  % 0-pad before with single non-stimulus event
  if ( isa(y2s,'logical') ) 
    y2s= cat(1,false([1,szy2s(2:end)]),y2s);
  else
    y2s= cat(1,zeros([1,szy2s(2:end)],class(y2s)),y2s);
  end   
  y2sidx{1}=max(1,y2sidx{1}+1); % update the indicies
end


szX = size(X);
% [szX x nStim]
X2  = zeros([szX(1:dim(1)-1) irflen szX(dim(1)+1:end) szy2s([stepDimy2s(numel(stepDimX)+1:end) stimDimy2s])],class(X)); 
xidx={}; for di=1:ndims(X); xidx{di}=int32(1:size(X,di)); end;
x2idx=cell(ndims(X2),1); x2idx(1:ndims(X))=xidx; x2idx{dim(1)}=1:irflen; % output assignment
nchnks=prod(szy2s(stepDimy2s)); % step over the sequential dims
if ( verb>0 ) fprintf('Deconv:'); end;
for yi=1:nchnks; % chunked de-conv
   if ( ~isempty(stepDimy2s) ) % map from linear to sub Idx
     [y2sidx{stepDimy2s}]      =ind2sub(szy2s(stepDimy2s),yi); 
   end  
   x2idx([stepDimX2 stimDimX2])=y2sidx([stepDimy2s stimDimy2s]); % which results to store
   xidx(stepDimX)              =x2idx(stepDimX);

   % make the convolution matrix 
   M = reshape(y2s(y2sidx{:}),[size(y2sidx{1}) szy2s(stimDimy2s)]); %[nSamp x irflen x stimDim(1...)]
   if ( ~isa(M,class(X)) )  M = feval(class(X),M); end;    %ensure M has compatiable type   
   % apply the window
   if ( ~isempty(win) ) M=repop(M,'*',win(:)); end
   if (isequal(alg(1),'n')) nf=sum(sum(M,1),2)./size(M,2);nf(nf==0)=1;M=repop(M,'/',nf); end; % normalise
   if (isequal(alg(1),'i')) nf=sum(M,2);       nf(nf==0)=1;M=repop(M,'/',nf); end; % normalise contribution to irf
   if (isequal(alg(1),'t')) nf=sum(M,1);       nf(nf==0)=1;M=repop(M,'/',nf); end; % normalise contribution to irf
   if (isequal(alg(1),'z')) 
     nf=sum(M,2);       nf(nf==0)=1;M=repop(M,'/',nf); % weight time points by # the convolve together
     if ( ~isempty(win) ) M=repop(M,'*',win(:)); end
     nf=sum(M,1);       nf(nf==0)=1;M=repop(M,'/',nf); % weight tau points are (weighted) average over time-points
   end; % normalise contribution to irf
   if ( strcmp(alg,'ls') || strcmp(alg,'nls') ) % use pseudo-inverse/lest-squares to deconv 
     M = reshape(pinv(reshape(M,size(y2sidx{1},1),[]))',size(M));
   end; 

   % apply it
   X2(x2idx{:})=tprod(X(xidx{:}),[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)],M,[-dim(1) dim(1) stimDimX2]);
   if( verb>0 ) textprogressbar(yi,nchnks); end;
end
if ( verb>0 ) fprintf('\n'); end;
return;
%--------------------------------------------------------------------------
function testCase()
[X,Y,M,y2s,xtrue,ptrue]=mkoverlapToy('nAmp',.05,'nCh',10,'nEpoch',1,'isi',20); Y=lab2ind(Y);
[X2]=deconv(X,y2s,32,2,'ls');

[X,Y,M,y2s,xtrue,ptrue]=mkoverlapToy('nAmp',.5,'nCh',10,'nEpoch',1000,'isi',20); Y=lab2ind(Y);
[X2]=deconv(X,y2s,32,2,'ls');
muX2=cat(3,mean(X2(:,:,Y(:,1)>0,:),3),mean(X2(:,:,Y(:,1)<0,:),3));
image3d(shiftdim(muX2,1),3,'colorbar',[],'disptype','plot')

[X2]=deconv(X,cat(1,zeros(32,2),y2s),32,2); % with pre-start stim info

[X2]=deconv(X,cat(3,y2s,double(y2s==0)),32,2); % with diff stim types

[X2]=deconv(X,repmat(reshape(y2s,size(y2s,1),1,size(y2s,2)),size(X,2)),32,[2 3]);%with per-epoch stims

y2sep=repmat(reshape(y2s,size(y2s,1),1,size(y2s,2)),size(X,2));
[X2]=deconv(X,cat(4,y2sep,double(y2sep==0)),32,[2 3]); % with per-epoch with diff types


nCh=1; nSrc=1; nSeq=100; irflen_samp=70; nAmp=.3; sAmp=1;  isi=20;
sigType = {'sum' {'cos' irflen_samp pi} {1}};% raised cos
noiseType={'coloredNoise',1}; 
codebook=[1 0 0 1 1 0 1 0 0;1 1 0 1 0 0 1 0 0]'; % fixed code which should have lots of overlap problems
nSamp = (size(codebook,1)*isi)+irflen_samp;
Y = (randn(nSeq,1)>0)+1; % [1 2]
% add the isi to the stim-sequ if wanted
stimTime_samp = ((0:size(codebook,1)-1)*isi)'+1; % record stim times
y2s=zeros(size(codebook,1)*isi+irflen_samp,size(codebook,2)); y2s(stimTime_samp,:) = codebook; % re-sample
[X,Y,M,y2s,xtrue,ptrue]=mkoverlapToy('sigType',sigType,'noiseType',noiseType,'irflen',irflen_samp,'y2s',y2s,...
                                     'nAmp',nAmp,'sAmp',sAmp,'nCh',nCh,'nSamp',nSamp,'Y',Y);

alg={'N' 'V' 'ave'}; 

dX=deconv(X,y2s,70,2,alg{end});

% with stim/non-stim decomp
y2s=zeros(size(codebook,1)*isi+irflen_samp,size(codebook,2),2); 
y2s(stimTime_samp,:,1) = codebook; y2s(stimTime_samp,:,2) = codebook==0;
dX=deconv(X,y2s,70,2,alg{end},3);

clf;image3d(shiftdim(cat(3,mean(dX(:,:,Y==1,:,:),3),mean(dX(:,:,Y==2,:,:),3))),2,'disptype','plot');



X_ep = windowData(X,stimTime_samp,irflen_samp,2); % [ch x time x win x symb=code ] 
y2s_ep=windowData(y2s,stimTime_samp-irflen_samp,size(X_ep,2)+irflen_samp,1); % [ time x win x code ]
if ( ~isempty(strmatch('W',alg)) ) % smoothing window to the data
  win  = mkFilter(size(X_ep,2),[1 10 size(X_ep,2)-[10 0]]-1);
  X_ep = repop(X_ep,'*',shiftdim(win,-1));
end
vin=[];
if ( ~isempty(strmatch('V',alg)) ) % window?
  vin  = mkFilter(size(X_ep,2),[1 35 size(X_ep,2)-[35 0]]-1);
end
if ( ~isempty(strmatch('N',alg)) )
  X_ep = repop(X_ep,'/',sqrt(msum(X_ep.^2,[1 2]))); % Normalise the power of the inputs
end; 
dX   = deconv(X_ep,y2s_ep,70,[2 3],alg{end},0,vin); % [ ch x time x win x symb x code ]  
clf;
col=0; clear p1 p2 p3
win  = 1:9; symb = 1; code = 1:2;
for symb=1:2;
  for code=1:2;
    col=col+1;
    p1(col)=subplott(3,4,(col-1)*3+1);plot(squeeze(X_ep(:,:,win,symb)));
    p2(col)=subplott(3,4,(col-1)*3+2);mcplot(squeeze(y2s_ep(:,win,code)));
    p3(col)=subplott(3,4,(col-1)*3+3);plot(squeeze(dX(:,:,win,symb,code)))
  end
end
suptitle(join('_',alg{:}));
%set(p3,'Ylim',[0 .01]);
