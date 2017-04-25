function z=jf_seq2epoch(z,varargin)
opts=struct('irflen_ms',[],'irflen_samp',[],'dim',{{'time' 'epoch'}},'nEpoch',[],'offset_ms',[],'stimPostProc',[]);
[opts,varargin]=parseOpts(opts,varargin);
dim=n2d(z,opts.dim,0,0);
if (dim(2)==0 && n2d(z,'letter',0,0)>0 ) opts.dim(2)={'letter'}; dim(2)=n2d(z,'letter'); end;
% setup the new per-window class labels
[stimSeq,stimTime_ms,target,targetSeq,targetKey]=jf_getstimSeq(z);

if ( ~isempty(opts.stimPostProc) )
   dss = diff(stimSeq,1);
   dts = diff(targetSeq,1);
   switch ( opts.stimPostProc ) 
     case 're';               stimSeq(2:end,:)=dss(:,:)>0;    stimSeq(1,:)=0;     % rising edges only
                              targetSeq(2:end,:)=dts(:,:)>0;  targetSeq(1,:)=0;

     case 'fe';               stimSeq(2:end,:)=dss(:,:)<0;    stimSeq(1,:)=0;     % falling edges only
                              targetSeq(2:end,:)=dts(:,:)<0;  targetSeq(1,:)=0;  

     case {'rfe','fre'};      stimSeq(2:end,:)=dss(:,:)~=0;   stimSeq(1,:)=0;     % rising or falling edges
                              targetSeq(2:end,:)=dts(:,:)~=0; targetSeq(1,:)=0;   

     case 'diff';             stimSeq(2:end,:)=dss;           stimSeq(1,:)=0;     % gradient of the stimulus    
                              targetSeq(2:end,:)=dts(:,:);    targetSeq(1,:)=0;   

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

% cut the data up based on the stimulus times
%stimTime_ms=shiftdim(cat(3,z.di(dim(2))).info.stimTime_ms));
irflen_samp=opts.irflen_samp;
irflen_ms=opts.irflen_ms;
if ( ~isempty(opts.nEpoch) )
	if ( opts.nEpoch > 0 ) 
	  windowIdx = 1:opts.nEpoch:size(stimTime_ms,1);
	else % neg is number of bits to cut into
	  windowIdx = round((((0:-opts.nEpoch-1)/-opts.nEpoch) * size(stimTime_ms,1))) +1;
	end
   stimTime_ms = stimTime_ms(windowIdx,:,:);	
	if ( ~isempty(irflen_ms) ) 
	  irflen_ms = max(max(diff(stimTime_ms(:,:),[],1)))+irflen_ms;
	elseif ( ~isempty(irflen_samp) ) 
	  fs=getSampleRate(z);
	  irflen_ms=max(max(diff(stimTime_ms(:,:),[],1)))+round(irflen_samp*1000/fs);
	  irflen_samp=[];
	end
end
if ( isempty(irflen_samp) && isempty(irflen_ms) ) 
	fs=getSampleRate(z);	irflen_samp=round(fs*.6);
end
offset_ms=opts.offset_ms;
if ( ~isempty(offset_ms) ) % include the requested offsets in the window
   stimTime_ms = stimTime_ms+offset_ms(1); 
   irflen_ms   = irflen_ms -offset_ms(1) + offset_ms(2);
end
if ( any(isnan(stimTime_ms(:))) ) % what do to with missing events?
   goodEvti = ~any(isnan(stimTime_ms),3);
   fprintf('%d missing events... removed from all letters.',sum(~goodEvti));
   stimTime_ms=stimTime_ms(goodEvti,:,:);
   stimSeq    =stimSeq(goodEvti,:,:);
   targetSeq  =targetSeq(goodEvti,:,:);   
end
z=jf_windowData(z,'dim',dim,'windowType',1,...
                'start_ms',stimTime_ms,'width_samp',irflen_samp,'width_ms',irflen_ms,...
					 'di','subepoch');
if ( 0 ) % normalise power in each epoch?
  nf=sqrt(tprod(z.X,[-1 -2 3:ndims(z.X)],[],[-1 -2 3:ndims(z.X)]));nf(nf==0)=1; z.X=repop(z.X,'/',nf);
end;
if ( ~isempty(offset_ms) ) % update the 0-time
  z.di(n2d(z,'time')).vals = z.di(n2d(z,'time')).vals+offset_ms(1);
end

% add the labels
foldIdxs=[]; if ( isfield(z,'foldIdxs') ) foldIdxs=z.foldIdxs; end;
if ( ~isempty(opts.nEpoch) ) % just replicate what's there
  z.Y=repmat(shiftdim(z.Y,-1),[size(z.X,n2d(z,'subepoch')),ones(1,ndims(z.Y))]);
  if ( isfield(z,'Ydi') ) z.Ydi=[z.di(n2d(z,'subepoch')) z.Ydi];   end;
else
  stimSeq = stimSeq(1:size(z.X,n2d(z,'subepoch')),:,:); % limit to #epochs sliced
  targetSeq = zeros(size(z.X,n2d(z,'subepoch')),size(z.X,n2d(z,opts.dim(2)))); % [ window x epoch ]
  for si=1:size(z.X,n2d(z,opts.dim{2}));
	 targetSeq(:,si) = stimSeq(:,target(si),min(end,si));
  end
  Ydi=z.di(n2d(z,{'subepoch' opts.dim{2}})); Ydi(2).info=[]; % remove markerdict info
  z=jf_addClassInfo(z,'Y',targetSeq,'Ydi',Ydi,'zeroLab',1);
end
% update the fold idxs, to take account of the new window dimension in Y - leave sequence out folding
if( ~isempty(foldIdxs) ) 
  z.foldIdxs = repmat(shiftdim(foldIdxs,-1),[size(z.X,n2d(z,'subepoch')),ones(1,ndims(z.Y))]); 
end;
z=jf_addprep(z,mfilename,'seq2epoch',[],[]);
return;



function testCase()
si=1;sessi=2;ci=9;
z=struct('expt',expt,'subj',subjects{si},'label',labels{1}{ci},'session',sessions{1}{sessi});

% test stuff
z=jf_load(z);
z=jf_retain(z,'dim','letter','idx',1:4);
z=stdpreproc(z,'dim','letter','eegonly',1,'bands',[.1 .2 12 15],'fs',32,'markerdict',markerdict,'nFold','loo');zpp=z;
% 1) sub-split/mean
z=jf_seq2epoch(zpp,'irflen_samp',round(32*.7));
figure(1);clf;jf_plotERP(z);
r=jf_seqperf(jf_cvtrain(z)); % per-epoch training

% 1.5) sub-split/combine
z=jf_seq2epoch(zpp,'irflen_samp',round(32*.7));
% per-epoch training
r=jf_seqperf(jf_cvtrain(z));
W=reshape(r.prep(end).info.res.opt.soln{1}(1:end-1),[size(zpp.X,1),round(32*.7)]);
jf_seqperf(r,'subSeqLen',6);
% whole sequence training, targets only
[stimSeq,stimTime_ms,target,targetSeq,targetKey]=jf_getstimSeq(z);
z.X=tprod(z.X,[1 2 -2 4],stimSeq,[-2 3 4]); % weighted combination
z.Y=lab2ind(target,targetKey); z.Ydi(1)=[]; z.foldIdxs=shiftdim(z.foldIdxs);
jf_cvtrain(z,'cvtrainFn','cvtrainMap','lossFn','est2loss','ydim',1,'objFn','mmlr_cg','labdim',3,'clsfr',1);
% whole sequence training, target-non-target diff (should be identical as equal numbers in all codes)
[stimSeq,stimTime_ms,target,targetSeq,targetKey]=jf_getstimSeq(z);
z.X=tprod(z.X,[1 2 -2 4],stimSeq*2-1,[-2 3 4]); % weighted combination
z.Y=lab2ind(target,targetKey); z.Ydi(1)=[]; z.foldIdxs=shiftdim(z.foldIdxs);
jf_cvtrain(z,'cvtrainFn','cvtrainMap','lossFn','est2loss','ydim',1,'objFn','mmlr_cg','labdim',3,'clsfr',1);
% sets of 6 flashes == no-averaging
[stimSeq,stimTime_ms,target,targetSeq,targetKey]=jf_getstimSeq(z);
szX=size(z.X); seqLen=6; nSeq=floor(szX(3)/6); nEp=szX(4);
z.X=reshape(z.X(:,:,1:seqLen*nSeq,:),[size(z.X,1),size(z.X,2),seqLen,nSeq*nEp]);
z.di(3).vals=z.di(3).vals(1:seqLen);z.di(4).vals=repmat(z.di(4).vals,[nSeq 1]);
z.Y=reshape(z.Y(1:seqLen*nSeq,:),[seqLen nSeq*nEp]);
z.Ydi(1)=mkDimInfo(seqLen,1,'tgt'); z.Ydi(2).vals=repmat(z.Ydi(2).vals,[nSeq 1]);
z.foldIdxs=reshape(repmat(z.foldIdxs,[nSeq 1 1]),nSeq*nEp,[]);
rmm=jf_cvtrain(z,'cvtrainFn','cvtrainMap','lossFn','est2loss','objFn','mmlr_cg','labdim',3,'clsfr',1);
Wmm=reshape(rmm.prep(end).info.res.opt.soln{1}(1:end-1),[size(zpp.X,1),round(32*.7)]);
tstf=rmm.prep(end).info.res.tstf; 
tstf=reshape(tstf,[seqLen*nSeq nEp size(tstf,3) size(tstf,4) size(tstf,5)]); % [ep x let x 1 x Cs]
dvf =tprod(tstf,[-1 2 0 3],stimSeq(1:seqLen*nSeq,:,:),[-1 1 2]); % [ nCls x let x Cs ]
yest=[];for ci=1:size(dvf,3); for li=1:size(dvf,2); [ans,yest(li,ci)]=max(dvf(:,li,ci));end;end;
clf;plot(yest,'linewidth',1);hold on; plot(target,'k-','linewidth',2);
sum(repop(target,'==',yest))



% 2) deconv/mean
z=jf_xytaucov(zpp,'irflen_samp',-(0:32*.67),'type','XY','normalize','none0');
mu=tprod(z.X,[1 -2 2 -1],single(z.Y>0),[-1 -2]); % [ch x tau]
figure(2);
clf;image3d(mu,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','ticklabs','sw')
jf_cvtrain(z,'cvtrainFn','cvtrainMap','lossFn','est2loss','ydim',1,'objFn','mmlr_cg','labdim',2,'clsfr',1);

% 3) sub-split/deconv/mean
z=jf_seq2epoch(zpp,'nEpoch',1,'irflen_samp',32);
z=jf_xytaucov(z,'irflen_samp',-(0:32*.7),'type','XY');
% compute the "ERP"
mu=tprod(z.X,[1 -3 2 -1 -2],single(z.Y>0),[-1 -2 -3]); % [ch x tau]
figure(3);
clf;image3d(mu,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','ticklabs','sw')

% 3) sub-split/deconv/mean
z=jf_seq2epoch(zpp,'nEpoch',12,'irflen_samp',32);
z=jf_xytaucov(z,'irflen_samp',-(0:32*.7),'type','XY','normalize','none0');
% compute the "ERP"
mu=tprod(z.X,[1 -3 2 -1 -2],single(z.Y>0),[-1 -2 -3]); % [ch x tau]
figure(4);
clf;image3d(mu,'plotPos',[z.di(1).extra.pos2d],'disptype','plot','ticklabs','sw')
