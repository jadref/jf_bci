function [z]=jf_rmOutliers(z,varargin)
% remove elements more than a certain number of std-deviations in variance from the median
%
% Options:
%  dim    -- dimension(s) along which to remove elements
%  thresh -- [2x1] threshold in data-std-deviations std-deviations to test to remove
%            1st element is threshold above (3), 2nd is threshold below (-inf)
%  maxIter-- [int] number of times round the remove+re-compute var loop (6)
%  idx    -- [Nx1] or [size(X,dim) bool] sub-set of indicies along dim to consider
%  mode   -- [str] what do we do: 
%            mark - just mark as bad, i.e. set z.di(dim(1)).extra(badIdx).isbad=true;
%            zero - set value to 0, reject - remove from dataset
%            zeroY - set label to 0 (so is ignored)
%  feat   -- [str] which feature type to use {'mu','var'}  ('var')
%  testFn -- {'rerun' 'fixed' 'skip'} how do we run on test data ('rerun')
%  summary-- additional descriptive info
%  
opts=struct('dim','epoch','feat',[],'thresh',3.5,'maxIter',6,'mode','reject',...
            'verb',0,'idx',[],'summary','','subIdx',[],'blockIdx',[],'testFn','rerun');
[opts,varargin]=parseOpts(opts,varargin);

szX=size(z.X); nd=numel(szX);
dim =n2d(z.di,opts.dim,0,0); dim(dim==0)=[]; dim=sort(dim,'ascend');

subIdx=opts.subIdx;
if( ~isempty(subIdx) ) % run on a sub-set of the data
  subIdx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
end
if( ~isempty(opts.idx) ) % 2nd way to specify a sub-set
  if ( isempty(subIdx) ) subIdx={}; for d=1:ndims(z.X); subIdx{d}=1:size(z.X,d); end; end;
  tmp=opts.idx; if( islogical(tmp) ) tmp=find(tmp); end;
  subIdx{dim}=intersect(subIdx{dim},tmp); 
end;

blockIdx=[]; nBlock=1;
if( ~isempty(opts.blockIdx) ) 
   [blockIdx,blkD,blockFound]=getBlockIdx(z,opts.blockIdx);
   if( blockFound ) 
      if ( ischar(blkD) ) blkD=n2d(z.di,blkD); end;
      bId=unique(blockIdx); nBlock  =numel(bId);    
   else
      blockIdx=[];
   end
end

% Call worker to do the work
if ( isempty(blockIdx) ) 
   if( isempty(subIdx) ) X=z.X; else X=z.X(subIdx{:}); end; % get subset to work on
   [badInd,feat,threshs,stds,mus]=idOutliers(X,dim,opts.thresh,opts.feat,opts.maxIter,opts.verb);

   % map back to full set of entries from the subset used
   if( ~isempty(subIdx) ) 
      tmp=badInd; badInd=false([szX(dim),1]); badInd(subIdx{dim})=tmp; clear tmp;
   end

else % outlier-id in blocks
   badInd=false([prod(szX(dim)),szX(blkD),1]); %[ sz(dim) x sz(blkD) ]
   feat  =zeros([prod(szX(dim)),numel(bId)]);
   idx=repmat({':'},numel(szX),1);
   for bi=1:numel(bId);
      blkIdx=(blockIdx==bId(bi));
      idx{blkD}=blkIdx;
      [badIndi,feati,threshsi,stdsi,musi]=idOutliers(z.X(idx{:}),dim,opts.thresh,opts.feat,opts.maxIter,opts.verb);
      badInd(:,blkIdx)=repmat(badIndi(:),[1,sum(blkIdx)]);
      % save the summary info
      feat(:,bi)=feati;      threshs{bi}=threshsi;       stds{bi}=stdsi;       mus{bi}=musi;
   end
   bIsz=ones(1,numel(szX));bIsz(dim)=szX(dim); bIsz(blkD)=szX(blkD);
   badInd=reshape(badInd,bIsz); % make similar size as X
   dim(end+1)=blkD; % add to the dim-info
end


odi=z.di(dim);
% mark these entries as bad
for d=1:numel(dim);
   if(numel(z.di(dim(d)).extra)==0) % fix an empty extra struct
      z.di(dim(d)).extra=repmat(struct(),1,numel(z.di(dim(d)).vals));
   end
   [z.di(dim(d)).extra.isbad] = num2csl(squeeze(badInd),setdiff(1:numel(dim),d));
end
% do what-ever else we were asked to
switch lower(opts.mode);
 case 'reject'; 
   if( numel(dim)>1 ) warning('reject not supported for multiple dims or blocked removal!'); badInd=any(badInd,blkD); end;
   z   = jf_retain(z,'dim',dim(1),'idx',~any(badInd(:,:),2));
 case 'mark'  ; % already done so do nowt
 case {'zero','nan'}  ; % zero out these values  
  idx={};for di=1:ndims(z.X); idx{di}=1:size(z.X,di); end;
  bi=squeeze(badInd);
  szbi=size(bi);
  if( strcmpi(opts.mode,'zero') ) val=0; elseif( strcmpi(opts.mode,'nan') ) val=NaN; end;
  for j=1:prod(szbi(2:end)); % deal with >1 dimensional detection
     if ( any(bi(:,j)) ) % bad in this col
        if ( numel(dim)>1 )[idx{dim(2:end)}]=ind2sub(szbi(2:end),j); end; %>1 d
        idx{dim(1)}=bi(:,j);        
        z.X(idx{:})=val;
     end
  end
  testFn={};
 case 'zeroY'  ; % zero out the label for these values
  if ( isfield(z,'Ydi') && any(n2d(z.Ydi,{odi.name},0,0)) )
	 szY=size(z.Y);
	 idx={};for d=1:ndims(z.Y); idx{d}=1:szY(d); end;
	 ydim=n2d(z.Ydi,{odi.name},1,0); %
	 bi=squeeze(badInd); if ( any(ydim)==0 ) bi=sum(bi,find(ydim==0)); end;	ydim(ydim==0)=[];
	 szbi=size(bi);
	 for j=1:prod(szbi(2:end)); % deal with >1 dimensional detection
		if ( any(bi(:,j)) ) % bad in this col
        if ( size(bi,2)>1 )[idx{ydim(2:end)}]=ind2sub(szbi(2:end),j); end; %>1 d
        idx{ydim(1)}=bi(:,j);
        z.Y(idx{:})=0;
		end
	 end
  end
end

% tidy up the summary info
summary=sprintf('%sed %d',opts.mode,sum(badInd(:)));
if ( numel(dim)>1 ) 
  summary=sprintf('%s [%s%s]s',summary,z.di(dim(1)).name,sprintf(' x %s',z.di(dim(2:end)).name));
else;
  summary=sprintf('%s %ss',summary,z.di(dim).name);
end;
if ( sum(badInd(:))<6 && sum(badInd(:))>0 ) 
   badIdx=find(any(reshape(badInd,size(badInd,dim(1)),[]),2));
   if ( iscell(odi(1).vals) ) 
      istr = [sprintf('%s,',odi(1).vals{badIdx(1:end-1)}),odi(1).vals{badIdx(end)}]; 
   else 
      istr = [sprintf('%g,',odi(1).vals(badIdx(1:end-1))) sprintf('%g',odi(1).vals(badIdx(end)))]; 
   end   
   summary=sprintf('%s (%s)',summary,istr);
end
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
info=struct();
if ( ~strcmp(opts.mode,'reject') ) 
   info.testFn={'error' 'not-implementated yet'};
else % over-ride the reject prep-info
   info=z.prep(end).info; 
   switch (opts.testFn)
    case 'fixed'; info.testFn={'jf_retain' 'dim' dim,'idx' ~badInd};      
    case 'skip';  info.testFn='';
   end
   z.prep(end)=[]; z.summary = z.summary(1:find(z.summary==10,1,'last')-1); % remove the retain's prep info
end
info.feat=feat; info.mus=mus; info.stds=stds; info.threshs=threshs; info.isbad=badInd;
z = jf_addprep(z,mfilename,summary,opts,info);   
return;
%--------------------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
t=jf_rmOutliers(z,'dim','ch');jf_disp(t),plot(t.prep(end).info.stds)
t=jf_rmOutliers(z,'dim','ch','feat','mu'),jf_disp(t);plot(t.prep(end).info.mus)
t=jf_rmOutliers(z,'dim',{'ch','epoch'},'mode','mark');jf_disp(t)
t=jf_rmOutliers(z,'dim',{'ch','epoch'},'mode','zero');jf_disp(t)
t=jf_rmOutliers(z,'dim',{'ch','epoch'},'mode','zeroY');jf_disp(t)
t=jf_rmOutliers(z,'dim','epoch','mode','zeroY');jf_disp(t)

t=jf_rmOutliers(z,'dim','ch','blockIdx','block','mode','zero'); % per-block zero