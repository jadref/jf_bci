function [z]=jf_recompPerf(z,varargin)
%  simple sequence performance comp where we just average away dim
opts=struct('dim',[],'lossFn','bal','mcdecoder',[],'subIdx',[],'verb',1);
opts=parseOpts(opts,varargin);
Y  =z.Y;
Ydi=z.Ydi;
dim=opts.dim;
if ( isempty(dim) && isfield(z,'Ydi') )
  dim=n2d(z,{Ydi.name},0,0);
  dim(dim==0)=[]; dim(dim==n2d(z.Ydi,'subProb'))=[];
else
  dim=1:ndims(z.Y)-1;
end

% extract the predictions and folding to use for performance computation
dv=z.X;
if ( isfield(z,'outfIdxs') ) % outer-fold computation
  foldIdxs=z.outfIdxs;
  %dv      =res.opt.f; % predictions after re-training on all the training data
else % normal inner fold computation
  if ( isfield(z,'foldIdxs') )
     foldIdxs=max(z.foldIdxs,[],ndims(z.foldIdxs));% any trials which were test at some fold
     if ( all(foldIdxs(:)>0) ) foldIdxs=1; end; % test only
  else
     foldIdxs=1;
  end
  %dv      =res.opt.tstf; % valdidation set predictions for all the training info
end

% build mc-perf info
% multi-class results recording
Y  =z.Y;
Ydi=z.Ydi;
spD  =n2d(Ydi,'subProb');
spMx =Ydi(spD).info.spMx;
spKey=Ydi(spD).info.spKey;
mcSp =Ydi(spD).info.mcSp;
nsubProbs=size(Y,spD); if ( mcSp ) nsubProbs=1; end;

if( numel(dim)>1 ) % compress n-d into 2-d, so dv2conf etc work
  if( ~all(sort(dim,'ascend')==(1:numel(dim))') )
	 error('Only 1st dims for dim is supported');
  end
  sz=size(Y); Y =reshape(Y,[prod(sz(1:numel(dim))) sz(numel(dim)+1:end) 1]);
  sz=size(dv);dv=reshape(dv,[prod(sz(1:numel(dim))) sz(numel(dim)+1:end) 1]);
  sz=size(foldIdxs);foldIdxs=reshape(foldIdxs,[prod(sz(1:numel(dim))) sz(numel(dim)+1:end) 1]);
  dim=1; spD=2:ndims(Y);
end


							  % now loop over folds computing the train/test performance
if ( opts.verb>0 ) 
fprintf('-------------------------\n');
if ( isfield(z,'outfIdxs') ) % outer-fold computation
   fprintf('(out)\t'); 
else
   fprintf('(ave)\t');
end
end
res=struct();
res.di = z.di; res.di(1)=mkDimInfo(1,1,'perf'); % store meta-info about results dimensions
for fi=1:size(foldIdxs,2);
  for spii=1:nsubProbs;
     spis=spii; if( mcSp ) spis=1:size(Y,2); end; % spi is set of sub-problems to compute performance for
     if( size(foldIdxs,1)>1 )
        trnInd=foldIdxs(:,fi)<0;  % training points
        tstInd=foldIdxs(:,fi)>0;  % testing points
        exInd =foldIdxs(:,fi)==0; % excluded points	 
        Ytrn=Y; Ytrn(tstInd,:)=0; Ytrn(exInd,:)=0;
        Ytst=Y; Ytst(trnInd,:)=0; Ytst(exInd,:)=0;
     else
        Ytrn=[];Ytst=Y;
     end
     dvspis=dv;
     if ( ~mcSp ) % select only this sub-problems info
        if ( ~isempty(Ytrn) ) Ytrn = Ytrn(:,spis); end;
        Ytst  = Ytst(:,spis);
        dvspis= dvspis(:,spis);
     end
     if ( ~isempty(Ytrn) && ~all(Ytrn(:)==0) )
        res.trnconf(:,spii,fi) =dv2conf(Ytrn,dvspis,[dim;spD],spMx);
        res.trn(:,spii,fi)     =conf2loss(res.trnconf(:,spii,fi),1,opts.lossFn);
        if ( opts.verb>0) fprintf('%0.2f/',res.trn(:,spii,fi)); end;
     else
        if ( opts.verb>0) fprintf('NA /'); end;
     end  
     res.tstconf(:,spii,fi) =dv2conf(Ytst,dvspis,[dim;spD],spMx);
     res.tst(:,spii,fi)     =conf2loss(res.tstconf(:,spii,fi),1,opts.lossFn);
     if( opts.verb>0 ) fprintf('%0.2f  ',res.tst(:,spii,fi)); end;
  end
end
if( opts.verb>0 ) fprintf('\n'); end;

info=struct('res',res); % preserve final filter state
summary='';
z =jf_addprep(z,mfilename,summary,opts,info);
return;
										  %------------------------
function testCase()
  z = jf_mksfToy('Y',ceil(rand(100,1)*2));oz=z;
  z = jf_addFolding(z); % 10-fold  
  z = jf_addFolding(z,'outfIdxs',1:50); % 1st half training										  
  z.foldIdxs(1:10:end,:)=0; % exclude some points from the folding
  z.X(:,:,51:end)=z.X(:,:,51:end)*100+10; % cov-shift for 2nd half
  c = jf_cvtrain(z)
  jf_recompPerf(c);

  % now test with sets of independent binary problems
  
