function [z,res]=jf_cvtrainPipe(z,pipeline,varargin)
% cross validated training of an entire signal processing pipeling
%
%  [z,res]=jf_cvtrainPipe(z,pipeline,varargin)
%
% Inputs:
%  z - data object
%  pipeline - { pipe1Fn, pipe2Fn,.... } cell array of functions to execute in the given order
%             to perform the required pre-processing
% Options:
% N.B. {pipe1}.{optnames} , {vals}     -- used for featEx step
%      classifier.{optnamess}, {vals} -- for the classifier as set in z
%      also, numeric inputs are cut down the columns (as in for loops) to 
%      make th value set.  Thus for array arguments you *must* wrap each 
%      value set in a cell, e.g. {[1 2;1 2],[1 1; 1 1]}
% e.g. 'featEx.cost',[.1:.1:1;.1:.1:1], 'classifier.nu',[0:.1:1], 'featEx.nfilt',6
%
%  subIdx - {str}  cell array of pipeline stages which could possibly overfit  ('all')
%                  & hence must be re-run every time the folding changes 
%                  if it is the string 'all' then all pipeline stages overfit
%  foldIdxs
%  verb
%  pipename - [str] name to use in the prep recording
%  mergeClassRes - [bool] merge the results from the different classifier runs (1)
%  mergeexcludefn - {str} list of field names not to merge over runs/folds ({'fold' 'di' 'opt'})
%  lossType - [str] type of loss to record/report
%
% TODO: Add ability to re-order hyper-parms -> folds for stages which don't overfit and 
%  hence can be fixed for all folds
opts=struct('foldIdxs',[],'verb',1,'pipename',[],...
  			   'subIdx','all','subIdxExclude',{{'jf_cvtrain','jf_compKernel'}},'mcPerf',0,'binsp',1,...
            'mergeClassRes',1,'mergeexcludefn',{{'fold','di','opt'}},...
            'lossType','bal','outerSoln',0,'recSoln',0);
[opts,varargin]=parseOpts(opts,varargin);
% convert function handles to strings.
if ( isempty(pipeline) ) pipeline=opts.pipeline; end;
if ( ischar(pipeline) )   pipeline={pipeline}; end;
if ( isa(pipeline,'function_handle') ) 
  strpl={};for pi=1:numel(pipeline); strpl{pi}=func2str(pipeline(pi)); end;
  pipeline=strpl;
else
  for pi=1:numel(pipeline); % convert func handles to string function names
    if(isa(pipeline{pi},'function_handle')) pipeline{pi}=func2str(pipeline{pi}); end; 
  end;
end
opts.pipeline=pipeline;

% parse the pipeline options
% strip out empty stages
tmp=[]; for pi=1:numel(pipeline); if ( isempty(pipeline{pi}) ) tmp=[tmp;pi]; end; end; pipeline(tmp)=[];
pipeopts=struct(); for pi=1:numel(pipeline); pipeopts.(pipeline{pi})=[]; end;
pipeopts=parseOpts(pipeopts,varargin);
for pi=1:numel(pipeline); % convert cell arrays of options into structs
  popts = pipeopts.(pipeline{pi});
  if ( iscell(popts) ) % convert from cell array to struct
    sopts=struct(); for fi=1:2:numel(popts)-1; sopts=setfield(sopts,popts{fi},popts{fi+1}); end; 
    pipeopts.(pipeline{pi})=sopts;
  end
end;
% add special bit for jf_cvtrain to prevent running outer fold training, and to record the solutions
if ( isfield(pipeopts,'jf_cvtrain') ) 
  pipeopts.jf_cvtrain.outerSoln=0; 
  pipeopts.jf_cvtrain.recSoln=1;
  pipeopts.jf_cvtrain.mcPerf =opts.mcPerf;
  if ( ~isfield(pipeopts.jf_cvtrain,'binsp') ) pipeopts.jf_cvtrain.binsp  =opts.binsp; end;
  % special case, make these hyper-parameters explicit
  if ( ~isfield(pipeopts.jf_cvtrain,'Cs') || isempty(pipeopts.jf_cvtrain.Cs) ) 
    pipeopts.jf_cvtrain.Cs=[5.^(3:-1:-3)]'; 
  elseif ( size(pipeopts.jf_cvtrain.Cs,1)==1 ) % make col vector
    pipeopts.jf_cvtrain.Cs=pipeopts.jf_cvtrain.Cs';
  end
end;
% identify the pipeline stages which will overfit
subIdxp=opts.subIdx;
if ( ischar(subIdxp) ) 
  if (strcmp(subIdxp,'all') ) % all pipe-stages
    subIdxp=pipeline;
  else
    subIdxp={subIdxp}; 
  end
end;
% convert to logical indicator
tmp=false(numel(pipeline),1);
for i=1:numel(subIdxp) 
  if ( iscell(subIdxp) )       pipeidx=strmatch(subIdxp{i},pipeline); pipeidx=pipeidx(1);
  elseif( isnumeric(subIdxp) ) pipeidx=subIdxp(i);
  end
  if ( ~any(strcmp(pipeline{pipeidx},opts.subIdxExclude)) ) % if not excluded auto then add to subidx
	 tmp(pipeidx)=true;
  end
end;
subIdxp=tmp;

% if ( isempty(subIdxp) ) subIdxp=false(numel(pipeline),1); end;

foldStage=find(subIdxp); % the point in the pipeline where the folding should best fit.

% make the parameter settings sets
[combo di] = makecombs(pipeopts);

% re-arrange dimensions into inverse-execution order -- so first to run
% changes slowest.
prm=[];
for i=numel(pipeline):-1:1
  tmp=strmatch(pipeline{i},{di.name});
  if ( ~isempty(tmp) ) prm=[prm(:); tmp(:)]; end;
end
di=di([prm(:); setdiff(1:numel(di),prm)]);
prm=[prm(:); setdiff(1:ndims(combo),prm)];
combo=permute(combo,prm); 

% extract the folding to use
foldIdxs=opts.foldIdxs;
if ( isempty(opts.foldIdxs) ) 
  if ( isfield(z,'foldIdxs') ) 
    foldIdxs=z.foldIdxs; 
  else
    foldIdxs=10;
  end
end
if ( isscalar(foldIdxs) ) 
   nFolds  = foldIdxs; 
   foldIdxs= gennFold(z.Y,nFolds);
end

% reshape to match the number of dimensions of Y
szY=size(z.Y); szf=size(foldIdxs); 
% work out the sub-problem dimension of Y/foldIdxs
spD=numel(szY);
if ( isfield(z,'Ydi') ) 
   dim = n2d(z.di,{z.Ydi(1:end-1).name},1,0); 
	if(~any(dim)) % relax and try prefix match
	  warning('Couldnt get an exact match for the trial dimension');
	  dim = n2d(z.di,{Ydi(1:end-1).name},0,0); 
	end
	dim(dim==0)=[]; % get dim to work along
	if ( isempty(dim) ) warning('Couldnt work out the trial dimension!...'); end;
   if ( ~isempty(dim) ) spD=setdiff(1:numel(szY),dim); end
end
if ( numel(szY)>=numel(szf) ) % insert the sub-prob dim in right place if not there
  szf=[szf(1:spD-1) 1 szf(spD:end)];
  foldIdxs=reshape(foldIdxs,szf);
end

nFold=size(foldIdxs,ndims(foldIdxs)); 
nSubProbFold=1; if ( ~isempty(spD) && szf(spD)>1 ) nSubProbFold=szf(spD); end; 
% addin the sub-prob and fold dims
if( numel(di)>1 )
  di=[di(1:end-1); mkDimInfo(nSubProbFold,1,'subProb'); mkDimInfo(nFold,1,'fold'); di(end)];
else
  di=[mkDimInfo(nSubProbFold,1,'subProb'); mkDimInfo(nFold,1,'fold'); di(end)];
end
if ( opts.verb>=-3 ) 
  fprintf('Order: '); 
  for i=numel(di)-1:-1:1; fprintf('%s (%d), \t',di(i).name,size(di(i).vals,2)); end;
  fprintf('\n');
end

%------------------------------------------------------------------------------------
% Next compute the per-fold values
res=[];
% for multi-dim foldIdxs.  N.B. assumes foldIdxs has shape [epoch dim(s) subProb, fold]
fidx={};for d=1:ndims(foldIdxs); fidx{d}=':'; end;
for foldi=1:nFold;
  for spi=1:nSubProbFold;
    % get the training test split (possibly sub-prob specific)
    fidx{end}=foldi; % set the fold
    if ( ~isempty(spD) ) fidx{spD}=min(size(foldIdxs,spD),spi); end; % set the sub-prob
    foldIdx=foldIdxs(fidx{:});
    zpipe=z;
    if( isfield(zpipe,'Y') )
      if ( nSubProbFold>1 ) zpipe.Y = z.Y(fidx{1:end-1}); end
		exIdx  =foldIdx==0;
      if ( any(exIdx(:)) )
         zpipe.Y=repop(zpipe.Y,'*',single(~exIdx)); % ensure excluded points have no label info
      end
    end; % subProb specific folding
  
    % run the pipe, possibly on multiple sub-problems?
    [zpipe,resfi]=runcombos(zpipe,pipeline,combo,foldIdx,subIdxp,foldi,spi,opts.verb);
    % store the results
    if ( isempty(res) ) % 1st time round
      res=resfi(:);
    else % rest of the times
      res(:,spi,foldi)=resfi(:); % [hyper-params x subProb x folds]
    end
      
  end
end
zout=zpipe{end}; % output is same as the final result of running the pipe...
% shape the results to the right shape!
if( numel(combo)==1 ) szCombo=[]; 
elseif ( ndims(combo)==2 && size(combo,2)==1 ) szCombo=numel(combo); 
else szCombo=size(combo); end;
res=reshape(res,[szCombo,size(res,2),size(res,3)]); % [ szHps x nSubProb x nFold] 

% extract classification results if is of the right type
if ( opts.mergeClassRes )
  if ( isfield(res(1).prep(end).info,'res') ) % store sub-calls res
    for ri=1:numel(res); 
      hpres(ri)=res(ri).prep(end).info.res;
    end; 
    hpres=reshape(hpres,size(res));
  end

  dv=[];
  if ( isfield(hpres,'tst') ) 
	  cvres=hpres(1);
	  % concatenate information over outer iterations
	  cvres.tst = reshape(cat(ndims(hpres(1).tst)+1,hpres.tst),[size(hpres(1).tst) size(res)]);
	  cvres.trn = reshape(cat(ndims(hpres(1).trn)+1,hpres.trn),[size(hpres(1).trn) size(res)]);
	  if ( isfield(hpres,'tstf') ) 
        cvres.tstf= reshape(cat(ndims(hpres(1).tstf)+1,hpres.tstf),[size(hpres(1).tstf) size(res)]);
     end
	  if ( isfield(hpres,'tstconf') ) 
        cvres.tstconf= reshape(cat(ndims(hpres(1).tstconf)+1,hpres.tstconf),[size(hpres(1).tstconf) size(res)]);
     end
	  if ( isfield(hpres,'trnconf') ) 
        cvres.trnconf= reshape(cat(ndims(hpres(1).trnconf)+1,hpres.trnconf),[size(hpres(1).trnconf) size(res)]);
     end
	  cvres.di=[cvres.di(1:end-1); di]; % update meta-info
	  % remove duplicate subProb dims
	  spD=find(strcmp('subProb',{cvres.di.name})); 
	  if(numel(spD)>1)
		 if(size(cvres.tst,spD(2))>1) badspD=spD(1); else badspD=spD(2); end; 
		 cvres.di=cvres.di([1:badspD-1 badspD+1:end]);
		 tmp=size(cvres.tst); cvres.tst=reshape(cvres.tst,tmp([1:badspD-1 badspD+1:end]));
		 tmp=size(cvres.trn); cvres.trn=reshape(cvres.trn,tmp([1:badspD-1 badspD+1:end]));
		 if(isfield(cvres,'tstf'))
          tmp=size(cvres.tstf); cvres.tstf=reshape(cvres.tstf,tmp([1:badspD-1 badspD+1:end]));
       end
		 if(isfield(cvres,'tstconf'))
          tmp=size(cvres.tstconf); cvres.tstconf=reshape(cvres.tstconf,tmp([1:badspD-1 badspD+1:end]));
       end
		 if(isfield(cvres,'trnconf'))
          tmp=size(cvres.trnconf); cvres.trnconf=reshape(cvres.trnconf,tmp([1:badspD-1 badspD+1:end]));
       end
	  end
	  % average away the fold information
	  foldD=n2d(cvres.di,'fold');
     if( isfield(cvres,'trnconf') )
        cvres.trn = conf2loss(sum(cvres.trnconf,foldD),opts.lossType);
        cvres.tst = conf2loss(sum(cvres.tstconf,foldD),opts.lossType);
     else
        cvres.tst = sum(cvres.tst,foldD)./size(cvres.tst,foldD);
        cvres.trn = sum(cvres.trn,foldD)./size(cvres.trn,foldD);
     end
	  if ( isfield(cvres,'tstf') ) 
        cvres.tstf= sum(cvres.tstf,foldD); % N.B. assumes tstf==0 for training examples
     end
	  cvres.di  = cvres.di([1:foldD-1 foldD+1:end]);
  else
	 % get the predictions
	 dv=agetfield(res,'X','uniformp',0); % [ nEp x szHpdims x nSubProb x nFold]
	 if ( iscell(dv) ) 
		dv=cat(ndims(res)+ndims(dv{1}),dv{:}); % [ nEp x szHpdims x nSubProb x nFold]
	 else
		dv=permute(dv,[ndims(res)+(1:ndims(res(1).X)) 1:ndims(res)]); % put the size(res) to the end
	 end
	 % BODGE! not sure this works in general....
	 dvdi=[res(1).di(setdiff(1:end-1,n2d(res(1).di,'fold',0,0))); di]; 
	 % remove duplicate subProb dims
	 spD=strmatch('subProb',{dvdi.name},'exact'); 
	 if(numel(spD)>1) 
		if(size(dv,spD(2))>1) badspD=spD(1); else badspD=spD(2); end; 
		szdv=size(dv); dv=reshape(dv,szdv([1:badspD-1 badspD+1:end])); 
		% use the more detailed sub-prob info
		if(isnumeric(dvdi(spD(1)).vals)) dvdi(spD(1))=dvdi(spD(2)); else dvdi(spD(2))=dvdi(spD(1)); end
		dvdi(badspD)=[];
	 end;
	 % compute performance summary
	 trDims={'epoch'}; if ( ~isempty(z.Ydi) ) trDims={z.Ydi(1:n2d(z.Ydi,'subProb')-1).name};end;
	 cvres=cvPerf(z.Y,dv,n2d(dvdi,{trDims{:} 'subProb' 'fold'}),foldIdxs,opts.lossType);
	 hpDs=setdiff(1:numel(dvdi)-1,n2d(dvdi,{trDims{:} 'subProb' 'fold'}));
	 for hi=1:numel(hpDs); cvres.di(2+hi)=dvdi(hpDs(hi)); end;
	 cvres.di(2) = dvdi(n2d(dvdi,'subProb'));
	 
	 % BODGE: use cvres.tst as the figure of merit
	 cvres.trn=cvres.trnbin;
	 cvres.tst=cvres.tstbin;
  end
  
  % extract the per-fold predictions
  if ( ~isfield(cvres,'f') && isfield(hpres(1),'fold') && isfield(hpres(1).fold,'f') ) 
    f =agetfield(hpres,'fold.f','uniformp',0);
    if ( iscell(f) ) 
       f=cat(ndims(res)+ndims(f{1}),f{:}); % [ nEp x szHpdims x nSubProb x nFold]
    else
       f =permute(f,[ndims(res)+ndims(f) 1:ndims(res)]); % put the size(res) to the end
    end
    f = mean(f,ndims(f));
    cvres.f   =f;
  end
  
  % extract the solutions
  if ( isfield(hpres(1),'fold') && isfield(hpres(1).fold,'soln') ) %&& isequal(szperf(hpD),szHpres(1:end-2)) )
    soln=agetfield(hpres,'fold.soln');
    % put the size(res) to the end
    if ( iscell(soln) ) %soln=cat(ndims(soln{1})+1,soln{:}); 
      tmp=cell([prod(size(soln{1})),prod(size(soln))]); 
      for ti=1:size(tmp,2); tmp(:,ti)=soln{ti}(:); end;
      soln=reshape(tmp,[size(soln{1}),size(soln)]);
    end;

    % N.B. folding is *always* the last dim of both f and soln so sum it away.
    idx={}; for d=1:ndims(soln); idx{d}=1:size(soln,d); end;
    for fi=1:size(soln,ndims(soln));      
      idx{ndims(soln)}=fi;
      if ( fi==1 )
        avesoln=soln(idx{:});
      else
        if( isnumeric(avesoln) )
          avesoln = avesoln+soln{idx{:}}./size(soln,ndims(soln));
        elseif ( iscell(avesoln) && isnumeric(avesoln{1}) )
          solni=soln(idx{:});
          for bi=1:numel(avesoln)
            avesoln{bi}=avesoln{bi}+solni{bi}./size(soln,ndims(soln));
          end
        else
          warning('dont know how to process solution');
        end
      end
    end
    % record all the solutions
    cvres.soln=avesoln;
  end  
    
  % compute the optimal parameter settings
  % record the optimal solution and it's parameters
  szperf = size(cvres.tst); 
  spD=n2d(cvres.di,'subProb');
  hpD=setdiff(2:numel(szperf),spD);% for indexing into hyper-params
  [ans,opthpi]=mmax(sum(cvres.tst,spD),2:ndims(cvres.tst));
  % convert to an index expression for the performance   
  if(~isempty(hpD)) 
    tmp=szperf; tmp(spD)=1;
    hpIdx={}; [hpIdx{1:numel(szperf)}]=ind2sub(tmp,opthpi); % get the indices
    hpIdx{spD}=1:szperf(spD); % add the selection of all sub-probs back in
  else hpIdx={1};
  end;
  % Now run on all the data with these settings to get the optimal solution and it's predictions    
  cvres.opt.hpi =hpIdx(hpD); % just for the hyper-params
  tmp=hpIdx;tmp{1}=1:size(cvres.trn,1); tmp{spD}=1:size(cvres.trn,spD); cvres.opt.trn=cvres.trn(tmp{:});
  tmp=hpIdx;tmp{1}=1:size(cvres.tst,1); tmp{spD}=1:size(cvres.tst,spD); cvres.opt.tst=cvres.tst(tmp{:});
  if ( isfield(cvres,'tstf') ) 
     tmp=hpIdx;tmp{1}=1:size(cvres.tstf,1); tmp{spD}=1:size(cvres.tstf,spD); cvres.opt.tstf=cvres.tstf(tmp{:});
  end
  % build the combo info
  szcombo=size(combo); if(szcombo(2)==1) szcombo(2)=[]; end;
  if ( numel(hpD)>numel(szcombo) || isequal(szcombo,1)  ) % 1st hp is the C parameter for the classifier
	 optcombo=combo; 
	 if( ~isempty(szcombo) && ~isequal(szcombo,1) ) optcombo = combo(hpIdx{hpD(2:end)}); end;
    if( numel(hpD)>0 )
       optcombo.jf_cvtrain.Cs=optcombo.jf_cvtrain.Cs(hpIdx{hpD(1)});
    end
  else
    optcombo = combo(hpIdx{hpD});
  end
  
  % report ave results
  if ( opts.verb>=0 ) 
    if ( opts.verb>0 ) fprintf('\n----------\n');  else fprintf('\n'); end;
    %szperf = size(cvres.trn); 
    %spD=n2d(cvres.di,'subProb');
    %hpD=setdiff(2:numel(szperf),spD);% for indexing into hyper-params
    for spi=1:szperf(spD);
      hpIdx={};for d=1:numel(szperf);hpIdx{d}=1;end;
      if ( numel(hpD)>0 ) hpIdx{hpD(1)}=1:szperf(hpD(1)); end;
      hpIdx{spD}=spi;
      for hpi=1:prod(szperf(hpD(2:end)));
        if ( numel(hpD)>1 )
          fprintf('(ave,%d/%s)\t',spi,[sprintf('%d,',hpIdx{hpD(2:end-1)}) sprintf('%d',hpIdx{hpD(end)})]);
        else
          fprintf('(ave,%d)\t',spi);
        end
        for hi=1:szperf(hpD(1)); % print values
          hpIdx{hpD(1)}=hi;
          fprintf('%0.2f/%0.2f',cat(1,cvres.trn(hpIdx{:}),cvres.tst(hpIdx{:})));
          if ( isequal(hpIdx(hpD),cvres.opt.hpi) ) fprintf('*'); else fprintf(' '); end; % mark opt param
          fprintf('\t');
        end
        fprintf('\n');
        % generate next sub-Idx
        for d=hpD(2:end);
          if(hpIdx{d}<szperf(d))hpIdx{d}=hpIdx{d}+1; break; else hpIdx{d}=1; end; 
        end
      end
    end
    % Print cross problem average
    if ( szperf(spD)>1 )
       hpIdx={};for d=1:numel(szperf);hpIdx{d}=1;end;
       hpIdx{spD}=1:szperf(spD);
       if ( numel(hpD)>0 ) hpIdx{hpD(1)}=1:szperf(hpD(1)); end;
       for hpi=1:prod(szperf(hpD(2:end)));
          if ( numel(hpD)>1 )
             fprintf('(ave,ave/%s)\t',[sprintf('%d,',hpIdx{hpD(2:end-1)}) sprintf('%d',hpIdx{hpD(end)})]);
          else
             fprintf('(ave,av)\t');
          end
          for hi=1:szperf(hpD(1)); % print values
             hpIdx{hpD(1)}=hi;
             fprintf('%0.2f/%0.2f',cat(1,mean(cvres.trn(hpIdx{:}),spD),mean(cvres.tst(hpIdx{:}),spD)));
             if ( isequal(hpIdx(hpD),cvres.opt.hpi) ) fprintf('*'); else fprintf(' '); end; % mark opt param
             fprintf('\t');
          end
          fprintf('\n');
          % generate next sub-Idx
          for d=hpD(2:end);
             if(hpIdx{d}<szperf(d))hpIdx{d}=hpIdx{d}+1; break; else hpIdx{d}=1; end; 
          end
       end
    end
  end

  % run the optimal parameters combination
  if ( opts.outerSoln~=0 )
	 % get outer fold split
	 if ( isfield(z,'outfIdxs') )   outfIdxs=z.outfIdxs;
	 else                           outfIdxs=-single(any(foldIdxs,ndims(foldIdxs)));
	 end;
	 zpipe=runcombos(z,pipeline,optcombo,outfIdxs,subIdxp,0,spi,opts.verb);
	 zout=zpipe{end}; % only keep the final result
	 % opt structure for these results
	 resopt     = zout.prep(end).info.res.opt; 
	 resopt.hpi = cvres.opt.hpi;
	 resopt.zopt= zout;   % store the actual pipeline results
	 cvres.inner.opt=cvres.opt;
    % replace opt results info with info from this run
	 cvres.opt  = resopt;
    if ( ~any(outfIdxs(:)>0) ) % no testing data, use the cv info, so results extraction works
       cvres.opt.tst     = cvres.inner.opt.tst;
       cvres.opt.tstconf = cvres.inner.opt.tstconf;
       cvres.opt.tstf    = cvres.inner.opt.tstf;
    end
	 
	 if ( isfield(z,'outfIdxs') ) % compute outer fold performance info
		outres = cvPerf(z.Y,resopt.tstf,[1 2 3],z.outfIdxs,[],opts.binsp);
		% print the resulting test set performances
		for spi=1:size(outres.trn,n2d(outres.di,'subProb')); % loop over sub-problems
        fprintf('(out/%2d*)\t%.2f/%.2f\n',spi,outres.trn(:,spi),outres.tst(:,spi));
		end
		% overwrite perf info with outer fold perf info
		cvres.inner.trnconf=cvres.trnconf; cvres.trnconf=outres.trnconf;
		cvres.inner.tstconf=cvres.tstconf; cvres.tstconf=outres.tstconf;
		cvres.inner.trn    =cvres.trn;     cvres.trn    =outres.trn;
		cvres.inner.tst    =cvres.tst;     cvres.tst    =outres.tst;
		if ( isfield(outres,'trnauc'))
		  cvres.inner.trnauc=cvres.trnauc; cvres.trnauc=outres.trnauc;
		  cvres.inner.tstauc=cvres.tstauc; cvres.tstauc=outres.tstauc;
		end;
	 end
  end

  % rename variables
  rres=res;res=cvres; 
end

% Record the total stats
z = zout; % final run of the pipe is output .....
info=struct('di',di,...
            'pipeline',{pipeline},...%            'combo',combo,...
            'res',res);
summary='';
for pi=1:numel(pipeline); 
  if(ischar(pipeline{pi}))summary=[summary ',' pipeline{pi}];
  else                    summary=[summary ',' func2str(pipeline{pi})];
  end
end
summary=sprintf('CV pipeline opt with: %s',summary);
z = jf_addprep(z,mfilename,summary,opts,info);
return

%-------------------------------------------------------------------------
% run the classification pipeline for the given training/testing sets and
% set of parameters (in combo) and return the results.
function [zpipe,res]=runcombos(zpipe,pipeline,combo,foldIdx,subIdxp,foldi,spi,verb) 

pipelen=numel(pipeline);
if ( ~iscell(zpipe) ) 
   tz=zpipe; zpipe=cell(pipelen,1); zpipe{1}=tz; clear tz; 
end;
zpipe{1}.foldIdxs=foldIdx;
if( isfield(zpipe{1},'Y') ) % ensure excluded points have no label info
  oY=zpipe{1}.Y;
  exIdx  =foldIdx>=0; % any test/excluded points
  zpipe{1}.Y=repop(zpipe{1}.Y,'*',single(~exIdx)); 
end;

szcombo=size(combo); if( numel(szcombo)==2 && szcombo(2)==1 ) szcombo=szcombo(1); end; % size
% do the runs
oopts=[]; cIdx=cell(ndims(combo),1);
for ci = 1:numel(combo)
  [cIdx{1:numel(szcombo)}]=ind2sub(szcombo,ci);
  if ( verb>=0 ) 
    if ( cIdx{1}==1 ) % only for the 1st of last parameter
      if ( numel(combo)>1 ) 
        fprintf('\n(%2d,%d/%s)\t',foldi,spi,[sprintf('%d,',cIdx{2:numel(szcombo)-1}) sprintf('%d',cIdx{numel(szcombo)})]);
      else
        fprintf('\n(%2d,%d)\t',foldi,spi);
      end
    end
  end
  opts=combo(ci);
  parentRan=0;
  for pj = 1:numel(pipeline);
    % current options for this pipeline stage
    popts=getfield(opts,pipeline{pj});
    % do the feature extraction bit with the indicated featExtAlg
    if ( parentRan || ci==1 || isempty(zpipe{pj+1}) || (ci>1 && ~isequal(getfield(oopts,pipeline{pj}),popts)) )
      % put in cell, so empty expands to disappear
      if( isempty(popts) ) popts={}; elseif (~iscell(popts) ) popts={popts}; end; 
      % BODGE: add the validation labels back in for the last pipeline stage so it can compute the validatio performance
      if ( pj==numel(pipeline) && isfield(zpipe{pj},'Y') ) zpipe{pj}.Y=oY; end
      %fprintf('%s,',pipeline{pj});
      if ( subIdxp(pj) ) % force to respect fold guide
                         % sub-set for non-fold aware fns
        subIdx = subsrefDimInfo(zpipe{pj}.di,'dim',{zpipe{pj}.Ydi(1:end-2).name},'idx',foldIdx<0); 
        zpipe{pj+1} = feval(pipeline{pj}, zpipe{pj}, popts{:}, 'subIdx', subIdx, 'verb', verb);
      else        
        zpipe{pj+1} = feval(pipeline{pj}, zpipe{pj}, popts{:}, 'verb', verb);
      end
      parentRan = 1;
    end
  end
      
  % rec the results
  res(ci)=zpipe{end};%rmPrepInfo(zpipe{end});
  oopts=opts;
end
res=reshape(res,size(combo));  % reshape
return

%-------------------------------------------------------------------------
% Expand the list of name,valList sets into a big n-d version!
function [h,di] = makecombs(pipeopts)

% extract the parameters and their values
fn=rfieldnames(pipeopts); fv=rstruct2cell(pipeopts);

% standardize the values
for i = 1:numel(fv)
   if( iscell(fv{i}) && size(fv{i},2)==1 ) fv{i}={fv{i}}; end; % double nest cell arguments
   % cut numeric matrices along the columns
   if(isnumeric(fv{i}) ) fv{i} = num2cell(fv{i},1); end
   if( isempty(fv{i}) ) fv{i} = {[]}; end;
   % special case for cell array of strings
   if( iscell(fv{i}) && ischar(fv{i}{1}) && size(fv{i},2)==1) fv{i}={fv{i}}; end;
   if( ~iscell(fv{i}) ) fv{i} = {fv{i}}; end;
   siz(i) = max(size(fv{i},2),1);
   ind{i} = 1:siz(i);
end

% expand to all combos
if length(ind) > 1, [ind{:}] = ndgrid(ind{:}); end
for i = 1:prod(siz)  % fill in the values
   for j = 1:numel(fv)
      %h(i)=setfield(h(i),fn{j},fv{j}{ind{j}(i)}); % don't work with nesting
      eval(sprintf('h(i).%s = fv{j}{:,ind{j}(i)};', fn{j}));
   end
end
% compres to remove singlentons
incD=find(siz>1);
siz =siz(incD); 
siz(end+1:2)=1;
h=reshape(h,siz);
di = mkDimInfo(siz(siz>1),fn(incD));
for i=1:numel(incD); di(i).vals=fv{incD(i)}; end;
return

%-------------------------------------------------------------------------
function []=testCase()
% make a simple ERP style toy problem
sources={{'gaus' 80 40} ...  % ERP peak
         {'coloredNoise' exp(-[inf(1,0) zeros(1,1) linspace(0,5,40)])} }; % noise signal
n2s=1; y2mix  =cat(3,[.5  n2s],[0  n2s]); % per class ERP magnitude % [ nSrcLoc x nSrcSig x nLab ]
Yl     =sign(randn(100,1));  % epoch labels
z=jf_mksfToy('Y',Yl,'sources',sources,'y2mix',y2mix,'nCh',10,'nSamp',300,'fs',100);

z=jf_import('rand','test','pipe',randn(10,20,30),{'ch','time','epoch'},'Y',sign(randn(30,2)),'Ydi',{'epoch'});

% just test the code
jf_cvtrainPipe(z,{@jf_compKernel @jf_cvtrain},'jf_compKernel.kerType','linear','jf_compKernel.kerParm',[1 2 3])

jf_cvtrainPipe(z,{'jf_compKernel' 'jf_cvtrain'},'jf_cvtrain',struct('outerSoln',0,'objFn','klr_cg'),'subIdx',[]);

% test with diff number csp filters
r=jf_cvtrainPipe(z,{'jf_csp','jf_compKernel','jf_cvtrain'},'jf_cvtrain',struct('outerSoln',0,'reorderC',0),'jf_csp.nfilt',[2 4 6],'subIdx',{'jf_csp'})

% test with 3-d hyper-params
r=jf_cvtrainPipe(z,{'jf_csp','jf_compKernel','jf_cvtrain'},'jf_cvtrain',struct('outerSoln',0,'reorderC',0),'jf_csp',struct('nfilt',[2 4 6],'ridge',[0 10]),'subIdx',{'jf_csp'})

r=jf_cvtrainPipe(z,{'jf_retain','jf_csp','jf_compKernel','jf_cvtrain'},'jf_retain',struct('dim',1,'idx',{{[1 2] [3 4]}}),'jf_cvtrain',struct('outerSoln',0,'reorderC',0),'subIdx',[0 1 0 0])

% alt way to specifiy arguments sets
r=jf_cvtrainPipe(z,{'jf_csp','jf_compKernel','jf_cvtrain'},'jf_cvtrain.outerSoln',0,'jf_cvtrain.reorderC',0,'jf_csp.nfilt',[2 4 6],'jf_csp.ridge',[0 10],'subIdx',{'jf_csp'})


% test with guassian kernel and kerne hyperparameters
jf_cvtrainPipe(z,{@jf_compKernel @jf_cvtrain},'jf_compKernel.kerType','rbf','jf_compKernel.kerParm',(2*CscaleEst(z.X))*10.^([-3:3]))



% compare with non-piped training
z=jf_mksfToy();
z=jf_addFolding(z,'foldSize',50,'nFold',2);
jf_cvtrain(z,'verb',1);
jf_cvtrainPipe(z,{'jf_cvtrain'},'jf_cvtrain.CscaleAll',1,'verb',1);