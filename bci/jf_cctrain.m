function [z,res]=jf_cctrain(zs,pipeline,varargin)
% double cross validated training of an entire signal processing pipeling
%
%  [z,res]=jf_cctrain(zs,pipeline,varargin)
%
% Inputs:
%  zs - struct array of data object(s) or 
%       {z} cell-array of data objects
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
%  dim    -  {str} dimension along which to combine the data-sets for training
%  subIdx - {str}  cell array of pipeline stages which could possibly overfit  ('all')
%                  & hence must be re-run every time the folding changes 
%                  if it is the string 'all' then all pipeline stages overfit
%  foldIdxs -- [numel(zs),nFold] +1/0/-1 indication of which conditions go into training/testing sets
%              OR 'loo' leave one data-set out folding
%  verb -- [int] verbosity level
%  pipename - [str] name to use in the prep recording
%  mergeClassRes - [bool] merge the results from the different classifier runs (1)
%  mergeexcludefn - {str} list of field names not to merge over runs/folds ({'fold' 'di' 'opt'})
%  lossType - [str] type of loss to record/report
%  zcache -- [int] number of z's to store in mem cache if not all already loaded (inf)
opts = struct('nInner',[],'verb',0,'foldIdxs','loo','dim','epoch','zcache',inf);
cvtrainopts = struct('lossType','bal');
[opts,cvtrainopts,varargin]=parseOpts({opts,cvtrainopts},varargin);
if ( nargin<2 || isempty(pipeline) ) pipeline={'jf_cvtrain'}; end;

if ( numel(zs)~=size(zs,1) ) zs=zs(:); end; % ensure col vector

% extract the folding to use
outerfIdxs=opts.foldIdxs;
if ( isscalar(outerfIdxs) ) 
  nFolds  = outerfIdxs; 
  outerfIdxs= gennFold(ones(size(zs,1),1),nFolds);
elseif ( isequal(outerfIdxs,'loo') )
  outerfIdxs = -ones(size(zs,1),size(zs,1)); outerfIdxs(1:size(outerfIdxs,1)+1:end)=1;
end
nFolds=size(outerfIdxs,ndims(outerfIdxs));

if ( opts.verb > -1 )
  fprintf('        \t'); for ci=1:numel(zs);  fprintf('  DS%2d   \t',ci); end; fprintf('\n'); 
end
% loop over outer folds calling the inner function
cached=false(numel(zs),1); 
fi=1;
for fi=1:size(outerfIdxs,2);
   ctrn = true(size(zs,1),1); ctrn(outerfIdxs(:,fi)>=0)=false; 
   ctst = true(size(zs,1),1); ctst(outerfIdxs(:,fi)<=0)=false;
   
   % discard any extra cached data-sets
   cached=false(numel(zs),1); 
   for ci=1:numel(zs); 
     if ( (iscell(zs) && isfield(zs{ci},'X') && ~isempty(zs{ci}.X)) || ...
          (isstruct(zs) && isfeield(zs(ci),'X') && ~isempty(zs(ci).X)) ) cached(ci)=true;
     end
   end
   if ( sum(cached)>opts.zcache ) 
     pos = find(cached & ~ctrn); pos = pos(1:max(1,end-(opts.zcache-sum(ctrn)))); % remove oldest data-sets
     if ( opts.verb > 0 ) fprintf('Removing datasets:'); end;       
     for ci=pos(:)'; 
       if ( opts.verb > 0 ) fprintf('%d ',ci); end;       
       cached(ci)=false;
       if ( iscell(zs) ) zs{ci}.X=[]; zs{ci}.prep=[]; else zs(ci).X=[]; zs(ci).prep=[]; end; 
     end;
     if ( opts.verb > 0 ) fprintf('\n'); end;       
   end
   % load any ctrn's not already loaded...
   pos = find(~cached & ctrn);
   if ( ~isempty(pos) )
     if ( opts.verb>0 ) fprintf('Loading datasets : '); end;
     for ci=pos(:)';
       if ( opts.verb>0 ) fprintf('%d ',ci); end;
       if( iscell(zs) ) zs{ci}=jf_load(zs{ci}); elseif ( isstruct(zs) ) zs(ci)=jf_load(zs(ci)); end;
       cached(ci)=true;
     end
     if ( opts.verb>0 ) fprintf('\n'); end;     
   end

   % concatenate the training conditions
   z=jf_cat(zs(ctrn),'dim',opts.dim,'autoPrune',1);
   if ( ~isempty(opts.nInner) )
     z=jf_addFolding(z,'nFold',opts.nInner);
   end
   
   % Inner cv to determine model parameters
   [zi,resfi]=jf_cvtrainPipe(z,pipeline,varargin{:},cvtrainopts,'outerSoln',1,'verb',opts.verb-2);
   % Model parameters are best on the validation set
   opt = resfi.opt;
   zopt= resfi.opt.zopt;
   if ( opts.verb>-1 ) 
      for spi=1:size(z.Y,2);
         if ( size(z.Y,2)>1 ) fprintf('(cv %3d/%2d)\t',fi,spi); else; fprintf('(cv %3d)\t',fi); end;
         fprintf('%0.2f/%0.2f\t',opt.trnbin(:,spi),opt.tstbin(:,spi));
         fprintf('\n');
      end
      if ( opts.verb>0 ) jf_disp(zopt); end;
   end
   
   % apply the trained pipeline on each of the testing data-sets in turn
   if ( opts.verb > -1 )
      if ( size(z.Y,2)>1 ) fprintf('(out%3d/%2d)\t',fi,1); else; fprintf('(out%3d)\t',fi); end;
   end
   for ci=1:numel(zs);
     if ( iscell(zs) ) zci=zs{ci}; else zci=zs(ci); end
     if ( ~isfield(zci,'X') || isempty(zci.X) ) zci=jf_load(zci); end; % load it if not already done
     zci=jf_follow(zci,opt.zopt,'steps',numel(z.prep)+1:numel(zopt.prep),'verb',opts.verb-2);     
     resci=zci.prep(end).info.res;
     % outer-cv performance recording
     if ( ctrn(ci) )  % training performance for this condition
       res.fold.trnconf(:,:,fi,ci) = resci.tstconf;
       res.fold.trnbin(:,:,fi,ci) = resci.tstbin;
       res.fold.trnauc(:,:,fi,ci) = resci.tstauc;
       if ( opts.verb > -1 ) fprintf('%0.2f/ NA \t',res.fold.trnbin(:,1,fi,ci)); end;
     else
       % testing performance for this condition
       res.fold.tstconf(:,:,fi,ci)=resci.tstconf;     
       res.fold.tstbin(:,:,fi,ci) =resci.tstbin;
       res.fold.tstauc(:,:,fi,ci) =resci.tstauc;     
       if ( opts.verb > -1 ) fprintf(' NA /%0.2f\t',res.fold.tstbin(:,1,fi,ci)); end;
     end
   end
   if( opts.verb>-1 && size(z.Y,2)>1 )
      % print results for other sub-problems
      for spi=2:size(z.Y,2);
         fprintf('\n(out%3d/%2d)\t',fi,spi);
         for ci=1:numel(zs);
            if ( ctrn(ci) )  fprintf('%0.2f/ NA \t',res.fold.trnbin(:,spi,fi,ci));
            else             fprintf(' NA /%0.2f\t',res.fold.tstbin(:,spi,fi,ci)); 
            end;
         end
      end
   end
   clear zci;   
   if ( opts.verb > -1 ) fprintf('\n'); end
 end

 szRes=size(res.fold.trnauc);
 res.fold.di = mkDimInfo(szRes,'perf',[],[],'subProb',[],[],'fold',[],[],'condn',[],[],'dv');
 res.fold.fIdxs=outerfIdxs;
 foldD=3; condnD=4;
res.trnconf= sum(sum(res.fold.trnconf,foldD),condnD);
res.tstconf= sum(sum(res.fold.tstconf,foldD),condnD);
res.trnbin = conf2loss(res.trnconf,cvtrainopts.lossType);
res.tstbin = conf2loss(res.tstconf,cvtrainopts.lossType);
if ( opts.verb > -1 )
  fprintf('============\n(out av)\t');
  for spi=1:size(res.trnbin,2);
    fprintf('%0.2f/%0.2f\t',res.trnbin(:,spi),res.tstbin(:,spi));
  end
  fprintf('\n');
end

% Record the total stats
for ci=1:numel(zs); % summary of the input datasets
  if ( iscell(zs) )
    condn(ci)=struct('expt',zs{ci}.expt,'subj',zs{ci}.expt,'label',zs{ci}.label,'session',zs{ci}.session); 
  else
    condn(ci)=struct('expt',zs(ci).expt,'subj',zs(ci).expt,'label',zs(ci).label,'session',zs(ci).session); 
  end
end;
% results info
info=struct('pipeline',{pipeline},'res',res,'condn',condn);
summary='';for pi=1:numel(pipeline); 
  if(isstr(pipeline{pi}))summary=[summary ',' pipeline{pi}];
  else                   summary=[summary ',' func2str(pipeline{pi})];
  end
end
z.X=[]; % clear the X info as it's useless
summary=['CC pipeline opt with: ',summary];
z = jf_addprep(z,mfilename,summary,opts,info);
return;
%----------------------------------------------------------------------
function testCase()
z1 = jf_mksfToy('Y',ceil(rand(100,1)*2));
z2 = jf_mksfToy('Y',ceil(rand(100,1)*2));
z3 = jf_mksfToy('Y',ceil(rand(100,1)*2));
zs = {z1,z2,z3};
zr=jf_cctrain({z1,z2,z3},[],'verb',2)
zr=jf_cctrain(zs); % leave one out training
zr=jf_cctrain(zs,[],'foldIdxs',eye(numel(zs))*2-1);  % test on 1, train on rest, leave on out
zr=jf_cctrain(zs,[],'foldIdxs',-eye(numel(zs))*2+1); % train on 1, test on rest, leave on in

zr=jf_cctrain(zs,{'jf_cov' ,'jf_cvtrain'}); % test with 2 step pipeline 
zr=jf_cctrain(zs,{'jf_fftfilter' 'jf_cov' 'jf_cvtrain'},'jf_fftfilter.bands',[1 5 15 20]','nInner',2); % test with 2 step pipeline, with options

% try with dynamic loading
for ci=1:numel(zs); zs{ci}.X=[]; end;
zr=jf_cctrain(zs)
zr=jf_cctrain(zs,[],'zcache',2); % with ds removal
