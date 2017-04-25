function [z,res]=jf_cv2trainPipe(z,pipeline,varargin)
% double cross validated training of an entire signal processing pipeling
%
%  [z,res]=jf_cv2trainPipe(z,pipeline,varargin)
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
opts = struct('nInner',5,'verb',0,'foldIdxs',[],'zeroLab',0);
cvtrainopts = struct('lossType','bal');
[opts,cvtrainopts,varargin]=parseOpts({opts,cvtrainopts},varargin);

% extract the folding to use
outerfIdxs=opts.foldIdxs;
if ( isempty(opts.foldIdxs) ) 
  if ( isfield(z,'foldIdxs') ) 
    outerfIdxs=z.foldIdxs; 
  else
    outerfIdxs=10;
  end
end
spD=ndims(z.Y); if (isfield(z,'Ydi')) spD=n2d(z.Ydi,'subProb'); end;
if ( isscalar(outerfIdxs) ) 
   nFolds  = outerfIdxs; 
   outerfIdxs= gennFold(z.Y,nFolds,'dim',spD);
end
foldD =ndims(outerfIdxs);
nFolds=size(outerfIdxs,foldD);

% loop over outer folds calling the inner function
Y=z.Y;
fi=1;
for fi=1:size(outerfIdxs,foldD);
   Ytrn = Y; Ytst = Y; 
   if ( spD<=2 ) % N.B. can be more than 1 subproblem & label-dim
      Ytrn(outerfIdxs(:,fi)>0,:)=0;   Ytst(outerfIdxs(:,fi)<0,:)=0;
   elseif( spD==3 )
      Ytrn=repop(Ytrn,'*',outerfIdxs(:,:,fi)<0); 
      Ytst=repop(Ytst,'*',outerfIdxs(:,:,fi)>0);
   else
      error('More than 2 label dimesions not currently supported');
   end
   innerfIdxs = gennFold(Ytrn,opts.nInner,'zeroLab',opts.zeroLab,'dim',spD);
   
   % Inner cv to determine model parameters
   [zi,res.outer(fi)]=jf_cvtrainPipe(z,pipeline,varargin{:},cvtrainopts,'foldIdxs',innerfIdxs,'outerSoln',1,'verb',opts.verb-2);
   % Model parameters are best on the validation set
   opt = res.outer(fi).opt;
   
   % outer-cv performance recording
   res.fold.f(:,:,fi)    =opt.f;
   if ( fi==1 ) % assign+grow of cell-array is weird on Octave
     res.fold.soln         =opt.soln;
   else
     res.fold.soln(:,:,fi) =opt.soln;
   end;
   res.fold.hpIdx(:,:,fi)=opt.hpi;
   for spi=1:size(Y,2);
     res.fold.trnauc(:,spi,fi) =dv2auc(Ytrn(:,spi),opt.f(:,spi));
     res.fold.tstauc(:,spi,fi) =dv2auc(Ytst(:,spi),opt.f(:,spi));
     res.fold.trnconf(:,spi,fi)=dv2conf(Ytrn(:,spi),opt.f(:,spi));
     res.fold.tstconf(:,spi,fi)=dv2conf(Ytst(:,spi),opt.f(:,spi));
     res.fold.trn(:,spi,fi) =conf2loss(res.fold.trnconf(:,spi,fi),cvtrainopts.lossType);
     res.fold.tst(:,spi,fi) =conf2loss(res.fold.tstconf(:,spi,fi),cvtrainopts.lossType);
     
     if ( opts.verb > -1 )
       if ( size(Y,2)>1 ) fprintf('(out%3d/%2d)\t',fi,spi); else; fprintf('(out%3d)\t',fi); end;
       fprintf('%0.2f/%0.2f \t',res.fold.trn(:,spi,fi),res.fold.tst(:,spi,fi));       
       fprintf('\n');
     end   
   end
 end
 szRes=size(res.fold.trnauc);
 res.fold.di = mkDimInfo(szRes,'perf',[],[],'subProb',[],[],'fold',[],[],'dv');
 res.fold.fIdxs=outerfIdxs;
foldD=3;
res.trnconf= sum(res.fold.trnconf,foldD);
res.tstconf= sum(res.fold.tstconf,foldD);
res.trn = mean(res.fold.trn,foldD);
res.trn_se=sqrt(abs(sum(res.fold.trn.^2,foldD)/nFolds-(res.trn.^2))/nFolds);
res.tst = mean(res.fold.tst,foldD);
res.tst_se=sqrt(abs(sum(res.fold.tst.^2,foldD)/nFolds-(res.tst.^2))/nFolds);
res.trnauc = mean(res.fold.trnauc,foldD);
res.trnauc_se=sqrt(abs(sum(res.fold.trnauc.^2,foldD)/nFolds-(res.trnauc.^2))/nFolds);
res.tstauc = mean(res.fold.tstauc,foldD);
res.tstauc_se=sqrt(abs(sum(res.fold.tstauc.^2,foldD)/nFolds-(res.tstauc.^2))/nFolds);
if ( opts.verb > -1 )
  fprintf('(out ave)\t');
  for spi=1:size(res.trn,2);
    fprintf('%0.2f/%0.2f\t',res.trn(:,spi),res.tst(:,spi));
  end
  fprintf('\n');
end

% Record the total stats
info=struct('pipeline',{pipeline},'res',res);
summary='';for pi=1:numel(pipeline); 
  if(ischar(pipeline{pi}))summary=[summary ',' pipeline{pi}];
  else                    summary=[summary ',' func2str(pipeline{pi})];
  end
end
summary=['CV pipeline opt with: ',summary];
z = jf_addprep(z,mfilename,summary,opts,info);
return;

%-------------------------------------------------------------------------
function []=testCase()
z=jf_mksfToy();

jf_cv2trainPipe(z,{@jf_compKernel @jf_cvtrain},'jf_compKernel.kerType','poly','jf_compKernel.kerParm',[1 2 3])
