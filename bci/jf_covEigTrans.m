function [z]=jf_covEigTrans(z,varargin);
% transform the eigenvalue structure of a set of covariance matrices, e.g. by log
%
% Options:
%  type -- 'str', one of:
%          'sphere' -- sperichical transform, \lambda' = 1
%          'sqrt'   -- square root: \lambda' = sqrt(\lambda)
%          'log'    -- log:         \lambda' = log(\labmda)
%          'keep' -- only the strongest param directions are retained
%          'ridge' -- add ridge of strength param of this strength
%  param -- [float] parameter for the given transformation type
%
% See Also: covEigTrans
opts=struct('subIdx',[],'type','sphere','param',[],'verb',0);
opts=parseOpts(opts,varargin);
if ( any(strcmp(opts.type,{'oas','rblw'})) && isempty(opts.param) ) % infer #examples for history
   covs = m2p(z,{'jf_cov','jf_taucov'},1,0);
   covs(covs==0)=[];
   if ( ~isempty(covs) ) 
      covinfo = z.prep(covs).info;
      opts.param = numel(covinfo.accdi(1).vals);
   end
end
z.X=covEigTrans(z.X,opts.type,opts.param);
if ( iscell(opts.type) ) summary=[sprintf('%s+',opts.type{1:end-1}) opts.type{end}]; else summary=opts.type; end;
z =jf_addprep(z,mfilename,summary,opts,[]);
return;
%---------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
z=jf_cov(z);
t=jf_covEigTrans(z);
jf_disp(t)
clf;image3d(clsAve(t.X,t.Y),[3]);

