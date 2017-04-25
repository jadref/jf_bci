function [jobInfo,ranJob]=submitJobs(z,fns,jobInfo,userprefix,nprocess,commonpreprocessing,commonargs,commonpostproc,varargin)
% submit a set of jobs to the cluster for processing
%
% [jobInfo,ranJob]=submitJobs(z,fns,jobInfo,userprefix,nprocess,commonpreprocessing,commonargs,commonpostprocessing,varargin)
%
% Inputs:
%  z - [struct] object to work on
%  fns - {nJobs x 1} cell array of job pipelines to run
%  jobInfo -- [struct] information about jobs already run to prevent re-running completed
%  userprefix -- [str] string to identify this set of jobs
%  nprocess -- [int] number of functions to run in each processs, 0=process per function.  (0)
%  commonpreprocessing -- {} set of common pre-processing steps for all jobs
%  commonargs -- {}
%  commonpostprocessing -- {} post-processing to do after the job has completed ({})
%                       e.g. 'rmPrepInfo' to return only the performance summary results
%  varargin -- remining options go to reval to configure the job settings.  see reval_conf
% Outputs:
%  jobInfo -- [struct] info about the submitted jobs
%  ranJob  -- [bool] flag if we actually submitted a job
if ( nargin<4 ) userprefix=[]; end;
if ( nargin<5 || isempty(nprocess) ) nprocess=0; end;
if ( nargin<6 ) commonpreprocessing={}; end;
if ( ~iscell(commonpreprocessing) ) 
  if ( isempty(commonpreprocessing) ) commonpreprocessing={}; else commonpreprocessing={commonpreprocessing}; end; 
end;
if ( nargin<7 ) commonargs={}; end;
if ( ~iscell(commonargs) ) 
  if ( isempty(commonargs) ) commonargs={}; else commonargs={commonargs}; end; 
end;
if ( nargin<8 ) commonpostproc={}; end;
if ( ~iscell(commonpostproc) ) 
  if ( isempty(commonpostproc) ) commonpostproc={}; else commonpostproc={commonpostproc}; end; 
end;
if ( isempty(jobInfo) ) jobInfo={}; end;
ranJob=false;
% extract info about previously run jobs
ji=0; jobnum={}; expts={}; subjs={}; labels={}; sessions={}; algs={};
for jii=1:numel(jobInfo); 
  ji=ji+1;
  if ( iscell(jobInfo) ) jji=jobInfo{jii}; else jji=jobInfo(jii); end;
  if ( ~isstruct(jji) || isempty(jji) ) continue; end;
  if ( isfield(jji,'job') ) jji=jji.job.description; end;
  if ( isfield(jji,'expt') )
    jobnum{ji}=jii; expts{ji}=jji.expt; subjs{ji}=jji.subj; labels{ji}=jji.label; algs{ji}=jji.alg;
    if ( ~isfield(jji,'session') || isempty(jji.session) ) jji.session=''; end;  sessions{ji}=jji.session; 
    % if ( ischar(jji.alg) ) algs{ji} =jji.alg;
    % elseif ( numel(jji.alg)>1 ) % expand out sets of algs ran at once
    %    ti=1;
    %    if ( iscell(jji.alg) ) algs{ji}=jji.alg{ti}; else algs{ji}=jji.alg(ti); end;
    %    for ti=2:numel(jji.alg);
    %       ji=ji+1;
    %       jobnum{ji}=jii; expts{ji}=jji.expt; subjs{ji}=jji.subj; labels{ji}=jji.label; sessions{ji}=jji.session;
    %       if ( iscell(jji.alg) ) algs{ji}=jji.alg{ti}; else algs{ji}=jji.alg(ti); end;
    %    end
    % else 
    %    warning('dont understand the algname, IGNORED'); algs{ji}='';
    % end       
  else
    expts{ji}='dummy'; subjs{ji}='dummy'; labels{ji}='dummy'; sessions{ji}='dummy'; algs{ji}='dummy';
  end
end;

%----------------------------- filter the list of fns to strip out ones we've run before --------------
keep=true(numel(fns),1);
for algi=1:numel(fns);
   alg=fns{algi}{1};
   if ( numel(z)>1 )
     warning('multi-objects input, only algname is used for matching');
     expt='multi'; subj='multi'; label='multi'; session='multi';
     matchp= strcmp(alg,algs);
   else
     expt=z.expt; subj=z.subj; label=z.label; session=z.session;
     if ( isempty(session) ) session=''; end;
     matchp = strcmp(expt,expts);
     if ( isempty(matchp) ) % check postfix match, (because prev st stripped common prefix)
       for i=1:numel(expts); if(strcmp(expts{i},expt(max(1,end-numel(expts{i})+1):end))) matchp(i)=true; end; end
     end
     matchp = matchp & strcmp(subj,subjs) & strcmp(label,labels);
     if ( ~isempty(sessions) ) matchp = matchp & strcmp(session,sessions); end;
     % match the alg, include recurse into sets of algs per dataset
     for mi=find(matchp); matchp(mi)=any(strcmp(alg,algs{mi})); end
   end
   % only the most recently executed matters
   lastrun= find(matchp,1,'last');
   if ( ~isempty(lastrun) ) 
      if ( iscell(jobInfo) ) jinfo=jobInfo{jobnum{lastrun}}; else jinfo=jobInfo(jobnum{lastrun}); end;
      if ( isfield(jinfo,'job') )
         stat=re_status(jinfo); 
         %if prev started & failed then re-submit
         if( isempty(stat) || (stat.started && stat.failed)) matchp(:)=false; end;
      end
   end
   if ( any(matchp) ) % skip if prev finished      
      keep(algi)=false;
      if ( sum(~keep)==1 ) fprintf('Skipping PREV RUN jobs:'); end;
      fprintf('%15s, ',alg);%,jobnum{find(matchp,1,'last')});
      if ( mod(sum(~keep),5)==0 ) fprintf('\n'); end;
   end
end
if ( any(~keep) ) fprintf('\n'); end;
fns=fns(keep);
fprintf('%d jobs left to run\n',numel(fns));

if( numel(fns)==0 ) ranJob=0; return; end;

%-----------------------------  now actually submit the remaining jobs -------------------------
if ( nprocess>numel(fns) || nprocess==0 ) 
   algIdx=[1:numel(fns) numel(fns)];
else
   if ( nprocess<0 ) nprocess=max(1,round(numel(fns)/abs(nprocess))); end; % neg is num fns/proc
   algIdx=round(linspace(0,numel(fns),nprocess+1));
end
for proci=1:numel(algIdx)-1;
   fprintf('--------------------------\n');
   algi = algIdx(proci)+1:algIdx(proci+1);
   if ( numel(algi)==1 )
      alg=fns{algi}{1};
      fprintf('Submitting: %40s\n',alg);
      description=struct('expt',expt,'subj',subj,'label',label,'session',session,'alg',alg);%store job descript
   else
      alg={};
      for ai=1:numel(algi);
         alg{ai}=fns{algi(ai)}{1};
         fprintf('Submitting: %40s\n',alg{ai});
      end
      description=struct('expt',expt,'subj',subj,'label',label,'session',session,'alg',{alg});%store job descript      
   end
   ranJob=true;
   % mark data with algorithm to run -- so it's available at collection time
   for zi=1:numel(z); if (iscell(z))z{zi}.alg=alg; else z(zi).alg=alg; end; end;

   % actually submit the jobs
   job = reval({'description',description,'nargout',1,'userprefix',userprefix,varargin{:}},...
               'runsubFn','runall',z,commonpreprocessing,commonargs,fns(algi),commonpostproc);  

   if ( iscell(jobInfo) ) jobInfo{end+1}=job; else jobInfo(end+1)=job; end;
   assignin('base','alljobs',jobInfo); % emergency copy
   fprintf('submitted. jobNum: %d, jobID : %s\n',job.conf.jobnum,job.conf.jobid);
end
