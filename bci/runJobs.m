function [allres,ranjob]=runJobs(z,fns,allres,debugp,randstate,commonpreprocessing,commonargs,commonpostprocessing)
% run a set of different classifiers on the data-set z
%
% [allres,ranjob]=runJobs(z,fns,allres,debugp,randstate,commonpreprocessing,commonargs,commonpostprocessing)
%  
%   commonpreprocessing - run once at start of the set
%   commonargs          - added to each sub-call to runsubFn
%   commonpostprocessing- run after job has completed, e.g. 'rmPrepInfo'
ranjob=false;
if ( nargin < 3 || isempty(allres) ) allres={}; end;
if ( nargin < 4 || isempty(debugp) ) debugp=0; end;
if ( nargin<5 || isempty(randstate) ) randstate=0; end;
if ( nargin<6 ) commonpreprocessing={}; end;
if ( ~iscell(commonpreprocessing) ) 
  if ( isempty(commonpreprocessing) ) commonpreprocessing={}; else commonpreprocessing={commonpreprocessing}; end; 
end;
if ( nargin<7 ) commonargs={}; end;
if ( ~iscell(commonargs) ) 
  if ( isempty(commonargs) ) commonargs={}; else commonargs={commonargs}; end; 
end;
if ( nargin<8 ) commonpostprocessing={}; end;
if ( ~iscell(commonpostprocessing) ) 
  if ( isempty(commonpostprocessing) ) commonpostprocessing={}; else commonpostprocessing={commonpostprocessing}; end; 
end;
if ( ~iscell(fns) ) fns={{fns}}; elseif ( ~iscell(fns{1}) ) fns={fns}; end;
expts={}; subjs={}; labels={}; sessions={}; algs={};
for ji=1:numel(allres); 
  if ( iscell(allres) ) jji=allres{ji}; else jji=allres(ji); end;
  if ( ~isstruct(jji) || isempty(jji) ) continue; end;
  if ( isfield(jji,'job') ) jji=jji.job.description; end;
  if ( isfield(jji,'description') ) jji=jji.description; end;
  if ( isfield(jji,'expt') )
    expts{ji}=jji.expt; subjs{ji}=jji.subj; labels{ji}=jji.label; algs{ji}=jji.alg;
    if ( ~isfield(jji,'session') || isempty(jji.session) ) sessions{ji}=''; 
    else sessions{ji}=jji.session; 
    end;
  else
    expts{ji}='dummy'; subjs{ji}='dummy'; labels{ji}='dummy'; sessions{ji}='dummy'; algs{ji}='dummy';
  end
end

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
     matchp = matchp & strcmp(subj,subjs);
     if ( any(matchp) ) 
        tmpmatchp = matchp & strcmp(label,labels);
        % try prefix only match
        if ( any(tmpmatchp) ) matchp=tmpmatchp; 
        else                  matchp= matchp & strncmp(label,labels,numel(label)); 
        end;
     end
     if ( ~isempty(sessions) ) matchp = matchp & strcmp(session,sessions); end;
     % match the alg, include recurse into sets of algs per dataset
     for mi=find(matchp); matchp(mi)=any(strcmp(alg,algs{mi})); end
   end
   % only the most recently executed matters
   lastrun= find(matchp,1,'last');
   if ( any(matchp) ) % skip if prev finished
      keep(algi)=false;
      if ( sum(~keep)==1 ) fprintf('Skipping PREV RUN jobs:'); end;
      fprintf('%15s, ',alg);%,jobnum{find(matchp,1,'last')});
      if ( mod(sum(~keep),5)==0 ) fprintf('\n'); end;
   end
end
fns=fns(keep);
fprintf('\nRunning remaining %d jobs: ',numel(fns));
for fi=1:numel(fns); fprintf('%s ',fns{fi}{1}); end;
fprintf('\n');

if ( numel(fns)==0 ) ranJob=0; return; end;

% --------------------- run the remaining algorithms -----------------------------------------
% run the common-preprocessing once
if ( ~isempty(commonpreprocessing) )
   z=runsubFn('preprocess',z,commonpreprocessing{:});
   jf_disp(z)
end
if ( isempty(fns) || (numel(fns)==1 && (isempty(fns{1}) || isempty(fns{1}{1}))) ) 
  allres=z;
  return; 
end;
for fni=1:numel(fns);
   alg=fns{fni}{1}; % label
   fprintf('\n-------------------------- %20s --------------------------\n',alg);
   % mark data with algorithm we're running
   for zi=1:numel(z); if ( iscell(z) ) z{zi}.alg=alg; else z(zi).alg=alg; end; end;
   if ( ~isempty(randstate) ) rand('state',randstate); end; % BODGE: for rand number gen reset
   if ( debugp ) 
      res=runsubFn(fns{fni}{2},z,fns{fni}{3:end-1},commonargs{:},fns{fni}{end}); % call with spec parameters
   else % soft-fail
      try;
         res=runsubFn(fns{fni}{2},z,fns{fni}{3:end-1},commonargs{:},fns{fni}{end}); % call with spec parameters
      catch; % print full stack trace for debugging and continue
         err=lasterror;
         msg=['Error: ' err.message];
         if ( isfield(err,'stack') )
            for i=1:numel(err.stack); 
               msg=sprintf('%s\nIn <a href="error:%s,%d,1">%s at %d</a>',...
                           msg,err.stack(i).file,err.stack(i).line,err.stack(i).name,err.stack(i).line);
            end
         end
         fprintf('%s\n',msg);
         continue; 
      end;
   end

   [res.alg]=deal(alg);
   if ( ~isempty(commonpostprocessing) ) res=runsubFn('preprocess',res,commonpostprocessing{:}); end;
   if ( isempty(allres) ) allres={res}; % default to cell array   
   elseif ( iscell(allres) )  allres{end+1}=res; 
   elseif ( isstruct(allres) )  allres(end+1)=orderfields(res,allres); 
   else error('huh! whats res?');
   end;
   
   assignin('base','allres',allres); % emergency save
   ranjob=true;
end
return;

