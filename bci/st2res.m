function [res]=st2res(st,nsessd)
% [res]=st2res(st,nsessd)
if ( nargin<2 ) nsessd=1; end;
% convert results summary as extracted by extRes back into a results summary structure
if ( ~isfield(st,'expt') ) st.expt=''; end;
res={}; 
expts={}; subjs={}; sessions={}; labels={}; % indices for matching info
idx={}; for di=1:max(numel(st.di)-1,ndims(st.X)); idx{di}=1; end;
dsD=n2d(st,'dataset');    labD=n2d(st,'label');   algD=n2d(st,'algorithm');
for i=1:numel(st.X); 
  if ( st.X(i)>0 )
    [idx{1:ndims(st.X)}]=ind2sub(size(st.X),i);
    % re-construct the full data-set name
	 if ( ~isfield(st,'expt') ) st.expt=''; end;
    dsname =[st.expt st.di(dsD).vals{idx{dsD}}];
    ii = find(dsname=='/');
    % split back into, expt/subj/session
    expt   = dsname(1:ii(end-nsessd)-1);
    subj   = dsname(ii(end-nsessd)+1:ii(end-nsessd+1)-1);
    session= dsname(ii(end-nsessd+1)+1:end);
    label  =st.di(labD).vals{idx{labD}};
    alg    =st.di(algD).vals{idx{algD}};
    if ( isfield(st,'prep') && any(strcmp('jf_cat',{st.prep.method})) ) % strip trailing prep postfix
       ii=regexp(alg,'_[0-9._]*$');
       if ( ~isempty(ii) ) alg=alg(1:ii-1); end;
    end
    % check if we have already got an entry for this combination
    matchp = strcmp(expt,expts) & strcmp(subj,subjs) & strcmp(label,labels);
    if ( ~isempty(sessions) ) matchp = matchp & strcmp(session,sessions); end;    
    if ( any(matchp) ) 
       mi=find(matchp,1); 
       if ( iscell(res{mi}.alg) ) res{mi}.alg{end+1}=alg; 
       else                       res{mi}.alg       ={res{mi}.alg alg};
       end
    else % add a new one
       res{end+1}=struct('expt',expt,'subj',subj,'label',label,'session',session,'alg',alg);
       expts{end+1}=expt; subjs{end+1}=subj; labels{end+1}=label; sessions{end+1}=session;
    end
  end
end
