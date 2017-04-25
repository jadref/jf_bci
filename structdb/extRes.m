function [st]=extRes(st,resdb,dsnmfmtstr,fieldnm,fielddi)
% extract classification per info from the given dataset
%
% [st]=extRes(st,resdb,dsnmfmtstr,fieldnm,fielddi)
%
if ( nargin < 2 ) resdb=st; st=[]; end;
if ( nargin < 3 || isempty(dsnmfmtstr) ) dsnmfmtstr='%s/%s/%s'; end;
if ( nargin < 4 ) fieldnm=[]; end;
if ( nargin < 5 ) fielddi=[]; end;
% get the data table
nres=0;nmcres=1;
if ( isempty(st) ) 
  st.di=mkDimInfo(zeros(1,4),'subProb',[],[],'algorithm',[],[],'label',[],[],...
                  'dataset',[],[],[],'cr');
end
% get the currently used names
dsnms =st.di(n2d(st,'dataset')).vals;   if (isempty(dsnms))  dsnms ={};  end;
algnms=st.di(n2d(st,'algorithm')).vals; if (isempty(algnms)) algnms={}; end;
labels=st.di(n2d(st,'label')).vals;     if (isempty(labels)) labels={}; end;
if ( ~isempty(fielddi) ) 
  if ( isstruct(fielddi) ) st.di(1)=fielddi;
  elseif ( iscell(fielddi) ) st.di(1).vals=fielddi(:)'; 
  end;
end 
spnms=st.di(1).vals;   if (isempty(spnms))  spnms ={};  end;

% flatten cell-of-cell arrays if needed
if (iscell(resdb) ) 
   nres={}; 
   for ri=1:numel(resdb); 
      if ( iscell(resdb{ri}) ) nres={nres{:} resdb{ri}{:}}; else nres={nres{:} resdb{ri}}; end; 
   end; 
   resdb=nres;
end

if ( ~isfield(st,'info') ) st.info=struct('expt',[],'summary',[]); end

dsi=1;algi=1;spi=1;
for resi=1:numel(resdb);
  fieldnmi=fieldnm;
  if ( iscell(resdb) ) dsres=resdb{resi}; else dsres=resdb(resi); end;
  if ( isempty(dsres) || ~isfield(dsres,'prep') ) continue; end;
  if ( isfield(dsres,'alg') ) alg=dsres.alg; else alg=''; end;
  if ( ~isfield(dsres.prep(end).info,'res') ) 
    warning(sprintf('No results structure: #%d alg=%s',resi,alg)); continue; 
  end
  if ( isfield(dsres.prep(end).info.res,'opt') && isfield(dsres.prep(end).info.res.opt,'res') )
    res=dsres.prep(end).info.res.opt.res; % use the opt-res
  else
    res=dsres.prep(end).info.res;
  end
  expt=dsres.expt;
  if(isfield(dsres,'session')) session=dsres.session; else session='';dsres.session=session;end
  if(isempty(session))session='';dsres.session=session;end;
  subj=dsres.subj;
  label=dsres.label;
  summary=jf_disp(dsres);

  mi=strmatch(alg,algnms,'exact'); % get alg
  if(~isempty(mi)) algi=mi; else algnms{end+1}=alg; algi=numel(algnms); end; 

  dsnm=sprintf(dsnmfmtstr,dsres.expt,dsres.subj,dsres.session);
  %if ( dsnm(end)=='/' ) dsnm=dsnm(1:end-1); end; % remove trailing slash..
  mi=strmatch(dsnm,dsnms,'exact');  % get dataset
  if( ~isempty(mi) ) dsi=mi; 
  else 
    % check for a postfix match (because prev st striped common prefix)
    dsi=0;
    for di=1:numel(dsnms); 
      if(strcmp(dsnms{di},dsnm(max(1,end-numel(dsnms{di})+1):end))==1) dsi=di; break;end;
    end;
    if ( dsi==0 ) dsnms{end+1}=dsnm; dsi=numel(dsnms); end;
  end;
  % remove alg information in the label if it's there
  algInLab=strfind(label,alg);
  if( ~isempty(alg) && ~isempty(algInLab) ) 
    rng=algInLab+[-1 numel(alg)]; 
    if ( rng(1)>1 && label(rng(1))=='_' ) rng(1)=rng(1)-1; end;
    if ( rng(2)<numel(label) && label(rng(2))=='_') rng(2)=rng(2)+1; end;
    label=label([1:rng(1) rng(2):end]); 
  end; % remove alg info
  mi=strmatch(label,labels,'exact'); % get prep/label
  if(~isempty(mi)) labi=mi; else labels{end+1}=label; labi=numel(labels); end
  
  % get multi-class results if available
  if ( isempty(fieldnmi) && isfield(res,'tstcr') && sum(size(res.tstcr)==1)==ndims(res.tstcr)-1 )
    spnm='mc';
    mi=strmatch(spnm,spnms); % get alg
    if(~isempty(mi)) spi=mi; else spnms{end+1}=spnm; spi=numel(spnms); end; 
    [st.X(spi,algi,labi,dsi),optI]=max(res.tstcr);
    if ( isfield(res,'tstcr_se') )
       st.X_se(spi,algi,labi,dsi)=res.tstcr_se(optI);
    end
    if ( isstruct(st.info) ) 
       st.info.optI(dsi,labi,spi,algi)=[]; % extra info
       st.info.hpNms{dsi,labi,spi,algi}=''; 
    end
  elseif ( isempty(fieldnmi) || strcmp(fieldnmi,'tstbin') )
    if ( isfield(res,'tstcr') ) fieldnmi='tstcr'; spnm='mc';% multi-class if there overrides
	 elseif (isfield(res,'tst') ) fieldnmi='tst';  % simple tst overrides
    else fieldnmi='tstbin'; 
    end;
    if ( isfield(res,'di') ) 
      spD=n2d(res.di,'subProb',0,0); if ( spD==0 ) spD=n2d(res.di,'SubProb',0,0); end;
      % BODGE: for multi-perf measures
      if ( (spD==0 || size(res.(fieldnmi),spD)==1) && size(res.(fieldnmi),1)>1 ) spD=1; end
    else
      spD=2;
    end
    if( isfield(res,'opt') && isfield(res.opt,fieldnmi)) % use opt info if available
      fieldvals=getfield(res.opt,fieldnmi);
      dvsz=size(fieldvals); hpD=setdiff(1:ndims(fieldvals),spD);
      optI=1:size(fieldvals,spD);
    else
      fieldvals=getfield(res,fieldnmi);
      dvsz=size(fieldvals); hpD=setdiff(1:ndims(fieldvals),spD);
      [ans optI]=mmax(fieldvals,hpD);
    end
    %if ( numel(optI)==2 ) optI(end)=[]; end; % deal with binary problems?
    if ( isfield(res,'di') && spD ) 
      tspnms=res.di(spD).vals; if ( ischar(tspnms) ) tspnms={tspnms}; end;
      hpnms={res.di(hpD).name};
    elseif ( numel(optI)==1 ) tspnms={'1 v 2'}; hpnms={}; 
    elseif ( prod(dvsz(hpD))==1 ) % No HP's
      tspnms={}; 
      if ( spD ) 
         for spi=1:dvsz(spD); tspnms{spi}=num2str(spi); end; 
      end;
      hpnms={};
    else
      warning('dont know subproblem names');
    end;
    nSp=1; if( spD ) nSp=size(fieldvals,spD); end;
    for spii=1:nSp;
      spnm=spii;
      if ( numel(tspnms)==nSp ) spnm=tspnms(spii); if(iscell(spnm)) spnm=spnm{:}; end;end
      if ( isnumeric(spnm) ) spnm=sprintf('%d',spii); end;
      mi=strmatch(spnm,spnms,'exact'); % get alg
      if(~isempty(mi)) spi=mi; else spnms{end+1}=spnm; spi=numel(spnms); end; 
      % store the data and some info
      st.X(spi,algi,labi,dsi)=fieldvals(optI(spii)); % the raw data
      hpIs={};[hpIs{1:ndims(fieldvals)}]=ind2sub(dvsz,optI(spii)); hpIs=cell2mat(hpIs); hpIs=hpIs(hpD); 
      hpNms={'C'}; if( isfield(res,'di') ) hpNms={{res.di(hpD).name}}; end;
      if ( isstruct(st.info) ) % extra info
         if ( numel(st.info)>1 ) % convert to new format
            oi=st.info;
            st.info=struct('optI',{cell(size(oi))},'hpNms',{cell(size(oi))});
            for i=1:numel(oi);st.info.optI{i}=oi(i).optI; st.info.hpNms{i}=oi(i).hpNms; end; 
            clear oi;
         end
         %st.info.optI{dsi,labi,spi,algi} =hpIs; 
         %st.info.hpNms{dsi,labi,spi,algi}=hpNms; 
         if( any(isfield(res,{'optI','hpNms'})) ) st.info=rmfield(st.info,{'optI','hpNms'}); end;% BODGE: don't save...
         if ( isfield(res,[fieldnmi '_se']) )
            fieldvals_se = getfield(res,[fieldnmi '_se']);
            if ( isequal(size(fieldvals_se),fieldvals) ) 
               st.info.([fieldnmi '_se'])(spi,algi,labi,dsi)=fieldvals_se(optI(spii));
            end
         end
         if ( isfield(res,'tstauc') && isequal(size(res.tstauc),size(fieldvals)) )
            st.info.tstauc(spi,algi,labi,dsi)=res.tstauc(optI(spii));
         end
         st.info.expt{spi,algi,labi,dsi}=expt;
         st.info.summary{spi,algi,labi,dsi}=summary; 
      end
    end
  else
    if ( strcmp(fieldnmi,'X') )
      fieldvals=dsres.X;
    elseif ( isfield(res,fieldnmi) )  % other field name to get      
      fieldvals=getfield(res,fieldnmi);
    else
      warning('%d) Couldnt find field : %s, skipping',resi,fieldnmi);
      continue;
    end
    if( ~(isfield(st,'X') && iscell(st.X)) &&(sum(size(fieldvals)>1)<=1 || numel(fieldvals)==numel(st.di(1).vals)))
      st.X(:,algi,labi,dsi)=fieldvals(:);
    else st.X{1,algi,labi,dsi}=fieldvals;
    end;
    st.info.expt{spi,algi,labi,dsi}=expt;
    st.info.summary{spi,algi,labi,dsi}=summary;
  end
end

% post-process dsnms to remove any common prefix and put it into the expt
dsD=n2d(st,'dataset'); 
vals=st.di(dsD).vals;
prefix=dsnms{1}; nPre=numel(prefix); 
for j=1:numel(dsnms);
  val=dsnms{j}; 
  len=min(numel(val),nPre);
  nPre=min(nPre,find([prefix(1:len)~=val(1:len) true],1)-1);
end
if ( nPre<len ) % don't remove all
  for j=1:numel(dsnms); dsnms{j}=dsnms{j}(nPre+1:end);end;
  if ( isfield(st,'expt') )
	 st.expt=[prefix(1:nPre) st.expt];
  else
	 st.expt=prefix(1:nPre);
  end
else
   st.expt='';
end

% update the dimInfo for the new values
st.di(1).vals=spnms;
st.di(2).vals=algnms;
st.di(3).vals=labels;
st.di(4).vals=dsnms;
% st=mkDimInfo(size(st.X),'subProb',[],spnms,'algorithm',[],algnms,'label',[],labels,...
%                 'dataset',[],dsnms,[],'cr');

% st.di=mkDimInfo(size(st.X),...
%                 'dataset',[],exptnms,...
%                 'label',[],labels,...
%                 'subProb',[],spnms,...
%                 'algorithm',[],algnms,[],'cr');
return
%---------------------------------------------------------------------
function testCase()
st=extRes([],datasetres);

% sort by mean performance over all algorithms
[ans,si]=sort(mean(st.X,2),'descend');
st.X=st.X(si,:); st.di(1).vals=st.di(1).vals(si); st.di(1).extra=st.di(1).extra(si);

selAlgs={'cspKLR' 'bpcspKLR' 'whtcovKLR' 'bpwhtcovKLR' 'welch50KLR'}; % no freq stuff

% Plot the results grouped by expt type
[idx,sdi]=subsrefDimInfo(st.di,'dim','algorithm','vals',selAlgs);
clf;plot(st.X(:,idx{2}));ylabel('Classification Rate (%)')
legend(st.di(2).vals(idx{2}));
