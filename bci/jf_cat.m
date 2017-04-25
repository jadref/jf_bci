function [z]=jf_cat(zz,varargin);
% concatenate a set of annotated matrices
%
% [z]=cat(zz,varargin)
% Inputs:
%  zz -- [Nx1 struct] OR [Nx1 cell] array of sub-structures to concatenate
% Options:
%  dim       -- dimension of zz to cat along ('epoch')
%  verb      -- [int] verbosity level
%  autoPrune -- [bool] auto-prune size of bits to match? (false)
%  summary   -- [str] additional text about why we did this
%  di        -- dimension information for the new dimension to be made,
opts=struct('dim','epoch','verb',0,'summary','','autoPrune',0,'di',[]);
opts=parseOpts(opts,varargin);

if ( iscell(zz) ) 
   % make sure they have the same field structure before we cat them together
   rec=true(1,numel(zz));
   fns={}; for i=1:numel(zz); if(~isstruct(zz{i})) rec(i)=false; continue;end; fnsi=fieldnames(zz{i}); fns=unique({fnsi{:},fns{:}}); end;
   zz=zz(rec); % remove non-data elements
   for i=1:numel(zz); for fi=1:numel(fns); if( ~isfield(zz{i},fns{fi}) ) zz{i}.(fns{fi})=[]; end;  end; end;
   zz=cat(1,zz{:}); 
end   
% Now attempt to cat the blocks info together
dim=n2d(zz(1),opts.dim,0,0);
newDim=false;
if ( dim==0 ) % new dim to be made
   newDim=true;
   dim=numel(zz(1).di);   
end

% check size compatability
szX=[]; for i=1:numel(zz); szXi=size(zz(i).X); szX(:,end+1:size(szXi,2))=1; szXi(end+1:size(szX,2))=1; szX(i,:)=szXi; end; szX(:,end+1:dim)=1;
if ( numel(zz)>1 && any(any(diff(szX(:,[1:dim-1 dim+1:end]),[],1)~=0)) )
   fprintf('Inconsistent sizes: \n');
   for i=1:size(szX,1); 
      fprintf(' size(zz(%2d).X)=[%s]\n',i,[sprintf('%dx',szX(i,1:end-1)) sprintf('%d',szX(i,end))]);
   end
   if ( opts.autoPrune )
      idx={}; for d=1:ndims(zz(1).X); idx{d}=1:min(szX(:,d)); end;
      for i=1:numel(zz); idx{dim(1)}=1:size(zz(i).X,dim(1)); zz(i).X=zz(i).X(idx{:}); end;
   else
      fprintf('try;\n\t\t for i=1:numel(zz); zz(i).X(:,min(szX(:,2))+1:end,:)=[];end;\n');
      fprintf('Please fix and type return\n');
      keyboard;
   end
   % fix the di to the new size
   fprintf('Fixed, size now:\n');
   for i=1:numel(zz);
      szXi=size(zz(i).X); szX(i,:)=[szXi ones(1,size(szX,2)-numel(szXi))];
      fprintf(' size(zz(%2d).X)=[%s]\n',i,[sprintf('%dx',szX(i,1:end-1)) sprintf('%d',szX(i,end))]);
      for d=1:ndims(zz(i).X);
         zz(i).di(d).vals(szX(i,d)+1:end)=[];
         if(~isempty(zz(i).di(d).extra)) zz(i).di(d).extra(szX(i,d)+1:end)=[];end
      end;
   end
end;

if ( opts.verb >0 ) fprintf('Concatenating the bits...'); end;
z    = zz(1);                 % start with 1st entry at template
z.X  = [];
z.X  = cat(dim,zz.X);         % cat the data
z.di = catDimInfo(dim,zz.di); % cat the meta-info
if ( newDim ) % setup new dims meta info
  if ( ~isempty(opts.di) )
    if ( isstruct(opts.di) ) z.di(dim)=opts.di; 
    elseif( ischar(opts.di) && size(z.X,dim)==1 ) z.di(dim).vals={opts.di};
    elseif( numel(opts.di)>=size(z.X,dim) )      z.di(dim).vals=opts.di(1:size(z.X,dim)); 
    end;
  end
  if( ischar(opts.dim) )  z.di(dim).name=opts.dim; end;
end

% ensure the vals along the cat'd dim are unique
if ( numel(unique(z.di(dim).vals))~=numel(z.di(dim).vals) )
   % record the orginal values
   for i=1:numel(z.di(dim).extra); z.di(dim).extra(i).oval=z.di(dim).vals(i); end;

   if ( isnumeric(z.di(dim).vals) ) % prefix with a number reflecting the source number
      ti=0;
      for si=1:numel(zz); % re-val
         idx=ti+(1:numel(zz(si).di(dim).vals));
         z.di(dim).vals(idx) = z.di(dim).vals(idx)+[z.di(dim).extra(idx(1)).src]*1000; 
         ti=ti+numel(zz(si).di(dim).vals);
      end
   elseif( iscell(z.di(dim).vals) )
      if ( ischar(z.di(dim).vals{1}) ) % strings add postfix
         for i=1:numel(z.di(dim).vals); 
            z.di(dim).vals{i}=sprintf('%s_%.3g',z.di(dim).vals{i},z.di(dim).extra(i).src); 
         end;
      elseif ( isnumeric(z.di(dim).vals{1}) ) % numbers add prefix
         for i=1:numel(z.di(dim).vals);
            z.di(dim).vals{i}= z.di(dim).vals{i} + 1000*z.di(dim).extra(i).src;
         end
      elseif ( isstruct(z.di(dim).vals{1}) )  % struct -- add new field
         for i=1:numel(z.di(dim).vals);
            z.di(dim).vals{i}.src=z.di(dim).extra(i).src;
         end
      else % something else -- bugger!
         warning('renamed values to numbers');
         z.di(dim).vals = 1:size(z.X,dim); % re-val
      end
   end
end
if ( opts.verb > 0 ) fprintf('done'); end;

if ( isfield(zz,'Y') && ~isempty(zz(1).Y) ) % cat the Y if necessary
   if ( isfield(zz,'Ydi') && ~isempty(zz(1).Ydi) )
      t=strmatch(zz(1).di(dim).name,{z.Ydi(1:end-1).name},'exact'); 
      if( ~isempty(t) ) 
         t=t(1);
         z.Y=cat(t,zz.Y);
         z.Ydi=catDimInfo(t,zz.Ydi);
         if ( isfield(zz,'foldIdxs') ) z.foldIdxs=cat(t,zz.foldIdxs); end;   
      end
   elseif ( strmatch(zz.di(dim).name,'epoch') )
     t=max(size(zz.Y)>1); % use last non-singlenton dimension
     z.Y=cat(t,zz.Y);
     if ( isfield(zz,'foldIdxs') ) z.foldIdxs=cat(t,zz.foldIdxs); end;   
   end
end

% cat the prep if necessary
if ( isfield(zz,'prep') )
   prep=zz(1).prep;
   for pi=1:numel(z.prep);
      piinfo=prep(pi).info; prep(pi).info=struct(); % clear the info
      for di=2:numel(zz) % get set of preps at this step
         if ( numel(zz(di).prep)<pi ) continue; end;
         if ( isfield(prep(pi).info,'src') && isfield(prep(pi).info.src,'info') )
            prep(pi).info.src.info{di}=zz(di).prep(pi).info;
         else%if ( ~isequal(piinfo,zz(di).prep(pi).info) ) %TODO: This can be *really* slow
            for i=1:di; prep(pi).info.src.info{i}=zz(i).prep(pi).info; end;
         end
         if ( isequal(prep(pi).method,'jf_cat_prep') )
            prep(pi).info.src.method{di}=zz(di).prep(pi).method;
         elseif ( ~isequal(prep(pi).method,zz(di).prep(pi).method) )
            [prep(pi).info.src.method{1:di-1}]=deal(prep(pi).method);
            prep(pi).info.src.method{di}=zz(di).prep(pi).method;
            prep(pi).method='jf_cat_prep';
         end
         if ( isequal(prep(pi).summary,'cat_prep') )
            prep(pi).info.src.summary{di}=zz(di).prep(pi).summary;
         elseif ( ~isequal(prep(pi).summary,zz(di).prep(pi).summary) )
            [prep(pi).info.src.summary{1:di-1}]=deal(prep(pi).summary);
            prep(pi).info.src.summary{di}=zz(di).prep(pi).summary;
            prep(pi).summary='cat_prep';
         end
         if ( iscell(prep(pi).opts) )
            prep(pi).opts{di}=zz(di).prep(pi).opts;
         elseif ( ~isequal(prep(pi).opts,zz(di).prep(pi).opts) )
            tmp=prep(pi).opts; prep(pi).opts=cell(0);
            [prep(pi).opts{1:di-1}]=deal(tmp);
            prep(pi).opts{di}    =zz(di).prep(pi).opts;
         end
      end;       
      if( isempty(prep(pi).info) || (isstruct(prep(pi).info) && isempty(fieldnames(prep(pi).info)))  )
         prep(pi).info = piinfo;
      elseif ( isfield(prep(pi).info,'src') && ~isfield(prep(pi).info.src,'info') )
         prep(pi).info.src.info=piinfo;
      end
   end
   z.prep = prep;
end

info=[];
info.testFn=''; % null test function
% record basic info about the source structures
expts={};subjs={};sessions={}; labels={};
for i=1:numel(zz);
  if( isfield(zz(i),'expt') )    expts(i)   ={zz(i).expt}; end;
  if( isfield(zz(i),'subj') )    subjs(i)   ={zz(i).subj}; end;
  if( isfield(zz(i),'label') )   labels(i)  ={zz(i).label}; end;
  if( isfield(zz(i),'session') ) sessions(i)={zz(i).session}; end;
end
info.src=struct();
if ( ~isempty(expts) )    info.src.expt=expts; end;
if ( ~isempty(subjs) )    info.src.subj=subjs; end;
if ( ~isempty(labels) )   info.src.label=labels; end;
if ( ~isempty(sessions) ) info.src.session=sessions; end;

summary = sprintf('%d datasets along %s',numel(zz),z.di(dim).name);
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
z = jf_addprep(z,mfilename,summary,opts,info);
z.summary = jf_disp(z); % re-write the summary info
return;
%-------------------------------------------------------------------------------
X = randn(100,100,100); Y=sign(randn(100,1)); 
di=mkDimInfo(size(X),'ch',[],[],'time','ms',10*(1:size(X,2)),'epoch',[],[]);
z = jf_import('a','b','c',X,di,'Y',Y);

zz= jf_cat({z,z});
