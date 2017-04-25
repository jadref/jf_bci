function [z]=jf_retain(z,varargin);
% function to remove elements of the jf-data-struct
%
% Options:
%  dim  - dimension to reject along
%  vals - element values to remove/keep
%  idx  - element indicies to remove/keep
%         N.B. negative idx index from the end back
%  range- idx/vals specifies a range to remove. which is either:
%         'before','after','between','outside' the spec elements
%  mode - {'reject','retain'} reject or retain the indicated elements
%  valmatch - {'exact','regexp','nearest'} how do we match values ('nearest')
%  outorder -- [str] what order are the outputs in, one-of:       ('input')
%                input - same order as input,  idx - same order as idx/vals
%  summary -- optional additional information summary string
subsrefOpts=struct('dim',[],'vals',[],'idx',[],...
                   'range',[],'mode','retain','valmatch','nearest','outorder','input');
opts = struct('summary','','testFn',[],'subIdx',[],'verb',0);
[opts,subsrefOpts,varargin]= parseOpts({opts,subsrefOpts},varargin);
szX=size(z.X);

if ( iscell(subsrefOpts.dim) ) % extract the dims being modified
   for di=1:numel(subsrefOpts.dim);
      tmp=strmatch(subsrefOpts.dim{di},{z.di.name});
      if ( numel(tmp)>1 ) tmp=strmatch(subsrefOpts.dim{di},{z.di.name},'exact'); end;
      dim(di)=tmp;
   end
elseif ( ischar(subsrefOpts.dim) ) dim=strmatch(subsrefOpts.dim,{z.di.name}); 
else dim=subsrefOpts.dim; 
end;
if ( dim < 0 ) dim=dim+ndims(z.X)+1; end;

% call the DimInfo function to do the actual index computation
[Xidx,di]=subsrefDimInfo(z.di,subsrefOpts,varargin{:});

szX(end+1:max(dim))=1;
ind=false([szX(dim(1)),1]);ind(Xidx{dim(1)})=true; % bool keep,remove indicator
z.X   =z.X(Xidx{:});
odidim= z.di(dim);
z.di  = di;
if(isempty(z.di(dim(1)).name)) dname=num2str(dim(1)); else dname=z.di(dim(1)).name;end
switch (subsrefOpts.mode); % make a nice summary string
 case 'reject';  
  oidx = find(~ind); nidx    = numel(oidx);
  if ( ~isempty(subsrefOpts.range) && ~isempty(oidx) )
     switch ( subsrefOpts.range )
      case 'before';  oidx=oidx(end); 
      case 'after';   oidx=oidx(1); 
      case 'between'; oidx=oidx([1 end]);
      case 'outside'; oidx=find(ind); oidx=oidx([1 end]);
    end
 end
 summary=[subsrefOpts.mode ' ']; 
 
 case 'retain';  
  oidx = find(ind);  nidx    = numel(oidx);
  if ( ~isempty(subsrefOpts.range) && ~isempty(oidx) )
     switch ( subsrefOpts.range )
      case 'before';   oidx = oidx(end);
      case 'after';    oidx = oidx(1);
      case 'between';  oidx = oidx([1 end]);
      case 'outside';  oidx = find(~ind); oidx=oidx([1 end]);
    end
 end
 summary = ''; 
end
summary = sprintf('%s%d %ss',summary,nidx,dname);
if ( ~isempty(oidx) ) 
   istr='';
   if ( iscell(odidim(1).vals) && numel(oidx) < 7 ) 
      istr = ['(' sprintf('%s,',odidim(1).vals{oidx(1:end-1)}) odidim(1).vals{oidx(end)} ')']; 
   elseif( numel(oidx)<10 )
      istr = ['(' sprintf('%g,',odidim(1).vals(oidx(1:end-1))) ...
                  sprintf('%g',odidim(1).vals(oidx(end))) ')']; 
   end   
   if ( ~isempty(istr) ) summary = [ summary istr ]; end;
   if ( ~isempty(subsrefOpts.range) ) summary = [ summary ' ' subsrefOpts.range ] ; end;
   if ( ~isempty(odidim(1).units) )   summary = [ summary odidim(1).units ]; end;
end;
if ( ~isempty(opts.summary) )
   summary = [summary  sprintf(' (%s)',opts.summary)]; 
end
odidim(1).vals=odidim(1).vals(~ind); 
if(~isempty(fieldnames(odidim(1).extra))) odidim(1).extra=odidim(1).extra(~ind); end% info for only removed bits
info=struct('idx',Xidx(dim(1)),'Xidx',{Xidx},'ind',ind,'rejectidx',find(~ind),'rejectdi',odidim);
if ( strcmp(opts.testFn,'skip') ) info.testFn=[]; end;
z=jf_addprep(z,mfilename,summary,mergeStruct(opts,subsrefOpts),info);

% special case stuff for Y
if ( isfield(z,'Ydi') ) % use Y's dimInfo to slice it too
   matchdi = n2d(z.Ydi,{odidim.name},1,0); % get matching dims in Ydi
   if ( any(matchdi) ) % something matched!
      %TODO: this should really be using the Xidx info...
      srOpts=subsrefOpts;        srOpts.dim={odidim(matchdi>0).name};
      if ( iscell(srOpts.idx) )  srOpts.idx=srOpts.idx(matchdi>0); end;
      if ( iscell(srOpts.vals) ) srOpts.vals=srOpts.vals(matchdi>0); end;
      [idx,z.Ydi]=subsrefDimInfo(z.Ydi,srOpts);
      z.Y=z.Y(idx{:});
      if ( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) )%update fold info, N.B. have more dims than Y
         if ( ndims(z.foldIdxs)==ndims(z.Y) ) % deal with per-all-sub-problems foldIdxs
            idx{end}=1:size(z.foldIdxs,ndims(z.foldIdxs));
         else % deal with per-sub-prob foldIdxs
            for d=1:ndims(z.Y); if ( size(z.foldIdxs,d)==1 ) idx{d}=1; end; end;
            for d=ndims(z.Y)+1:ndims(z.foldIdxs); idx{d}=1:size(z.foldIdxs,d); end;
         end
         z.foldIdxs=z.foldIdxs(idx{:});
      end
   end
elseif ( strcmp(z.di(dim(1)).name,'epoch') )
   if ( isfield(z,'Y') )
      z.Y=z.Y(ind,:); 
      rmClass = all(z.Y==-1,1);
      z.Y(:,rmClass)=[];
      if ( isfield(z,'keyY') ) z.keyY(rmClass)=[]; end;
      if ( isfield(z,'Ydi') ) 
         z.Ydi(1).vals=z.Ydi(1).vals(ind); z.Ydi(1).extra=z.Ydi(1).extra(ind);
      end
   end;
   if ( isfield(z,'foldIdxs') ) z.foldIdxs=z.foldIdxs(ind,:); end;
end
return;
%------------------------------------------------------------------------
function testCase()
a=jf_reject(z,'dim','ch','range','after','EXG1');

