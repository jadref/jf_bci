function [z]=jf_mkSubProb(z,varargin);
opts=struct('dim','epoch','idx',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

idx=opts.idx;
if ( ischar(idx) )
   if ( strcmp(idx,'block') ) 
      idx=getBlockIdx(z,idx);
   else
      error('unrecog sub-prob index spec');
   end
end

spId = unique(idx);
nSp  = numel(spId);

spD=ndims(z.Y); if ( isfield(z,'Ydi') ) spD=n2d(z.Ydi,'subProb'); end;
if ( size(z.Y,spD)>1 ) warning('adding sub-problems to an existing multi-sub-problem situation'); end;
dim=spD-1;      if ( isfield(z,'Ydi') ) dim=n2d(z.Ydi,opts.dim); end;

szY=size(z.Y);
oY =z.Y;
z.Y=zeros([szY(1:spD-1) szY(spD)*nSp szY(spD+1:end)]); % make bigger one
idxY={}; idxoY={}; for di=1:numel(szY); idxY{di}=1:size(z.Y,di); idxoY{di}=1:size(oY,di); end;
for bi=1:nSp; % insert in place
   bidx = idx==spId(bi);
   idxY{spD} =bi;
   idxY{dim} =bidx; idxoY{dim}=bidx; % which elements to use for this block
   z.Y(idxY{:})=oY(idxoY{:}); % copy into new Y
end

% update meta-info
if ( isfield(z,'Ydi') ) % update spD
   z.Ydi(spD).vals = repmat(z.Ydi(spD).vals,1,nSp);
   for bi=1:nSp; if ( iscell(z.Ydi(spD).vals) ) z.Ydi(spD).vals{bi} = sprintf('%s_b%d',z.Ydi(spD).vals{bi},bi); end; end;
   % and spMx
   z.Ydi(spD).info.label = repmat(z.Ydi(spD).info.label,[1,nSp]);
   z.Ydi(spD).info.spMx  = repmat(z.Ydi(spD).info.spMx,[nSp,1]);
end
summary = sprintf('to %s',z.di(n2d(z,opts.dim)).name);
info=struct('idx',idx);
z=jf_addprep(z,mfilename,summary,opts,info);
return;
