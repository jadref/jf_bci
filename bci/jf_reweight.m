function [z,opts]=jf_reweight(z,varargin)
% re-reference with a weighted mean over a subset of indicies
% Options:
%  dim -- dimension(s) to re-weight along ('time')
%  wght-- [size(X,dim),1] weighting for the included elements. ([])
%         OR
%          [2/3/4x1] spec of which points to use for the base-line computation in format 
%                    as used for mkFilter, 
%                    e.g. [-100 -400] means use a smooth *gaussian* window from -100ms to -400ms
%  op  -- [str] operator to us for the reweighting ('*')
%  summary -- additional summary string
opts=struct('dim','time','wght',[],'op','*','summary','','subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

dim=n2d(z,opts.dim);

wght = opts.wght;
if ( ~isnumeric(wght) || numel(wght)~=size(X,dim) ) 
  wght=mkFilter(size(z.X,dim),wght,z.di(dim).vals);
  wght=wght./sum(wght); % make it compute an average
end
% shift to the right dimension
if ( numel(wght)==max(size(wght)) ) wght=shiftdim(wght(:),-(dim-1)); end;

% do the re-ref, on the selected elements
%if ( ~isempty(subsrefOpts.idx) || ~isempty(subsrefOpts.vals) )
%   z.X(idx{:})=repop(z.X,opts.op,wght);
%else
   z.X        =repop(z.X,opts.op,wght);
%end

% update the meta-info
summary = sprintf('over %s',z.di(dim(1)).name);
if( numel(dim)>1 ) summary = [summary sprintf('+%s',z.di(dim(2:end)).name)]; end;
if(ischar(opts.wght)) summary = [summary ' with ' opts.wght]; 
elseif( isnumeric(wght) ) summary=['weighted ' summary];
end;
if(~isempty(opts.summary)) summary=[summary  sprintf(' (%s)',opts.summary)];end
info.wght=wght;
z=jf_addprep(z,mfilename,summary,opts,info);
return;


