function [z]=jf_addClassInfo(z,varargin)
% add class-label-information to a jf_bci object
%
% Options:
%  Y   - size(z.X,dim) the labels to assign to the data along dimensions dim
%  Ydi -- [dimInfo struct] describing the structure of Yl
%         OR
%         {ndims(Yl) x 1} cell array of dimension names for Yl ({'epoch'})
%  spType -- specificiation of the sub-problem decomposition to use, (see mkspMx)
%            {nSp x 1} cell array of 2x1 cell arrays for each sub-problem, e.g. {sp1 sp2 sp3} 
%                      Each sub-problem cell array, sp1 etc, holds the negative then positive 
%                      class label sets, either as numbers which match the numbers in spKey
%                      or as strings which match the labels in the markerdict.  e.g.
%                  {{'c1' 'c2' 'c3'} {'c6'}}  = class labeled (c1 or c2 or c3) vs. (c6)
%                  {[1 2 4] [6 7]} = epochs marker numbers (1 or 2 or 4) vs. marker numbers (6 or 7)
%          OR    
%             '1v1' - shorthand for one-vs-one decomposition 
%          OR
%             '1vR' - shorthand for one-vs-Rest decomposition
%  spMx   - [nClass x nSp] en/de-coding matrix describing binary sub-problems
%           N.B. automatically generated if you specify spType
%  spD    - [nSp x 1] set of dimensions of input which are used for multi-class decoding ('subProb')
%  mcSp   - [bool] is spMx an encoding of a multi-class problem as a set of    (1)
%           binary sub-problems? (If so then we can can compute multi-class performance results) 
%           compute multiClass performance results?
%  spKey  - [nClass x 1] mapping from spMx rows to marker numbers
%  label  - {nClass x 1 str} mapping from spMx rows to textual marker names
%  markerdict -- [struct] struct containing info mapping from marker numbers to textual names, with fields:
%             marker -- marker ID
%             label  -- [str] human readable name for this marker 
%  zeroLab - [bool] do we treat a value of 0 as a valid label?                 (0)
opts=struct('Y',[],'dim',[],'Ydi',[],'markerdict',[],'summary',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
% setup the labels
dim=opts.dim; Ydi=opts.Ydi;
if ( ~isempty(opts.Ydi) ) % Ydi overrides dim
  if ( iscell(opts.Ydi) ) dim=opts.Ydi; 
  elseif ( isstruct(opts.Ydi) ) 
     dim={Ydi.name}; dim=n2d(z.di,dim,0,0); dim(dim==0)=[]; % match x an y 
  end;
end
if ( isempty(dim) ) if ( n2d(z,'epoch',0,0) ) dim='epoch'; else dim=-1; end; end;
if ( iscell(dim) ) % remove dims' we'll re-generate
  if ( strmatch(dim{end},'lab') ) dim(end)=[]; end; % remove label dim
  dim(strmatch('subProb',dim))=[];  % remove subProb dim
end;
dim=n2d(z.di,dim);
if ( isempty(Ydi) && ~isempty(dim) ) Ydi=z.di(dim); end;

Yl=opts.Y; 
if ( isempty(Yl) && isfield(z.di(dim(1)).extra,'marker'))
   if ( isnumeric(z.di(dim(1)).extra(1).marker) )
      Yl = cat(1,z.di(dim(1)).extra.marker);   
   elseif ( isstruct(z.di(dim(1)).extra(1).marker) ) % event marker
	  Yl={}; numVal=false(numel(z.di(dim(1)).extra),1);
	  for ei=1:numel(z.di(dim(1)).extra); 
		 Yl{ei} = z.di(dim(1)).extra(ei).marker.value; 
		 if ( isnumeric(Yl{ei}) ) numVal(ei)=true; end;
	  end;
	  if ( all(numVal) ) Yl=cell2mat(Yl); end;
	  Yl=Yl(:);
	else
      Yl = {z.di(dim(1)).extra.marker}; Yl=Yl(:);
   end
end
if ( ndims(Yl)==2 && size(Yl,1)==1 && size(Yl,2)>1 ) Yl=Yl'; end; % col vector only

markerdict=opts.markerdict;
if ( isempty(markerdict) && isempty(opts.Y) ) % only extract if using object labels
  if ( isfield(z.di(dim(1)).info,'markerdict') )
    markerdict=z.di(dim(1)).info.markerdict;
  elseif ( isfield(z,'Ydi') && isfield(z.Ydi(n2d(z.Ydi,'subProb')).info,'markerdict') )
    markerdict=z.Ydi(n2d(z.Ydi,'subProb')).info.markerdict;
  elseif ( isfield(z,'Ydi') && isfield(z.Ydi(n2d(z.Ydi,'subProb')).info,'label') )
    markerdict.label = z.Ydi(n2d(z.Ydi,'subProb')).info.label;
    markerdict.marker= z.Ydi(n2d(z.Ydi,'subProb')).info.spKey;
  end
end
if ( ~isfield(z.di(dim(1)).info,'markerdict') )
  z.di(dim(1)).info.markerdict=markerdict;
end

% get the label key and its label
oYdi=[];if( isfield(z,'Ydi') ) oYdi=z.Ydi; end;
[z.Y,z.Ydi,aciOpts]=addClassInfo(Yl,'Ydi',Ydi,'markerdict',markerdict,varargin{:});
aciOpts.Ydi=[]; aciOpts.markerdict=[]; % clear the stuff we fed in...

% TODO: make this work in a more general sense to preserve the old folding..
if ( isfield(z,'foldIdxs') ) % update the folding info if needed
   fIdxs=z.foldIdxs;
   % standardize the shape of fIdxs by inserting a sub-prob dim if not there
   spD=n2d(oYdi,'subProb',0,0); foldD=ndims(fIdxs);
   if(ndims(fIdxs)<=spD) 
      szfIdxs=size(fIdxs); fIdxs=reshape(fIdxs,[szfIdxs(1:spD-1) 1 szfIdxs(spD:end)]);
      foldD=spD+1;
   end

   nFold=size(fIdxs,ndims(fIdxs)); % assume, won't work and just use the number folds
   fdim=n2d(oYdi,{z.di(dim).name},0,0); % current dims which match dim's for folding
   if( any(fdim) ) % simple case, current folding is over epochs, just add extra dims to foldIdxs
      % find new dim locations.
      ndim = n2d(z.Ydi,{oYdi(1:end-1).name},0,0);
      szfIdxs=size(fIdxs);  szfIdxs(end+1:numel(ndim))=1; szY=size(z.Y); szY(end+1:max(max(ndim),numel(z.Ydi)-1))=1;
      % check non-matching entries are unit size.. and all other entries match in size
      if( all(szfIdxs(ndim==0)==1) && all(szfIdxs(ndim>0)==szY(ndim(ndim>0))) ) % OK to re-use
         % make the permutation
         perm(ndim(ndim>0))=1:numel(ndim(ndim>0));
         perm(numel(szY)+1)=foldD; % insert the current fold info
         perm(ndims(szY)+1+(1:sum(ndim==0)))=find(ndim==0); % insert missing singlentons
         perm(perm==0)=foldD+(1:sum(perm==0));
         z.foldIdxs=permute(fIdxs,perm); % N.B. use ipermute as specify where dim goes too
         nFold=[];
      end
   end
   if( ~isempty(nFold) ) 
      warning('New targets invalidated old foldguide, removed!');
      z = rmfield(z,'foldIdxs'); 
   end
end

% record what we've done
summary=sprintf('to %s',[z.di(dim(1)).name sprintf(',%s',z.di(dim(2:end)).name)]);
if ( ~isempty(opts.summary) ) summary=[summary '(' opts.summary ')']; end;
z = jf_addprep(z,mfilename,summary,{opts aciOpts},[]);
return;

%------------------------------------------------------------------------
function testCase()
z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_rc_sep');
z=jf_addClassInfo(z,'Y',{z.di(n2d(z.di,'letter')).extra.target},'dim','letter')

z=jf_addClassInfo(z,'Y',{z.di(n2d(z.di,'letter')).extra.target},'spKey',z.di(3).info.lettergrid,'dim','letter')
