function [Y,Ydi,opts]=addClassInfo(Yl,varargin)
% add class-label-information to a dimInfo thing
%
%  [Y,Ydi]=addClassInfo(Yl,varargin)
%
% Inputs:
%  Yl -- [n-d] set of class labels
% Options:
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
%  spKey  -- mapping from spMx rows to marker numbers
%            [nClass x 1] list of unique marker numbers
% % %            OR  % N.B. not implemented yet!
% % %            {Nclass x 1} sets of marker numbers to treat as one class
%  mcSp   - [bool] is spMx an encoding of a multi-class problem as a set of    (1)
%           binary sub-problems? (If so then we can can compute multi-class performance results) 
%           compute multiClass performance results?
%  label  -- mapping from marker numbers to textual names in same order as spKey
%  spMx   -- [nClass x nSp] en/de-coding matrix describing the binary sub-problems
%  markerdict -- [struct] struct containing info mapping from marker numbers to textual names, with fields:
%             marker -- marker ID
%             label  -- [str] human readable name for this marker 

opts=struct('Ydi',{{'epoch'}},'spType','1vR','mcSp',[],'spMx',[],'markerdict',[],'spKey',[],'label',[],'zeroLab',0,'compBinp',1,'spD',{{'subProb'}});
opts=parseOpts(opts,varargin);

if( ndims(Yl)==2 && size(Yl,1)==1 && size(Yl,2)>1 ) Yl=Yl'; end; % col vector only
szYl=size(Yl); if(szYl(end)==1) szYl(end)=[]; end; % get right size
% setup the labels
Ydi=opts.Ydi;
if ( ischar(Ydi) ) Ydi=mkDimInfo(szYl,Ydi); 
elseif(iscell(Ydi) ) 
  tmp=Ydi;
  Ydi=mkDimInfo(szYl);for id=1:numel(tmp); Ydi(id).name=tmp{id};end;
  if (isempty(strmatch('subProb',tmp)) && numel(tmp)<numel(szYl)) % last dim is subProb dim
    Ydi(end-1).name='subProb';
  end
end
if ( isempty(Yl) && ~isempty(Ydi) && isfield(Ydi(1).extra,'marker')) 
   if ( isnumeric(Ydi(1).extra(1).marker) )
      Yl = cat(1,Ydi(1).extra.marker);   
   else
      Yl = {Ydi(1).extra.marker}; Yl=Yl(:);
   end
end

% get the label key and its label
markerdict=opts.markerdict;
if ( isempty(markerdict) && isfield(Ydi(1).info,'markerdict') )
   markerdict=Ydi(1).info.markerdict;
end
if ( ~isempty(markerdict) ) 
   info.spKey = markerdict.marker; info.label = markerdict.label; 
else
   info.spKey = opts.spKey;
   info.label = opts.label;
   if( isempty(info.spKey) )  
     info.spKey=unique(Yl); info.spKey=info.spKey(:)'; info.label=[]; 
     if ( numel(info.spKey)>20 ) 
        warning('%d labels.... treating as continuous values',numel(info.spKey));
        info.spKey=[];
     end
     if ( isequal(info.spKey,[-1 1]) || ... % fix binary special cases to not change order/values
          (~opts.zeroLab && isequal(info.spKey,[-1 0 1])) ||...
          (opts.zeroLab && isequal(info.spKey,[0 1])) ) 
       info.spKey=info.spKey(end:-1:1); 
     end
   end
   if( isempty(info.label) )
      if( ischar(info.spKey) ) % set the label
         info.label={};for i=1:numel(info.label); info.label{i}=info.spKey(i); end; 
      elseif ( iscell(info.spKey) && ischar(info.spKey{1}) )
         info.label=info.spKey;
      elseif ( isnumeric(info.spKey) )
         if ( ~opts.zeroLab ) info.spKey(info.spKey==0)=[]; end; % check the zero-lab thing
         for i=1:numel(info.spKey); info.label{i}=sprintf('%d',info.spKey(i)); end;
      else % don't know what else to do!
         for i=1:numel(info.spKey); info.label{i}=sprintf('%d',i); end;   
      end
   end
end
% get the encoding/decoding matrix
info.spD=opts.spD; if ( isempty(info.spD) ) info.spD='subProb'; end;
info.spMx=opts.spMx;  info.spType=opts.spType;
if ( isempty(info.spMx) && ~isempty(info.spKey)  ) 
   info.spMx  =mkspMx(info.spKey,info.spType,opts.compBinp,info.label);
end
if ( isempty(info.spKey) ) % mark-up as regression problem
   info.spDesc='regression';
else
   info.spDesc=mkspDesc(info.spMx,info.label);
end
info.mcSp=opts.mcSp;
if ( isempty(info.mcSp) ) % is this a multi-class setup?
  if ( ischar(info.spType) && any(strcmp(info.spType,{'1v1','1vR'})) ) % 1v1/1vR is auto mcSp
    info.mcSp=1;
  else
    info.mcSp=0;
  end
end

% convert to set bin-subprobs +1/-1 (if needed)
if(~isempty(info.spKey) && ...
   ( (numel(Yl)==size(Yl,1) && isnumeric(Yl) && ~all(Yl(:)==-1 | Yl(:)==0 | Yl(:)==1)) || ...
     (opts.zeroLab && isnumeric(Yl) && all(Yl(:)==0 | Yl(:)==1)) || ...
     islogical(Yl) || iscell(Yl) ) ) 
   szYl = size(Yl); szYl(find(szYl==1,1,'last'):end)=[];
   Y=lab2ind(Yl(:),info.spKey,info.spMx,opts.zeroLab,opts.compBinp);  
   Y=reshape(Y,[szYl size(Y,2)]);
else
  Y=Yl;
end
% setup the Ydi
szY=size(Y); if(numel(szY)==2 && szY(2)==1) szY(2)=[]; end; 
% extra dim for bin subProb
if( size(info.spMx,1)==1 || isempty(info.spKey) || (numel(info.spD)>ndims(info.spMx)-1) ) szY(end+1)=1; end; 
if ( numel(Ydi)==1 || (numel(Ydi)<numel(szY)) || (numel(Ydi)==numel(szY) && strcmp(Ydi(end).name,'subProb')) ) 
   Ydi(end+1)=Ydi(end); % add extra value dim if needed
end
Ydi(end)=mkDimInfo([],1,'lab',[],[]);
if ( n2d(Ydi,'subProb',0,0)==0 ) % add extra subProb dim if needed
  Ydi = Ydi([1:end end]); 
  Ydi(end-1) = mkDimInfo(szY(end),1,'subProb'); % last Y dim
end; 
if ( numel(info.spDesc)==szY(n2d(Ydi,info.spD))) 
  Ydi(n2d(Ydi,info.spD(1))).vals=info.spDesc; 
else
  info.mcSp=0; % mark as non-multi-class
end;
if( numel(info.spD)>1 ) Ydi(n2d(Ydi,info.spD(2))).vals=info.spDesc(1:size(Y,n2d(Ydi,info.spD{2}))); end;
if ( ~isempty(Y) && numel(Y)==max(size(Y)) ) % vector labels
   if ( numel(Ydi(1).extra)==0 ) Ydi(1).extra=repmat(struct('marker',[]),size(Y,ndims(Y)),1); end; 
   [Ydi(1).extra.marker]=num2csl(Y,ndims(Y)); % override epoch marker info with true info
end
Ydi(end-1).info      =info;         % decoding info in the subProb dimension
return;

%------------------------------------------------------------------------
function testCase()
L=3;
Y=round(rand(100,1)*L); % L-classes
[Y2,Ydi]=addClassInfo(Y,'Ydi','epoch')
[Y2,Ydi]=addClassInfo(Y,'Ydi','epoch','spType',{{1 2} {2 3}})

z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_rc_sep');
[Y,Ydi]=addClassInfo({z.di(n2d(z.di,'letter')).extra.target},'Ydi',z.di(n2d(z.di,'letter')))

[Y,Ydi]=jf_addClassInfo({z.di(n2d(z.di,'letter')).extra.target},'spKey',z.di(3).info.lettergrid)

% try with 2-d sub-prob
L=2;
Y=round(rand(100,1)*L); % L-classes
[Y,Ydi]=addClassInfo(repmat(shiftdim(Y,-1),5),'Ydi',{'time' 'epoch'},'spMx',repmat([1 -1],[5,1]),'spD',{'time','subProb'},'mcSp',1);
