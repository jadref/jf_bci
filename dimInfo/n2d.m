function [d]=n2d(di,nm,exactmatchp,mustmatchp,mmatch)
% convert name/id into correct dim index
% Inputs:
%  di -- dimInfo Struct
%      OR
%        {str} cell array of dimension names to match on
%  nm -- str OR cell array of dimension names to match
%  exactmatchp -- [bool] flag must match exactly, or only prefix (0)
%  mustmatchp  -- [bool] flag error if doesn't match (1)
%  mmatch      -- [str] if multiple dims match which one to return ('first')
%                   'first' or 'last'
%                 [int] which match to return.  1='first', 2=second, -1='last', -2=secondlast
if ( nargin < 3 || isempty(exactmatchp) ) exactmatchp=0; end;
if ( nargin < 4 || isempty(mustmatchp) ) mustmatchp=1; end;
if ( nargin < 5 || isempty(mmatch) ) mmatch='first'; end;
if ( numel(di)==1 && isfield(di,'di') ) di=di.di; end;
if ( ischar(di) ) di={di}; end;
if ( ischar(nm) ) nm={nm}; end;
if ( ischar(di) ) di={di}; end;
if ( iscell(di) && ischar(di{1}) ) strs=di; 
elseif ( isstruct(di) && isfield(di,'name') ) 
  if ( isfield(di,'vals') && numel(di(end).vals)==1 && isempty(di(end).info) ) % ignore last entry if it's just the name/units of the elements 
    di=di(1:end-1); 
  end;
  strs={di.name}; 
else 
   strs={};
end;
d=zeros(numel(nm),1);
if ( iscell(nm) ) % convert names to dims
   for i=1:numel(nm)  
      if ( isnumeric(nm{i}) ) d(i)=nm{i}; 
      elseif ( ischar(nm{i}) && ~isempty(nm{i}) )
         if ( exactmatchp>0 ) t=strmatch(nm{i},strs,'exact');
         else                 t=strmatch(nm{i},strs);
         end
         if(~isempty(t)) 
            if ( numel(t)>1 && ~exactmatchp ) 
               tt=strmatch(nm{i},strs,'exact'); if( ~isempty(tt) ) t=tt; end;
            end;
            if ( numel(t)>1 ) 
              if ( ischar(mmatch) ) 
                if ( strcmpi(mmatch,'last') ) 
                  t=t(end); 
                elseif ( strcmpi(mmatch,'first') ) 
                  t=t(1); 
                %elseif ( ~strcmpi(mmatch,'all') )
                %  ;
                else
                  error('Unrec multi-match type');
                end;
              elseif ( isnumeric(mmatch) && mmatch~=0 ) 
                if( mmatch<0 ) mmatch = numel(t)+1+mmatch; end; 
                t=t(min(end,mmatch)); 
              else
                warning('%s matched %d times. Returning all entries',nm{i},numel(t)); 
              end
            end;
            d(i,1:numel(t))=t; 
         elseif( mustmatchp) error('Couldnt find dim: %s',nm{i});
         end
      else
      end
   end;
elseif ( isnumeric(nm) )
   d=nm;
end
ndi=numel(di); if ( isstruct(di) && isfield(di,'name') && isempty(di(ndi).name) ) ndi=ndi-1; end;
d(d<0)=d(d<0)+ndi+1; 
return;
