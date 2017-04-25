function [ures]=uniformizeCellArray(res,squeezep)
% Convert a cell array of matrices to a single matrix -- if possible.
% if squeeze > 1 then also compress out redundant singlenton dimensions
if ( nargin < 2 ) squeezep=0; end;
if ( iscell(res) && iscell(res{1}) || isempty(res{1}) )
   warning('Cant (yet) uniformize cellarray of cellarrays');
   ures=[]; return;
end

nres=size(res);  
rsz=size(res{1}); 
rcls=class(res{1});

% % array to hold the result
ures=repmat(res{1}(1),[prod(nres) prod(rsz)]);
for objId=1:numel(res);
   if ( ~isequal(class(res{objId}),rcls) || ... % check its possible
        ~isequal(size(res{objId}),rsz) )
      warning('Unequal classes or results sizes -- cannot uniformize');
      ures=[];
      return;
   else
      ures(objId,:)=res{objId}(:);
   end      
end

% remove un-necessary singlenton dimensions?
if(prod(rsz)<=1) rsz=1; 
elseif(squeezep>0) rsz=rsz(rsz>1);   % remove all singlentons
else rsz=rsz(1:find(rsz>1,1,'last'));% remove only trailing singlentons
end;
if(prod(nres)<=1) nres=1; 
elseif( squeezep>0) nres=nres(nres>1);  % remove all singlentons
else nres=nres(1:find(nres>1,1,'last'));%remove only trailing singlentons
end   
ures=reshape(ures,[nres,rsz]); % shape to output size
