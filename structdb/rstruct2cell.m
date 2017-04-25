function [vals]=rstruct2cell(obj)
% recrusive  extension of matlabs struct to cell.
%
% This version reports all field values in a *flattened* fashion such that the
% values corrospend to those provided by rfieldnames, e.g.
%  'field1'
%  'field2.field21'
%  'field2.fidle22'
% etc.
% SEE ALSO: rfieldnames
%
% Inputs:
%  obj  -- struct to get the values of
% Outputs:
%  vals -- { nFieldNames(obj) x size(obj) } cell array of field values
% $Id: rstruct2cell.m,v 1.6 2006-11-15 21:36:04 jdrf Exp $
if( isempty(obj) || isequal(obj(1),struct()) ) % empty struct is a special case
   vals = cell([1,size(obj)]);
elseif ( numel(obj) > 1 ) % are we a struct array? -- N.B. must be identical form
   %vals = cell(1,numel(obj));
   % N.B. this reshape will fail if the structs aren't *IDENTICAL* in
   % sub-structures also -- and it adds an extra dimension because the 
   % sub-structs could be arrays also!
   vals = innerrstruct2cell(obj(1));
   for i=2:numel(obj);
      vals(:,i)=innerrstruct2cell(obj(i)); % [nsubfields x numel(obj)]
   end
   vals=reshape(vals,[size(vals,1) size(obj)]);
else
   vals=innerrstruct2cell(obj);
end



function [vals]=innerrstruct2cell(obj)
if ( isempty(obj) ) vals=cell([numel(fieldnames(obj)),0]); return; end;
fn=fieldnames(obj);fv=struct2cell(obj);
if ( isempty(fn) ) vals=struct(); return; end;
vals=cell(0);%numel(fn),1);
for i=1:numel(fn);
   if ( isstruct(fv{i}) )
      tres=rstruct2cell(fv{i}); % recursively eval
      tsize=size(tres); % [ nFields x size(fv{i}) ]
      if ( prod(tsize(2:end)) > 1 ) % deal with array inputs..
         for j=1:size(tres,1); 
            % Insert sub-fields as new fields in the output
            vals{end+1,1}=reshape({tres{j,:}},[tsize(2:end) 1]); 
         end;
      else % non-array so treat normally.
         if ( isempty(tres) ) % struct with no fields!
            vals{end+1,1}=struct();
         else
            for j=1:size(tres,1);  vals{end+1,1}=tres{j}; end;
         end
      end
   else
      vals{end+1,1}=fv{i};
   end
end

%--------------------------------------------------------------------------
function testCase()

