function [A]=mergeStruct(A,varargin);
% recursively merge B into A, optionally overwriting common fields with B's
% info, optionally B can be a cell array of name,val pairs.
override=1;
if ( isnumeric(varargin{end}) ) override=varargin{end}; varargin{end}=[];end;
for i=1:numel(varargin); % merge in in order
   B=varargin{i};
   if ( isstruct(B) ) fn=fieldnames(B);fv=struct2cell(B);
   elseif ( iscell(B) ) fn=B(1:2:end); fv=B(2:2:end); 
   end
   for fi=1:numel(fn);
      if ( ~isfield(A,fn{fi}) ) % assign this field in
         A.(fn{fi}) = fv{fi};
      elseif ( override )       % assign in new info if wanted..
         if ( isstruct(A.(fn{fi})) && isstruct(fv{fi}) )
            % both structures so recursively set sub-fields
            A.(fn{fi}) = mergeStruct(A.(fn{fi}),fv{fi},override);
         else % one or other isn't a struct so just override
            A.(fn{fi}) = fv{fi};
         end
      end
   end
end
return;
%------------------------------------------------------------------------
function []=testCase();
t1=struct('this','is','a','test');
t2=struct('and','so','is','this');
t3=struct('this','was','the','final','straw','argh');
mergeStruct(t1,t2)
mergeStruct(t1,t2,t3)