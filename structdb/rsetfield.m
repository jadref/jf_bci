function [obj]=rsetfield(obj,fieldname,val)
% recursive version of set field
di=[0 find(fieldname=='.') numel(fieldname)+1];
% first find how deep we allready exist.
tobj=obj;
for prei=1:numel(di)-1;
   subfield=fieldname(di(prei)+1:di(prei+1)-1);
   if ( isfield(obj,subfield) )
      tobj=tobj.(subfield);
   else
      break;
   end
end
prefix=fieldname(1:di(prei+1)-1); % can assign 1 past last valid entry
% build the rest of the chain
for i=numel(di)-1:-1:prei+1;
   subfield=fieldname(di(i)+1:di(i+1)-1);
   val.(subfield)=val;
end
eval(['obj.' prefix '=val;']); % eval to perform the assignment

