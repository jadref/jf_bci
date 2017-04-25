function [res]=dgetfield(obj,fieldname,default)
% get field with optional default value if not found
if ( nargin>2 && ~any(strcmp(fieldnames(obj),fieldname) ) ) 
   res=default;
else
   res=getfield(obj,fieldname);
end
