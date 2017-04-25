function [found]=risfield(obj,fieldname)
% recursive evalutation extension of isfield
%   risfield(obj,'hello.there');
% i.e. index into sub-structures to get the field
% N.B. assumes struct arrays are *IDENTICAL* in every sub-level!
if ( isstr(fieldname) ) fieldname=split(fieldname,'.'); end
found=true;
for fi=1:numel(fieldname);
   if ( isfield(obj,fieldname{fi}) ) 
      obj=obj(1).(fieldname{fi});  % get this sub-field
   else
      found=false; break; 
   end;
end

function [res]=split(str,c) % used to turn string into sub-bits
tmpstr=str;
di=[0 find(str==c) numel(str)+1];
for i=1:numel(di)-1;
   res{i}=tmpstr(di(i)+1:di(i+1)-1);
end
