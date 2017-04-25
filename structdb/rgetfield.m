function [res,found]=rgetfield(obj,fieldname,default,usedefault)
% recursively getfield, i.e. index into sub-structures to get the field
% if not found defaults to returning [], or given default value.  Use 4th
% arg to override.
% e.g.   obj=struct('hello',struct('there','me'));
%        res=rgetfield(obj,'hello.there')
%        res=rgetfield(obj,{'hello','there'},0);

if ( nargin < 3 ) default=[]; end;
if ( nargin < 4 ) usedefault=true; end;

if ( isobject(obj) ) obj=struct(obj); end;

if ( numel(obj)>1 ) % deal with matrix inputs
  for oi=1:numel(obj);
    [res{oi},found(oi)]=rgetfield(obj(oi),fieldname,default,usedefault);
  end
  res=reshape(res,size(obj)); found=reshape(found,size(obj));
  return;
end


% convert field names to cell array
if ( isstr(fieldname) ) fieldname=split(fieldname,'.'); end
subobj=obj;  % recurse into the object to find the bit we need
found=true;
for fi=1:numel(fieldname);
   if ( isfield(subobj,fieldname{fi}) ) 
      subobj=cat(1,subobj.(fieldname{fi}));  % get this sub-field
   else
      found=false; break; 
   end;
end
if ( found )
   res=subobj;
else
   if ( usedefault ) res=default; 
   else error('Fieldname not found!'); 
   end;
end
   
function [res]=split(str,c)
tmpstr=str;
di=[0 find(str==c) numel(str)+1];
for i=1:numel(di)-1;
   res{i}=tmpstr(di(i)+1:di(i+1)-1);
end
