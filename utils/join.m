function [res]=join(c,varargin)
if ( numel(varargin)==1 && iscell(varargin{1}) ) strs=varargin{1}; 
else  strs=varargin;
end
if ( isempty(strs) ) res=char(); return; end;
res=repmat(' ',1,sum(cellfun('length',strs))+(numel(strs)-1)*numel(c));
lst=0;
for i=1:numel(strs)-1; 
   if ( isempty(strs{i}) ) continue; end; 
   str=lst+1; lst=str+numel(strs{i})-1;
   res(str:lst)=strs{i}; 
   str=lst+1; lst=str+numel(c)-1;
   res(str:lst)=c;
end
str=lst+1; lst=str+numel(strs{end})-1;
res(str:lst)=strs{end}; 
res(lst+1:end)=[]; % delete anything which is left over