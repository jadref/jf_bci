function [idx]=sscanto(str,flag,past)
% first scan until we see flag or eof
cps=[0:1000:length(str) length(str)];
for i=1:length(cps)-1;
  cp=strIdx(str(cps(i)+1:cps(i+1)),flag);
  if ( cp ) break; end;
end
if ( ~isempty(cp) ) 
  if ( nargin > 2 ) cp=cp+length(flag);end;
  idx=cps(i)+cp-1;
else 
  idx=-1;
end
