function [onset,dur,Desc,type,typeDesc,typeIdx]=parseAnnotation(str)
% Parse an EDF+ Annotations channel and return event information
%
% N.B. this code lifted from Biosig sopen
if ( ~isstr(str) ) str=char(str(:)'); end;
N = 0; 
onset = []; dur=[]; Desc = {};
[s,t] = strtok(str,0);
while ~isempty(s)
  ix = find(s==20);
  if ( ~isempty(ix) )
    N  = N + 1; 
    [s1,s2] = strtok(s(1:ix(1)-1),21);
    tmp = str2double(s1);
    onset(N,1) = tmp;
    tmp = str2double(s2(2:end));
    if  ~isempty(tmp)
      dur(N,1) = tmp; 	
    else 
      dur(N,1) = 0; 	
    end;
    Desc{N} = char(s(ix(1)+1:end-1));
  end
  [s,t] = strtok(t,0);
  type(N,1) = length(Desc{N});
end;		
[typeDesc, typeIdx, type] = unique(Desc(1:N)');

[onset,dur,Desc,type,typeDesc,typeIdx]
return;
