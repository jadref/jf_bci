function [res]=split(c,str) 
% turn string into sub-bits
%
% res=split(c,str)
%
% c - {char} cell array of characters to split with
% str -[char] string to split
tmpstr=str;
if (isempty(c))  c = num2cell(char([9:13 32])); end
if ( ~iscell(c) ) c={c}; end; 
di=[0 numel(str)+1];
for i=1:numel(c);
   starts=find(str(1:end-numel(c{i})+1)==c{i}(1));
   for j=2:numel(c{i}); % match the rest of the string
      starts(str(starts+1)~=c{i}(j))=[];
   end
   di=[di,starts];                % match start
   de=[di,starts+numel(c{i})-1];  % match end
end
[di,si]=sort(di);de=de(si); % sort into ascending order
if ( de(end)>numel(str)+1 ) de(end)=[]; di(end)=[]; end; % strip empty last split
for i=1:numel(di)-1;        % split them out
   res{i}=tmpstr(de(i)+1:di(i+1)-1);
end
