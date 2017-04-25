function [ostr]=varexpand(str,dict,strbgn,strend)
% expand variables in a string
%
% [str]=varexpand(str,dict[,strbgn,strend])
%
% Inputs:
%  str    -- [str] string to expand
%  dict   -- [struct] dictionary of variable names and values
%  strbgn -- [str] string denoting the start of var name to expand ('++')
%  strend -- [str] string denoting the end of var name expansion ('--')
if ( nargin < 3 || isempty(strbgn) ) strbgn='++'; end;
if ( nargin < 4 || isempty(strend) ) strend='--'; end;
bgns = strfind(str,strbgn); % all possible start positions
ends = strfind(str,strend); % all possible end positions
if ( isequal(strbgn,strend) ) bgns=bgns(1:2:end); ends=ends(2:2:end); end; % deal with equal-start/end
bgns = [-1 bgns]; 
ends = [1-numel(strend) ends]; 
ostr = '';
vnames = fieldnames(dict); 
for ei=2:numel(bgns); % loop over the possible epxansion locations
   ostr = [ostr str(ends(ei-1)+numel(strend):bgns(ei)-1)];
   vname=str(bgns(ei)+numel(strbgn):ends(ei)-1); % var name
   if ( isfield(dict,vname) )
      vali=getfield(dict,vname);
      if ( isnumeric(vali) ) ostr=[ostr sprintf('%g',vali)];
      elseif( ischar(vali) ) ostr=[ostr vali];
      else warning('Can only expand strings or numerics, sorry'); 
      end
   else
      warning('no var named : %s',vname); continue; 
   end;
end
ostr=[ostr str(ends(ei)+numel(strend):end)];
return;
%----------------------------------------------------------
function testCase()
str='++var1--==++var2--';
dict=struct('var1','one','var2',1);
varexpand(str,dict)


fid=fopen('mastermfile.template');str=fscanf(fid,'%c');fclose(fid);
dict=struct('statusfile','sfile','jobname','jname','description','desc','matlabpath','mpath','cwd','.',...
            'matlab_preflight','','globalfile','globfile','callstr','sub stuff');
varexpand(str,dict)