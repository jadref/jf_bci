function [matchi,substrs]=regexpstrmatch(strs,vals,substrregexp)
% match 2 sets of strings or a sub-string defined by a regexp
%
% [matchi,substrs]=regexpstrmatch(strs,vals,substrregexp)
%
if ( nargin < 3 ) substrregexp=''; end;
if ( iscell(vals) && numel(vals)>0 && isnumeric(vals{1}) ) vals=cat(1,vals{:}); end;
if ( ischar(vals) ) vals={vals}; end;
if ( ischar(strs) ) strs={strs}; end;
if ( (isempty(vals) || iscell(vals) && numel(vals)==1 && isempty(vals{1})) && isempty(substrregexp) ) 
   matchi=true(numel(strs),1); 
   substrs={};
   return; % no val matches everything
end; 
matchi=zeros(numel(strs),1);
substrs={};
for si=1:numel(strs);
   if ( isempty(vals) ) % just a regexp match needed
      if( ~isempty(regexp(strs{si},substrregexp)) ) matchi(si)=1; end;
   else
	  substr=strs{si}; % start assuming should match the whole string
	  if ( ~isempty(substrregexp) ) % use regex to extract the sub-string to match on
       substr=regexp(strs{si},substrregexp,'tokens');
       if( isempty(substr) || numel(substr)>1 ) continue; end;
       substr=substr{1}{1};
	  end
     substrs{si}=substr;
     if( isnumeric(vals) )                     mi = str2num(substr)==vals;
     elseif( iscell(vals) && ischar(vals{1}) ) mi = strcmp(substr,vals);
     end;
     if ( any(mi) ) matchi(si)=find(mi,1); end;
   end
end
