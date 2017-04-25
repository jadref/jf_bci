function [x2y,y2x]=matchstrs(x,y,icase,prefix)
% compute a mapping between 2 sets of strings
%
% [x2y,y2x]=matchstrs(x,y,icase,prefix)
%
% Inputs:
%  x - 1st set of strings to match, 
%      do a regexp match if x starts and ends with either # or %
%  y - 2nd set of strings to match
%  icase - [bool] case invarient (0)
%  prefix - [bool] match only prefix (0)
% Outputs:
%  x2y - mapping from x -> y
%  y2x - mapping form y -> x
if ( nargin<3 || isempty(icase) ) icase=false; end;
if ( nargin<4 || isempty(prefix) ) prefix=false; end;
if ( ~iscell(x) ) x={x}; end;
if ( ~iscell(y) ) y={y}; end;
if ( icase ) x=lower(x); y=lower(y); end;
x2y=zeros(numel(x),1); y2x=zeros(numel(y),1);
for i=1:numel(x);
   if ( prefix ) 
      mi=strmatch(x{i},y);
   else
      mi=strmatch(x{i},y,'exact'); 
   end
   if ( isempty(mi) && ( x{i}(1)=='#' & x{i}(end)=='#' || x{i}(1)=='%' && x{i}(end)=='%' ) ) % regexp match?
     for j=1:numel(y);
       tmp=regexp(y{j},x{i}(2:end-1));
       if ( ~isempty(tmp) && tmp==1 ) mi=[mi j]; end
     end
   end 
   if(~isempty(mi)) x2y(i,1:numel(mi))=mi; y2x(mi)=i; end;
end;
return;
%-------------------------------------------------------
function testCase();
s1={'hello' 'there' 'stupid' 'here'}
s2={'here' 'stupid' 'there' 'hello'}
[x2y,y2x]=matchstrs(s1,s2);
% regexp match
[x2y,y2x]=matchstrs('#.*er.*#',s2)