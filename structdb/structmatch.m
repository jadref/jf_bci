function [matchp]=structmatch(X,varargin)
% Function to test if one struct matches the fields given by another
% TODO: expand the types of test to include comparasions etc?
% Usage:
%    structmatch(struct('hello','there'),'hello','me')
% N.B. * as a value is a wild-card match anything
%      and if the value matched is also a string the we use a regexp match
%
%  matchp = structmatch(X,name1,val1,name2,val2,...)
%
% Inputs:
%  X -- [struct n-d] structure to match
%  name1,value1 -- name,value pairs to match, 
%       name1 - can be 'n1.n2...' to match in sub-structs
%       ~name1 - to require value to *not* match
%       value1 - can be {v1 v2} to match set of values (N.B. use {{v1 v2}} to match a cell value)
%                * to match any value, i.e field must just exist
%                %regexp% or #regexp# - to regexp match a string
% Outputs:
%  matchp -- [bool n-d] indicator of which elements of X matched
%
% N.B. for strings if the 1st and last value are # or % then we treat the bit between as a 
%      regular experssion to match with.
if ( isempty(X) ) matchp=false(size(X)); return; end;
if ( ~(isstruct(X) || (iscell(X) && isstruct(X{1}))) ) error('X must be a struct'); end
if ( numel(varargin)==1 && isstruct(varargin{1}) ) 
   fn=rfieldnames(varargin{1}); fv=rstruct2cell(varargin{1});
elseif ( iscell(varargin) && numel(varargin)>1 ) 
   fn=varargin(1:2:end); fv=varargin(2:2:end); 
else
   error('match pattern must be struct or cell array of name/value pairs');
end;
matchp=true(size(X));
for ifn=1:numel(fn);
   shouldmatch=true;
   if ( isequal(fn{ifn}(1),'~') ) % negate the sense of the match
      fn{ifn}=fn{ifn}(2:end); shouldmatch=false; 
   end
   fni=fn{ifn}; fvi=fv{ifn};
   for iel=1:numel(X); % loop over X
      matchpiel=matchp(iel);
      if ( ~matchpiel ) continue; end; % only if not already discounted
      if ( isstruct(X) )                     [val,found]=rgetfield(X(iel),fni,[]);
      elseif ( iscell(X) && isstruct(X{iel}) ) [val,found]=rgetfield(X{iel},fni,[]);
      else found=false;
      end
      if ( ~found ) matchpiel=false; 
      elseif ( isequal(fvi,'*') ) % wild-card match
      elseif ( iscell(fvi) )      % option list match
         submatchp=false;
         for ival=1:numel(fvi); 
            if ( isequal(val,fvi{ival}) ) submatchp=true; break; end;
         end
         if ( ~submatchp ) matchpiel=false; end;
      elseif( isa(fvi,'function_handle') ) % function handle to test match
         matchpiel=feval(fvi,val);
      elseif( ischar(fvi) )
         % do regexp match if not simple string match
         if ( isempty(strmatch(val,fvi,'exact')) )
            if ( fvi(1)=='#' & fvi(end)=='#' || fvi(1)=='%' && fvi(end)=='%' )
               mi = regexp(val,fvi(2:end-1));
               matchpiel=~isempty(mi) && (mi==1);
            else
               matchpiel=false;
            end
         end
      elseif( ~isequal(val,fvi) ) matchpiel=false; 
      end;
      % test if the match is what we wanted
      if ( xor(shouldmatch,matchpiel) ) matchpiel=false; else matchpiel=true; end;
      matchp(iel)=matchp(iel) & matchpiel;   % store the result
   end
end;
return;

%----------------------------------------------------------------------------
function []=testCases()
A=struct('hello',{'there','is','here','test'},'you',{'fool','if','you','think'},'ss',struct('sub','val'));
structmatch(A,'hello','#.*her.*#')
structmatch(A,'ss.sub','val')