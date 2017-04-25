function [str]=tabDisp(sarray,fields,rotate)
% make a tabular display of the input structure array
%
% [str]=tabDisp(sarray,fields,rotate)
%
% Inputs:
%  sarray - structure array to display
%  fields - set of fieldnames to display
%  rotate - put fields in rows?
if ( nargin < 2 ) fields=[]; end;
if ( nargin < 3 || isempty(rotate) ) rotate=true; end;
fn=rfieldnames(sarray);
fv=rstruct2cell(sarray);
if ( ~isempty(fields) ) 
   fi=[]; for i=1:numel(fn); if ( strmatch(fn{i},fields) ) fi=[fi i]; end; end;
    fn=fn(fi,:,:,:); fv=fv(fi,:,:,:);
end;
if ( rotate )
   disp([fn repmat({'---'},size(fn)) reshape(fv,size(fn,1),[])]'); 
else
   disp([fn repmat({'|'},size(fn)) reshape(fv,size(fn,1),[])]); 
end
return;

%------------------------------------------------------------------------------
function []=testCase()
s=repmat(struct('hi','there','hello',struct('to','me')),10,1);
tabDisp(s);
tabDisp(s,false);
