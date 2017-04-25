function [nms]=rfieldnames(obj)
% for arrays we *assume* they're all identical!
%
% SEE ALSO: rstruct2cell
%
% $Id: rfieldnames.m,v 1.3 2006-11-16 18:23:47 jdrf Exp $
if ( nargin==0 ) testCase; return; end; % run test case if no inputs
if ( numel(obj) > 1 ) obj=obj(1); end;
fn=fieldnames(obj);fv=struct2cell(obj);
nms=cell(0);
for i=1:numel(fn);
   if ( isstruct(fv{i}) )
      tres=rfieldnames(fv{i}); % recursively eval
      if ( ~isempty(tres) ) 
         for j=1:numel(tres); nms{end+1}=[fn{i} '.' tres{j}]; end;
      else % struct with no fields!
         nms{end+1}=fn{i};
      end
   else
      nms{end+1}=fn{i};
   end
end
nms=nms';

%----------------------------------------------------------------------------
function testCase()
a=struct('hello',struct('there',struct('is',struct('me',struct()))));
b=repmat(a,[10,1]);
rfieldnames(a)
rfieldnames(b)