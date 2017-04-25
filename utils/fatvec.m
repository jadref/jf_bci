function [x,fat]=fatvec(varargin)
% simple method to take a list of named arbitary sized arrays as input and
% vectorise them and return a structure (i.e. the vector allocation table)
% which can be used to recover the orginal inidicies *AND* their size info.
% e.g.  [x,lyt]=fatvec('hello',eye(3,3),'there',randn(10,1));
%       hello=x(lyt.hello); there=x(lyt.there);
% N.B. disadvantage is fat is the same size as x
if numel(varargin)==1 && iscell(varargin{1}) varargin=varargin{1}; end;
if ( isnumeric(varargin{1}) ) % incrementall grow if given first
   x=varargin{1};lst=numel(x); fat=varargin{2}; strt=3;; 
elseif ( numel(varargin)==2 ) % single arg special case
   x=varargin{2};
   fat=struct(varargin{1},reshape(1:numel(varargin{2}),size(varargin{2})));
   return;
else
   x=[]; lst=0; fat=struct(); strt=1;
end
% populate the fat structure
for i=strt:2:numel(varargin);   
   if(isfield(fat,varargin{i})) error('No stupid, file names must be unique!');
   end
   sz=size(varargin{i+1});n=prod(sz);
   fat.(varargin{i})=single(reshape([1:n]+lst,sz));
   lst=lst+n;
end
% populate the vector itself -- so we don't incrementaly grow it
if ( isempty(x) ) x=zeros(lst,1); end;
for i=strt:2:numel(varargin);
   x(fat.(varargin{i}))=varargin{i+1};
end
return;
% ----------------------------------------------------------------------------
% testcases...
function []=testCases()
[f,fat]=fatvec('hello',1:100);
[x,fat]=fatvec('hello',1,'hello',10);
hello=randn(3,3); there=10; stupid=randn(50,1);
[x,fat]=fatvec('hello',hello,'there',there,'stupid',stupid);
size(x(fat.hello)),size(x(fat.there)),size(x(fat.stupid))
norm(hello-x(fat.hello)),norm(there-x(fat.there)),norm(stupid-x(fat.stupid))

[x,fat]=fatvec(x,fat,'stupider',stupid);

tic,for i=1:100;[x,fat]=fatvec('hello',hello,'there',there,'stupid',stupid);end,toc
