function [x2,fat2,rmIdx,restIdx]=fatvecrmfields(x,fat,varargin)
fn=fieldnames(fat);
if ( numel(fn)>=numel(varargin) && ...
     isequal(fn(1:numel(varargin))',varargin) ) % pop front
   i=numel(varargin);
   sub=fat.(fn{i})(end);
   % build the new fat structure
   x2=x(sub+1:end);
   fat2=struct();
   for j=i+1:numel(fn);
      fat2.(fn{j})=fat.(fn{j})-sub;
   end
   if ( nargout > 2 ) rmIdx=[1:sub]'; restIdx=[sub+1:numel(x)]'; end;
elseif ( numel(fn)>=numel(varargin) && ...
         isequal(fn(numel(fn)-numel(varargin)+1:end)',varargin) ) % pop back
   i=numel(fn)-numel(varargin)+1;
   sub=fat.(fn{i})(1);
   % build the new fat structure
   x2=x(1:sub-1);
   fat2=struct();
   for j=1:i-1;
      fat2.(fn{j})=fat.(fn{j});
   end      
   if ( nargout > 2 ) rmIdx=[sub:numel(x)]'; restIdx=[1:sub-1]'; end;
else  % deal with the general case
   sub=0; fat2=struct(); rmIdx=[]; restIdx=[]; x2=[];
   for i=1:numel(fn);
      if ( ~isempty(strmatch(fn{i},varargin,'exact')) )
         sub=sub+numel(fat.(fn{i}));
         if ( nargout > 2 ) rmIdx=[rmIdx;fat.(fn{i})(:)]; end;
      else
         oidx=fat.(fn{i});
         fat2.(fn{i})=oidx-sub;
         x2(fat2.(fn{i}))=x(oidx);
         if ( nargout > 2 ) restIdx=[restIdx;fat.(fn{i})(:)]; end;
      end
   end
   x2=x2(:); % ensure this is a column vector!
end
return;
%---------------------------------------------------------------------------
function []=testcases()
hello=randn(3,3); there=10; stupid=randn(50,1);
[x,fat]=fatvec('hello',hello,'there',there,'stupid',stupid);

[x2,fat2,rmIdx,restIdx]=fatvecrmfields(x,fat,'there');fat2
norm(x(fat.hello)-x2(fat2.hello)),norm(x(fat.stupid)-x2(fat2.stupid))
tic, for i=1:100; [x2,fat2]=fatvecrmfields(x,fat,'there'); end,toc

[x2,fat2,rmIdx,restIdx]=fatvecrmfields(x,fat,'hello','there');fat2
norm(x(fat.stupid)-x2(fat2.stupid))
tic, for i=1:100; [x2,fat2]=fatvecrmfields(x,fat,'hello','there'); end,toc

[x2,fat2,rmIdx,restIdx]=fatvecrmfields(x,fat,'there','stupid');fat2
norm(x(fat.hello)-x2(fat2.hello))
tic, for i=1:100; [x2,fat2]=fatvecrmfields(x,fat,'there','stupid'); end,toc
