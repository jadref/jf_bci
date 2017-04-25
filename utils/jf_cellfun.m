function [res]=cellfun(fun,cll,varargin)
% provide a matlab-6 equivalent of 7's more flexible cellfun method
% which allows arbitary function handel/string inputs
uniformp=true;
if ( ~isempty(varargin) && isequal(varargin{1},'UniformOutput') )
   uniformp=varargin{2}; varargin=varargin(3:end);
end
if ( isstr(fun) ) % translate matlab 6 strings to function equavilents
   switch fun
    case 'isclass'; fun='isa';
    case '';
  end
end
%   res=builtin('cellfun',fun,cll,varargin{:});
%elseif ( isa(fun,'function_handle') )
res=cell(0);
if ( ~iscell(cll) ) cll={cll}; end; % make cell if not one
for i=1:numel(cll)
   res{i}=feval(fun,cll{i},varargin{:});
end


% convert to uniform output
if ( uniformp && isempty(res) ) res=[]; return; end;
if ( uniformp && numel(res)>0 && iscell(res) && ...
     ~iscell(res{1}) && ~isempty(res{1}) && ~isstr(res{1}) )
   nres=size(res);  rsz=size(res{1}); rcls=class(res{1});
   for objId=2:numel(res);
      if ( ~isequal(class(res{objId}),rcls) || ...
           ~isequal(size(res{objId}),rsz) )
         uniformp=false;
         break;
      end
   end
      
   if ( uniformp ) % Uniformize, but make sure its size is nice!
                   % add in extra dimension to concatentate along.
      for i=1:numel(res); res{i}=reshape(res{i},[1 rsz]); end;
      
      if(prod(rsz)<=1) rsz=1; else rsz=rsz(rsz>1); end;
      if(prod(nres)<=1) nres=1; else nres=nres(nres>1); end;
      res=reshape(cat(1,res{:}),[nres,rsz]);
   end;       
end

% end
   
