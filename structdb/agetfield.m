function [res,found]=agetfield(obj,fieldname,default,uniformp)
% [res,found]=agetfield(obj,fieldname,default,uniformp)
% array and default value aware version of getfield.
% works on cell arrays and structure arrays.
% returns an array of results when all the field values are of compatable
% types or a cell-array otherwise. (set uniformp to false to override and
% force cell-array outputs);
%Inputs:
% obj       -- struct/cell-array 
% fieldname -- name of the field to extract
% default   -- default value to use if field not found           ([])
%              'agetfield::error' means throw an error if not found.
% uniformp  -- return matrix (i.e. not cell array) if possible.. (2)
%      N.B. for
%          uniformp==1 we leave singlention dims intact
%          uniformp==2 we remove all trailing singlenton dimesions from
%                      each level of indirection.
%          uniformp==3 we remove all singlenton dims
%Outputs:
% res       -- input sized cell-array of found values
% found     -- indicator matrix of where we found values
if ( nargin < 2 ) error('Insufficient arguments'); end;
usedefault=true;
if ( nargin < 3 ) default=[]; 
elseif ( isequal(default,'agetfield::error') ) usedefault=false; 
end
if ( nargin < 4 ) uniformp=2; end; % return matrix if uniform outputs?
if ( ischar(fieldname) ) fieldname=split(fieldname,'.'); end;

% strip leading singleton's
if ( isobject(obj) ) obj=struct(obj); end;
while ( iscell(obj) && numel(obj)==1 ) obj=obj{1}; end;
while ( isstruct(obj) && numel(obj)==1 && numel(fieldname)>1 ) 
   if ( isfield(obj,fieldname{1}) ) % get this sub-field
      obj=obj.(fieldname{1}); fieldname=fieldname(2:end);
   else
      break; % stop if cant find the fieldname!
   end;
end

% now get the array elements
if ( isempty(obj) ) 
   if ( usedefault ) res=default; found=false(size(res)); end;
elseif ( numel(obj) == 1 )
   
   if ( usedefault && ~isfield(obj,fieldname{1}))
      res=default; found=false(size(res));
   else
      res=obj.(fieldname{:});  found=true(size(res));
   end      

elseif ( iscell(obj) ) % cell array, recursively call on each element
   res=cell(size(obj));
   for objId=1:numel(obj);
      [res{objId},rfound]=agetfield(obj{objId},fieldname,default,uniformp);
      found(objId)=any(rfound(:)); % N.B. say found if sub-call found something
   end;   
   
elseif ( isstruct(obj) ) % struct array

   res=cell(size(obj)); found=true(size(obj));
   if ( numel(fieldname) > 1 ) % use nested version
      for objId=1:numel(obj);
         [tres,tfound]=rgetfield(obj(objId),fieldname);
         if ( tfound ) res{objId}=tres; 
         elseif ( usedefault ) res{objId}=default; found(objId)=false;
         else error('Unknown fieldname'); end;
      end;
   else       
      if ( usedefault && ~isfield(obj,fieldname{1}) )
         for i=1:numel(res); res{i}=default; end;
         found(:)=false;
      else
         res={obj.(fieldname{1})};
         res=reshape(res,size(obj));
      end
   end   
else
   error('Dont know what to do with this object');
end

% Uniform-ise the results if wanted
if ( uniformp && numel(res)>1 && iscell(res) && ...
     ~iscell(res{1}) && ~isempty(res{1}) && ~ischar(res{1}) )
   nres=size(res);  rsz=size(res{1}); rcls=class(res{1});
   for objId=2:numel(res);
      if ( ~(isequal(class(res{objId}),rcls) || ... % same class or up-convertable
             isfloat(res{objId})==isfloat(res{1}) || isinteger(res{objId})==isinteger(res{1}) ) || ...
           ~isequal(size(res{objId}),rsz) )
         uniformp=false;
         break;
      end
   end
   
   if ( uniformp ) % Uniformize, but make sure its size is nice!
      % add in extra dimension to concatentate along.
      for i=1:numel(res); res{i}=reshape(res{i},[1 rsz]); end;

      % squeeze if uniformp > 1
      if(prod(rsz)<=1) rsz=1; 
      elseif(uniformp>2) rsz=rsz(rsz>1); % remove all singlentons
      elseif(uniformp>1) rsz=rsz(1:find(rsz>1,1,'last'));% remove only trailing singlentons
      end;
      if(prod(nres)<=1) nres=1; 
      elseif( uniformp>2) nres=nres(nres>1); % remove all singlentons
      elseif( uniformp>1) nres=nres(1:find(nres>1,1,'last'));%remove only trailing singlentons
      end
      res=reshape(cat(1,res{:}),[nres,rsz]);
   end; 
end

return;

function [res]=split(str,c) % used to turn string into sub-bits
tmpstr=str;
di=[0 find(str==c) numel(str)+1];
for i=1:numel(di)-1;
   res{i}=tmpstr(di(i)+1:di(i+1)-1);
end

function [obj,found]=rgetfield(obj,fieldname)
% nested getfield, i.e. index into sub-structures to get the field
found=true;
for fi=1:numel(fieldname);
   if ( isobject(obj) ) obj=struct(obj); end; % convert objects to structs
   if ( isfield(obj,fieldname{fi}) ) 
      obj=obj.(fieldname{fi});  % get this sub-field
   else
      found=false; break; 
   end;
end
