function [str]=serialize(val,indent,lang)
% turn matlab structures into a string version
%
% [str]=serialize(val,indent,lang)
%
% Inputs:
%  val    -- value to serialize
%  indent -- [str] indentation string for start of each new line
%  lang   -- [str] langague to output, {'matlab','json'}
% Outputs:
%  str    -- [str] output serialized string
if ( nargin < 2 ) indent='\t'; end;
if ( nargin < 3 || isempty(lang) ) lang='matlab'; end;
langTokens=serializeLangTokens(lang,indent);

if ( isobject(val) ) val=struct(val); end; % convert obj's to structs
% convert funchandles to strings
if ( isa(val,'function_handle') ) val=func2str(val); end; 

if ( isempty(val) ) 
   if ( isstruct(val) )  str=sprintf([langTokens.prestruct,langTokens.poststruct]);
   elseif ( isstr(val) ) str=sprintf([langTokens.prestr,langTokens.poststr]);
   elseif ( iscell(val) )str=sprintf([langTokens.precell,langTokens.postcell]);
   else                  str=sprintf([langTokens.prearray,langTokens.postarray]);
   end
elseif( isstr(val) )
    oval=val; val=char();
    for j=1:size(langTokens.stresc,1);
      for k=1:size(oval,1)
        val(k,:)=strrep(oval(k,:),langTokens.stresc{j,1},langTokens.stresc{j,2}); 
      end
    end
   str=val(1,:);if(size(val,1)>1)val=cellstr(val);str=[str sprintf([langTokens.strsep,'%s'],val{:})];end
   str = sprintf([langTokens.prestr,'%s',langTokens.poststr],str);
elseif ( isnumeric(val) || islogical(val) )
   if ( isinteger(val) ) frmtStr='%d'; else frmtStr='%g'; end;
   str=sprintf(frmtStr,val(1));
   if( numel(val)>1 )
     if ( issparse(val) ) % sparse printout....
       val=full(val); %BODGE
     end
      str=[langTokens.prearray str ...
           sprintf([langTokens.arraysep,frmtStr],val(2:end)) langTokens.postarray];
   end
elseif ( iscell(val) )
   str = serialize(val{1},[indent '\t'],lang);
   for i=2:numel(val); 
      str=sprintf(['%s',langTokens.cellsep,'%s'],str,serialize(val{i},[indent '\t'],lang));
   end
   str = sprintf([langTokens.precell,'%s',langTokens.postcell],str);
elseif ( isstruct(val) && numel(val) > 1 ) % deal with struct arrays as a special case
   str = serialize(val(1),[indent '\t'],lang);
   for i=2:numel(val); 
      str=sprintf(['%s',langTokens.arraysep,'%s'],str,serialize(val(i),[indent '\t'],lang)); 
   end
   if( numel(val)>1 ) str = sprintf([langTokens.prearray,'%s',langTokens.postarray],str); end;
elseif ( isstruct(val) && numel(val)==1 ) % individual struct as special case
   % get the name/val pairs
   str='';
   fn=fieldnames(val); fv=struct2cell(val);
   if( numel(fn)>0 ) 
      str=sprintf([langTokens.prekey,'%s',langTokens.postkey,langTokens.keyvalsep,...
                  langTokens.preval,'%s',langTokens.postval],fn{1},serialize(fv{1},[indent '\t'],lang)); end;
   for i=2:numel(fn); 
      str=sprintf(['%s',langTokens.structsep,...
                  langTokens.prekey,'%s',langTokens.postkey,langTokens.keyvalsep,...
                  langTokens.preval,'%s',langTokens.postval],str,fn{i},serialize(fv{i},[indent '\t'],lang));
   end
   str = sprintf([langTokens.prestruct,'%s',langTokens.poststruct],str);
else % don't know how to deal with this so fall back on disp
   str = evalc('disp(val)');
   str(end)=[]; % strip the final return
end

% add the size info
if ( ndims(val)>2 || size(val,1)>1 || (numel(val)>1 && (isnumeric(val) || islogical(val))) ) % with size info   
   % serialise the size info -- N.B. use explicit to stop infinite recursion
   sz=size(val); szstr=sprintf('%d',sz(1)); 
   for i=2:numel(sz); szstr=sprintf(['%s',langTokens.arraysep,'%d'],szstr,sz(i)); end
   szstr = sprintf([langTokens.prearray,'%s',langTokens.postarray],szstr);

   if ( isnumeric(val) || islogical(val) ) typestr='''numeric'''; 
   elseif ( isstr(val) )                   typestr='''string''';
   else                                    typestr='''cell'''; 
   end
   if ( isempty(langTokens.prendmxtype) && isempty(langTokens.postndmxtype) )
      typefrmtStr='%.0s'; % don't include type info      
   else
      typefrmtStr='%s';   % do include type info         
   end
   str=sprintf([langTokens.prendmx,...
                langTokens.prendmxdata,'%s',langTokens.postndmxdata,...
                langTokens.prendmxsz,'%s',langTokens.postndmxsz,...
                langTokens.prendmxtype,typefrmtStr,langTokens.postndmxtype,...
                langTokens.postndmx...
               ],...
               str,szstr,typestr);
end
return;

%--------------------------------------------------------------------------
function testCase()
tt=struct('this','is','very',struct('a','test','of','stuff'),'array',[1 2 3],'cell',{{'a' 8 struct('help',1)}})
serialize(tt)

serialize(randn([2 2]))

s=struct('method','disp','params',{{'hello world'}},'version',1)
serialize(s,[],'json')


s=struct('method','fprintf','params',{{'%s = %d'  'field'  [10]}},'version',1)
str=serialize(s)