function [s,str]=unserialize(str,lang)
if ( nargin<2 || isempty(lang) ) lang='matlab'; end;
if ( ~isstr(str) ) str=char(str); end;
indent='';

matlabTokens=serializeLangTokens('matlab');

% transform from input langauge into matlab
switch lang
 case 'matlab'; 
  str=strrep(str,sprintf(',\n'),sprintf(',...\n'));  
 case 'json'; % transform into matlab syntax
  % get the token types to map between them
  jsonTokens=serializeLangTokens('json');

  str=strrep(str,jsonTokens.keyvalsep, matlabTokens.keyvalsep);
  str=strrep(str,jsonTokens.prestruct, matlabTokens.prestruct);
  str=strrep(str,jsonTokens.poststruct,matlabTokens.poststruct);
  str=strrep(str,jsonTokens.prearray,  matlabTokens.precell);  % arrays -> cells
  str=strrep(str,jsonTokens.postarray, matlabTokens.postcell); % arrays -> cells
  str=strrep(str,sprintf('\n'),sprintf('...\n'));
  for j=1:size(matlabTokens.stresc,1); % unescape the string
     if ( matlabTokens.stresc{j,1}=='''' ) % quotes are special case
        str=strrep(str,matlabTokens.stresc{j,2},'''''');      
     else
        str=strrep(str,matlabTokens.stresc{j,2},matlabTokens.stresc{j,1}); 
     end
  end
  
  % Now deal with ndmx type and convert to normal matlab array
  % 1st search for correctly nested string bits
  nesting=compnesting(str,'(',')'); % set bracketed regions -- only structs
  % find such regions which start with struct(data
  matlabprendmxjson='struct(data,';
  strts=[0 strfind(str,matlabprendmxjson)]; % this gives set of ndmx's to convert
  fins(1)=0;
  nstr='';
  for ind=2:numel(strts);
     strt=strts(ind);
     % find this ndmxs closing brace
     fin=strt+numel(matlabprendmxjson);
     fin=fin+find(nesting(fin+1:end)==nesting(fin)-1,1,'first');%ends next time nesting lower than at start
     fins(ind)=fin;
     ndmxstr=str(strt:fin);
     % find the type in this block
     typeIdx=strfind(ndmxstr,jsonTokens.prendmxtype(1:end-1));
     szIdx  =strfind(ndmxstr,jsonTokens.prendmxsz(1:end-1));
     typestr=ndmxstr(typeIdx+numel(jsonTokens.prendmxtype):end-1); 
     typestr(find(typestr==''''))=[]; % remove the quotes
     szstr  =ndmxstr(szIdx+numel(jsonTokens.prendmxsz):typeIdx-1);
     datastr=ndmxstr(numel(matlabprendmxjson)+1:szIdx-1);
     switch (lower(typestr)); % need to convert data to numeric array format        
      case 'numeric';      
       datastr([1 end])=[matlabTokens.prearray matlabTokens.postarray];
      case 'string';
      case 'cell';         % do nowt
     end
     % ensure size string is valid numeric type array
     szstr([1 end])=[matlabTokens.prearray matlabTokens.postarray];
     
     if ( isempty(matlabTokens.prendmxtype) && isempty(matlabTokens.postndmxtype) )
        typefrmtStr='%.0s'; % don't include type info      
     else
        typefrmtStr='%s';   % do include type info         
     end
     % build the replacement text
     ndmxstr=sprintf([matlabTokens.prendmx ...
                      matlabTokens.prendmxdata '%s' matlabTokens.postndmxdata,...
                      matlabTokens.prendmxsz '%s' matlabTokens.postndmxsz,...
                      matlabTokens.prendmxtype,typefrmtStr,matlabTokens.postndmxtype,...
                      matlabTokens.postndmx...
                     ],...
                     datastr,szstr,typestr);
     % insert in-place
     nstr=[nstr str(fins(ind-1)+1:strts(ind)-1) ndmxstr];     
  end
  % insert any leftovers
  nstr=[nstr str(fins(end)+1:end)];
  % replace with the new one
  str=nstr;
end


% find top-level arrays *inside structs* and convert to double nested to stop
% matlab creating struct arrays when used as field values inside a struct
structNesting=compnesting(str,matlabTokens.prestruct(end),matlabTokens.poststruct);
arrayNesting =compnesting(str,matlabTokens.precell, matlabTokens.postcell);  
arrayNesting(structNesting<=0)=0; % only arrays inside structs  
strts=find(arrayNesting==1 & diff([0 arrayNesting])>0); 
fins=find(arrayNesting==1 & diff([arrayNesting 0])<0)+1;
if ( numel(strts)~=numel(fins) ) error('parsing broken'); end;
strts(end+1)=numel(str)+1; nstr=str(1:strts(1)-1); 
for i=1:numel(strts)-1; 
   sstrt=find(structNesting(1:strts(i))==structNesting(strts(i))-1,1,'last')+1; % start enclosing struct
   if( strmatch(str(sstrt+(-numel(matlabTokens.prestruct)+1:0)),matlabTokens.prestruct) )      
      nstr=[nstr matlabTokens.precell matlabTokens.precell ...
            str(strts(i)+1:fins(i)-1) ...
            matlabTokens.postcell matlabTokens.postcell ...
            str(fins(i)+1:strts(i+1)-1)];
   else % leave it alone!
      nstr=[nstr str(strts(i):strts(i+1)-1)];
   end
end
str=nstr;

s = eval(str);
return

function [nesting]=compnesting(str,pre,post)
nesting=zeros(size(str),'single'); 
nesting(strfind(str,pre))=1; nesting(strfind(str,post))=-1;
nesting=cumsum(nesting); % nesting >0 indicates inside


%----------------------------------------------------------------------------
function testCase()
tt=struct('this','is','very',struct('a','test','of','stuff'),'array',[1 2 3],'cell',{{'a' 8 struct('help',1)}})
str=serialize(tt,[],'json')
[ttt,ustr]=unserialize(str,'json')

str=serialize(tt,[]);
[ttt,ustr]=unserialize(str);

% test size transmission
str=serialize(randn([2 2]),'','json')
[ttt,ustr]=unserialize(str,'json')
str=serialize({1 2;3 4},'','json')


str=serialize({'fname','options',{'1' '2' '3';'4' '5' '6';'7' '8' '9'}},'','json')
s=unserialize(str,'json')

[obj,mstr]=unserialize('{ ''hello'', [] }','json')

str=serialize([1 2 3 4 5 6 7],'','json')


s=struct('method','disp','params',{{'hello world'}},'version',1)
str=serialize(s,'','json')
[us ustr]=unserialize(str,'json')
