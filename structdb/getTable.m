function [varargout]=getTable(db,query,varargin);
% [fv1,fv2,...]=getTable(db,match,f1,f2,...)
%
% Inputs:
%  db -- database of struts/tables to match over
%  query -- [numel(query) logical] members of db to extract info from
%           OR
%           {'fn',fv,..} --  set of field names and values to match
%  fn1,fn2,.. -- field names to extract
% get the table of field values from the structure data-base db
% Extract the optional arguments we need.
tbl={};
opts=struct('entIdx',[],'uniform',1);
if(numel(varargin)>0 && isstruct(varargin{1})) % struct->cell
   varargin=[fieldnames(varargin{1})'; struct2cell(varargin{1})'];
end
mOpts=false(numel(varargin),1);
for i=1:2:numel(varargin); % process the options 
   if( isfield(opts,varargin{i}) ) % leave unrecognised entries
      opts.(varargin{i})=varargin{i+1};
      mOpts(i:i+1)=true;
   end;
end
varargin(mOpts)=[]; % remove processed arguments

% get the entries that matched
if ( isempty(query) ) idx=1:numel(db); 
else
   if ( ~iscell(query) ) query={query}; end;
   idx=cellfun(@(x) structmatch(x(1),query{:}),db);
   idx=find(idx);
end
%fprintf('%d entries matched',numel(idx));
if ( numel(idx)==0 ) return; end;
% loop over the matched entries extracting the bits we need?
for i=1:numel(varargin); % loop over fields
   tbli = agetfield(db(idx),varargin{i},0,2); % subj x fieldvals
   clear tmptbl; %tmptbl=zeros(numel(tbli),1);
   if ( ( iscell(tbli) && ~ischar(tbli{1}) ) ...
        || ~isempty(opts.entIdx) ) % flatten out the results
      for j=1:size(tbli,1); 
         if ( iscell(tbli) ) 
            if ( iscell(tbli{1}) ) 
               tmp=tbli{j}{min(end,opts.entIdx)}; 
            else
               tmp=tbli{j}(min(end,opts.entIdx)); 
            end
         else
            tmp=tbli(j,min(end,opts.entIdx));
         end
         if ( ~isnumeric(tmp) ) tmptbl{j}=tmp; else tmptbl(j,:)=tmp; end
      end
   else
      tmptbl=tbli;
   end
   tbl{i}=tmptbl; % fields x subj x fieldentries
end
tbl=tbl';
% finaly flatten the returned structure if that's possible+wanted
if ( opts.uniform>0 ) 
   tmp=uniformizeCellArray(tbl,opts.uniform-1);
   if ( ~isempty(tmp) ) tbl=tmp; end;
end;
if ( nargout > 1 ) varargout=tbl; else varargout{1}=tbl; end;


