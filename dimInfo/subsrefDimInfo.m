function [idx,di]=subsrefDimInfo(di,varargin);
% Function to sub-reference into a dim-info structure to return a smaller one
%
% [idx,di]=subsrefDimInfo(di,idx1,idx2,idx3,...) 
% OR
% [idx,di]=subsrefDimInfo(di,varargin)
% 
% Inputs:
%  di   -- dim-info struct
%  idx1,idx2,... -- logical or numeric index's into the 1,2...rd dim of di
%                   N.B. empty or unspecifed dims are assumed to be full range
% Options: 
%    dim  -- [nDims x 1] OR {nDims x 1 str} the dimension(s) to index down
%    vals -- [int] OR or {str}
%               OR 
%            {nDims x [int] OR OR {cell}}
%            the set of values along the dimensions as specified by dim to keep/remove
%            N.B. #str# or %str% for string values means treat str as regular expression to match
%    idx  -- [int] OR [logical]
%               OR 
%            {nDims x [int] OR [logical]}
%             indexs along the dimensions as specified by dim to keep/remove
%    range-- [str] idx/vals specifies a range to remove. which is either:
%            oneof {'before','after','between','outside',[]} the spec elements ([])
%    valmatch - [str] type of value match to use, one of {'exact','icase','regexp','nearest'} ('nearest')
%            N.B. nearest = case insensitive for strings
%    mode -- [str] reject or retain the indicated elements
%             {'reject','retain'}                                   ('retain')
%             'reject' = return *non-matching* elements
%             'retain' = return matching elements
%    outorder -- [str] what order are the outputs in, one-of:       ('input')
%                input - same order as input,  idx - same order as idx/vals
% Outputs:
%  idx  -- the indices used to subsref the matrix described by di
%  di   -- the input dim-info restricted to the selected indexs


% Now process the more complex dim,val/idx arguments
opts=struct('dim',[],'vals',[],'idx',[],'range',[],'mode','retain','valmatch','exact','outorder','input');
[opts,varargin]= parseOpts(opts,varargin);

if( isstruct(di) && isfield(di,'di') ) di=di.di; end;
if( ~any(strcmp(opts.valmatch,{'exact','icase','regexp','nearest'})) )
  warning('unrecognised val-match type')
end

% Convert them to the simpler format
if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
if ( isempty(opts.dim) ) dim=1:max(1,numel(di)-1);
elseif ( iscell(opts.dim) )
   dim=zeros(numel(opts.dim),1);
   for i=1:numel(opts.dim)
      t=strmatch(opts.dim{i},{di.name},'exact'); 
      if(isempty(t)) 
         t=strmatch(opts.dim{i},{di.name}); % inexact match
         if ( isempty(t) )
            warning('Couldnt find dim: %s',opts.dim{i}); continue; 
         end
      end;
      if( numel(t)>1 ) warning('%s matched %d times',opts.dim{i},numel(t)); end;
      dim(i)=t(1);
   end;
elseif ( isnumeric(opts.dim) ) dim=opts.dim; 
end
dim(dim<0)=dim(dim<0)+numel(di);

idx=cell(max(1,numel(di)-1),1); vals=cell(max(1,numel(di)-1),1);
if ( numel(varargin)>0 ) idx(1:numel(varargin))=varargin; end; % extract from pos info
if ( ~isempty(opts.idx) ) % specified an idx
   if ( ~iscell(opts.idx) ) opts.idx={opts.idx}; end;
   if ( numel(opts.idx) > numel(dim) ) error('dim & idx must have same number of entries'); end;
   for i=1:numel(opts.idx); % only for spec dims
      if ( dim(i)>0 ) idx{dim(i)}=opts.idx{i}; end
   end;
end
if ( ~isempty(opts.vals) ) % specified a set of values to keep
   if ( ischar(opts.vals) ) opts.vals={{opts.vals}}; end;
   if ( iscell(opts.vals) && ischar(opts.vals{1}) ) opts.vals={opts.vals}; end;
   if ( ~iscell(opts.vals) ) opts.vals={opts.vals}; end;
   if ( numel(opts.vals) > numel(dim) ) error('dim & vals must have same number of entries'); end;
   for i=1:numel(opts.vals); if( dim(i)>0 ) vals{dim(i)}=opts.vals{i};  end; end;
end

% work out the size of the matrix summarised by di.
for d=1:numel(di); szX(d)=numel(di(d).vals); end;

% compute the set of indices we want to use
for d=1:numel(idx); 
   if ( ~isempty(idx{d}) ) % specified as a set of indices to remove
     if ( isequal(idx{d},'skip') ) continue; end; % skip one's logical has already filled
     if ( islogical(idx{d}) && numel(idx{d})==max(size(idx{d})) ) % vector logicals to index lists
        idx{d}=find(idx{d}); 
      end;
      if ( ~isnumeric(idx{d}) && ~islogical(idx{d}) ) error('Index should be numeric or logical');end
      if ( idx{d} < 0 )  idx{d}=szX(d)+idx{d}(idx{d}<0)+1; end;
      if ( ~isempty(idx{d}) )
        if ( isnumeric(idx{d}) )
          if ( min(idx{d}(:)) < 1)  error('Negative indicies along %d',d);  end
          if ( max(idx{d}(:)) > szX(d) ) 
            warning('Idx > size of X along %s removed',di(d).name);
            if ( isempty(opts.range) ) idx{d}(idx{d}>szX(d)) = [];
            else idx{d}(idx{d}>szX(d)) = szX(d);
            end
          end
        elseif ( islogical(idx{d}) )
          % comp num dims, dealing with matlabs rep of vector as matrix....
          if ( ndims(idx{d})==2 && numel(idx{d})==max(size(idx{d})) ) nds=1; 
          else nds=ndims(idx{d}); 
          end 
          for td=1:nds; 
            if ( size(idx{d},td)>szX(d+td-1) )
              error('logical indices behyond size of x');
            end
            if ( td>1 ) idx{d+td-1}='skip'; end % mark as already filled
          end
        end
      end % isempty(idx{d})  
   elseif ( ~isempty(vals{d}) ) % set of values to keep
      ind=false(szX(d),numel(vals{d})); % match indicators
      for i=1:numel(vals{d}); % Loop over patterns
         v1=vals{d}(i); if ( iscell(v1) ) v1=v1{:}; end;
         if( ~iscell(di(d).vals) && isnumeric(di(d).vals) && isnumeric(v1) )%use faster array match
            ii=find(di(d).vals==v1,1,'first');
            if ( isempty(ii) && strcmp(opts.valmatch,'nearest') ) 
               [ans,ii]=min(abs(di(d).vals-v1)); % closest match fallback
            end;
            ind(ii,i)=true;
         else % fall-back on looped match
            for j=1:szX(d); 
               v2=di(d).vals(j); if(iscell(v2)) v2=v2{:}; end;
               if( isequal(v1,v2) ) ind(j,i)=true;break; end; % until we find a match
               if( ischar(v1) )
                 if ( isequal(opts.valmatch,'regexp') )            % regexp match
                   ind(j,i)= ~isempty(regexp(v2,v1,'once')); 
                 elseif( ( v1(1)=='#' & v1(end)=='#' ) || ( v1(1)=='%' && v1(end)=='%' ) ) % regexp
                   ind(j,i)= ~isempty(regexp(v2,v1(2:end-1),'once')); 
                 elseif( isequal(opts.valmatch,'icase') || isequal(opts.valmatch,'nearest') ) % case insensitve
                   ind(j,i)= strcmpi(v2,v1); 
                 end;
               end
            end
         end
      end % patterns/values
      if( ~all(any(ind,1)) )
         fprintf('%s: Warning Some vals didnt match anything: #(%s)\n',...
                 mfilename,sprintf('%d,',find(~any(ind,1)))); 
       end;
       
       if ( strcmp(opts.outorder,'input') )
          idx{d}=any(ind,2);
       else % return in vals/idx order
          idx{d}=[];
          for vi=1:size(ind,2);
             idx{d}=[idx{d}; find(ind(:,vi))];
          end
       end
       %idx(idx>0);      
   else
      if ( any(d==dim) && strcmp(opts.mode,'reject') ) 
         idx{d}=[]; 
      else
         idx{d}=1:szX(d);
      end
   end

   if ( isempty(idx{min(d,end)}) ) 
     warning(['Nothing specified to ',opts.mode]);
   end
   % if ( strcmp(opts.outorder,'input') && isnumeric(idx{d}) )
   %   idx{d}=sort(idx{d},'ascend'); % out in input order
   % end
end % dims



% use idx to update the dim info
% use the range spec to update the idx -- N.B. only for *dim*
if ( ~isempty(opts.range) )
   if ( islogical(idx{dim}) ) idx{dim}=find(idx{dim}); end;
      nidx = numel(idx{dim});
      switch lower(opts.range);
       case 'before';                if(nidx~=1) warning('1 idx for before, last found used');end
        idx{dim} = 1:idx{dim}(end)-1;
       case 'incbefore';             if(nidx~=1) warning('1 idx for before, last found used');end
        idx{dim} = 1:idx{dim}(end);
       case 'after';                 if(nidx~=1) warning('1 idx for after, 1st found used'); end
        idx{dim} = idx{dim}(1)+1:szX(dim);
       case 'incafter';              if(nidx~=1) warning('1 idx for after, 1st found used'); end
        idx{dim} = idx{dim}(1):szX(dim);
       case {'incbetween','between'};if(nidx~=1 && mod(nidx,2)~=0) error('2 idx for between');end
        ind=false(szX(dim),1);
        for i=1:2:nidx; ind(idx{dim}(i):idx{dim}(min(end,i+1)))=true; end;
        idx{dim} = ind; %idx{dim}(1):idx{dim}(min(end,2));
       case 'outside';        if(nidx>2||nidx<1) error('2 idx for outside');end
        ind=true(szX(dim),1);
        for i=1:2:nidx; ind(idx{dim}(i):idx{dim}(i+1))=false; end;
        idx{dim} = ind; %[1:idx{dim}(1)-1 idx{dim}(min(end,2))+1:szX(dim)];
       otherwise;
        if ( ~isempty(opts.range) ) 
           error('Unrecognised range spec: %s',opts.range); 
        end;
     end
end

if ( strcmp(opts.mode,'reject') ) 
   for d=dim; % invert
     if ( isnumeric(idx{d}) ) tmp=idx{d}; idx{d}=true(szX(d),1); idx{d}(tmp)=false;
     elseif ( islogical(idx{d}) ) idx{d}=~idx{d}; 
     end     
   end 
elseif ( strcmp(opts.mode,'retain') ) ;
else error('Mode must be reject or retain');
end

% use the idxs to update the dim info
if ( nargout>1 ) 
  for d=1:numel(idx);
    if ( isnumeric(idx{d}) || (islogical(idx{d}) && numel(idx{d})==max(size(idx{d}))) )
      di(d).vals =di(d).vals(idx{d});
      if ( (numel(idx{d})<szX(d) || (islogical(idx{d}) && sum(idx{d})<szX(d))) && numel(di(d).extra)==szX(d) ) 
        di(d).extra=di(d).extra(idx{d});  
      end
    elseif( islogical(idx{d}) ) % BODGE: compress out when n-dim logical index is used
      ndi=di(d);
      for dd=2:ndims(idx{d}); 
        ndi.name=[ndi.name '_' di(dd).name];
        ndi.vals=repmat(ndi.vals(:),[1 numel(di(dd).vals)]);
      end
      ndi.extra=struct();
      di(d)=ndi;
      di(d+(1:ndims(idx{d})-1))=[];
    end
  end
end

% BODGE: clean out the idx's which wont use
skipd=0; 
for d=1:numel(idx); 
  if ( isequal(idx{d},'skip') ) skipd=d; 
  elseif ( skipd )
    if ( ~(isequal(idx{d},1) || isequal(idx{d},true)) ) 
      warning('non singlentions after logical doesnt work');
    end
  end
end
if ( skipd>0 ) idx(skipd:end)=[]; end;

return;
%------------------------------------------------------------------------
function testCase()
di=mkDimInfo([10 20 110],'ch',[],{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'},'time',[],[],'epoch',[],[]);

% Normal indexing -- with repetition of indicies
[idx2,di1]=subsrefDimInfo(di,'idx',{floor(rand(50,1)*numel(di(1).vals))+1,[false(50,1);true(50,1)]})
% Value based indexing
[idx,sdi]=subsrefDimInfo(di,'dim','epoch','idx',1)
% Set based indexing
[idx,sdi]=subsrefDimInfo(di,'dim','epoch','vals',[10 100])
[idx,sdi]=subsrefDimInfo(di,'dim','epoch','range','outside','vals',[10 100])
[idx,sdi]=subsrefDimInfo(di,'dim','epoch','range','between','vals',[10 100])
[idx,sdi]=subsrefDimInfo(di,'dim','ch','range','before','vals',{'4'})
[idx,sdi]=subsrefDimInfo(di,'mode','reject','dim','ch','range','before','vals',{'4'})

[idx,sdi]=subsrefDimInfo(di,'mode','reject','dim',{'ch' 'epoch'},'vals',{{'4' '6'}},'idx',{[] [1:4]})
[idx,sdi]=subsrefDimInfo(di,'dim',{'ch' 'epoch'},'vals',{{'4' '6'}},'idx',{[] [1:4]})
