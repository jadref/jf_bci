function [di,dim]=compressDims(di,varargin);
% compress 2 separate dimensions of z into a single new one
% Options:
%  dim -- compress entries from dim(1) into dim(2)
% Outputs:
%  di  -- compressed dimInfo
%  dim -- the indices of the dims we compressed away
opts=struct('dim',[],'relabel',0);
[opts]=parseOpts(opts,varargin);

dim=[]; % get the dim to work along
if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( isstr(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  
      mi=strmatch(opts.dim{i},{di.name}); 
      if ( numel(mi)>1 ) mi=strmatch(opts.dim{i},{di.name},'exact'); end;
      if ( numel(mi)>1 ) warning('matched multiple names'); end;
      dim(i)=mi(1);
   end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+numel(di)-1+1; % convert neg dim specs

% Do the compression
% 1 -- permute so it works
if ( ~all(abs(diff(dim))==1) ) 
   error('Only implemented for consequetive dims, sorry'); 
end

sz=[]; for d=1:numel(di)-1; sz(d)=numel(di(d).vals); end;

% update the di
ndi = di(dim(1));
ndi.name=[di(dim(1)).name '_' di(dim(2)).name];
ndi.vals=repmat(ndi.vals',[1,sz(dim(2))]); % rep down
if( isnumeric(ndi.vals) ) 
   ndi.vals=repop(single(ndi.vals),'+',(1:sz(dim(1)))'*.1);
elseif ( iscell(ndi.vals) && ischar(ndi.vals{1}) ) 
   for i=1:size(ndi.vals,1); for j=1:size(ndi.vals,2);
         ndi.vals{i,j}=[ndi.vals{i,j} '_' di(dim(2)).vals{j}];
      end
   end
end
ndi.extra=repmat(ndi.extra',[1,sz(dim(2))]); % rep down
for i=1:size(ndi.extra,1); for j=1:size(ndi.extra,2);
      if ( iscell(di(end-2).vals) )
         ndi.extra(i,j).(di(end-2).name)=di(dim(1)).vals{i};
      else
         ndi.extra(i,j).(di(end-2).name)=di(dim(1)).vals(i);
      end
   end
end

odi = di(dim); % save the removed value
% remove the compressed dim
ii=1:numel(di); ii(dim(2))=[];
di=di(ii); di(dim(1)-(dim(2)<dim(1)))=ndi;

return;

%--------------------------------------------------------------------------
function testCase()
