function [z]=jf_compDist(z,varargin);
% compute the pairwise distance between every pair of examples
% Options:
%  dim      -- the dimension(s) which contains the trials ('epoch')
%  distType -- [str] or [function_handel] distance function to use ('sqDist')
%  distParm  -- [cell-array] of arguments to give to the distance function ({})
%  Z        -- [n-d] the set of values to compute the kernel with?
%              OR
%              [C x 1] sub-set of indices to use for the distance computation
%  X        -- [n-d] the set of values to compute the kernel with
%  recX     -- [bool] record the X data for later test runs? [Warning: uses a *lot* of RAM] (1)
%
opts=struct('dim',[],'distType','sqDist','distParm',{{}},'Z',[],'X',[],'recX',0,'subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
if(~iscell(opts.distParm)) if(isempty(opts.distParm)) opts.distParm={}; else opts.distParm={opts.distParm}; end; end

dim=n2d(z.di,opts.dim); % get the dim to work along
if ( isempty(dim) || all(dim==0) )
   if ( isfield(z,'Ydi') ) % Infer dim from Ydi
      dim = n2d(z.di,{z.Ydi(1:end-1).name},0,0); dim(dim==0)=[]; % get dim to work along
   else
      dim = ndims(z.X); % default to last dim of X
   end
end

% extract the trials to use for the computation...
szzX=size(z.X);
X = opts.X; Z=[];
if ( isempty(X) ) X=z.X;
elseif( numel(Z)==max(size(Z)) && size(Z,1)~=size(z.X,1) ) % vector and not same size as z.X
  idxc={}; for d=1:numel(szzX); idxc{d}=1:szzX(d); end;
  if ( numel(dim)==1 ) idxc{dim}=X;
  else  error('Multiple trial dims not supported yet');
  end;
  X= z.X(idxc{:});
  Z= z.X;
end
szX=size(X);
if ( isempty(Z) ) Z = opts.Z;  end
if ( ~isempty(Z) )
  if ( ~isnumeric(Z) && ~islogical(Z) )
    if ( isstruct(Z) && isfield(Z,'X') ) % extract from the other object
      Z=Z.X; 
    else 
      error('Dont know how to deal with this type of Z');
    end;
  elseif( numel(Z)==max(size(Z)) && size(Z,1)~=size(X,1) ) % vector and not same size as X
	 idxc={}; for d=1:numel(szX); idxc{d}=1:szX(d); end;
	 if ( numel(dim)==1 ) idxc{dim}=Z;
	 else  error('Multiple trial dims not supported yet');
	 end;
	 Z= X(idxc{:});
  end
end

% do the work
oX = z.X; 
z.X = feval(opts.distType,X,Z,dim,opts.distParm{:},varargin{:});

% reshape back to n-d if it's not right
if ( ndims(z.X)==2 && numel(dim)>1 )
   szX=size(X); if ( isempty(Z) ) szZ=szX; else szZ=size(Z); end;
   z.X = reshape(z.X,[szX(dim),szZ(dim)]);
end

% setup the diminfo
odi = z.di;
z.di = odi([dim(:)' dim(:)' 3:min(dim)-1 end]);
if( ~isequal(szX(dim),szzX(dim)) )
  if( numel(dim)==1 )
	 z.di(1).name=['X_' z.di(1).name];
	 z.di(1).vals=1:szX(dim);
  else
	 warning('Multi dim with subset not supported yet....');
  end
end
for di=numel(dim)+(1:numel(dim));
  z.di(di).name=[z.di(di).name '_dist'];
  if( ~isempty(Z) ) % update the info for the 2nd kernel dim
	 if( numel(dim)==1 )
		z.di(end-1).vals=1:size(Z,dim);
	 else
		warning('multi-dim not correctly supported yet....');
	 end
  end
end
z.di(end).name = 'dist';

summary=sprintf('%s distance over %s',opts.distType,...
                [sprintf('%s+',z.di(1:numel(dim)-1).name) z.di(numel(dim)).name]);
info=struct('odi',odi); % info to apply to new data
if ( opts.recX ) 
  info.X=oX;
  if( isempty(Z) ) info.Z=oX; else info.Z=Z; end;
  info.testFn={'jf_compDist' opts 'Z' info.Z};
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
z = jf_mksfToy(ceil(rand(100,1)*2));
kz= jf_compDist(z);
kz= jf_compDist(z,'Z',1:10);
