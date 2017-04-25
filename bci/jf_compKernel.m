function [z]=jf_compKernel(z,varargin);
% code to linearly map a given dimension in z to a new space
% Options:
%  dim      -- the dimension(s) which contains the trials ('epoch')
%  kerType  -- [str] or [function_handel] kernel function to use ('linear')
%  kerParm  -- [cell-array] of arguments to give to the kernel function ({})
%  Z        -- [n-d] the set of values to compute the kernel with?
%              OR
%              [C x 1] sub-set of indices to use for the distance computation
%  X        -- [n-d] the set of values to compute the kernel with
%  grpDim/grpIdx -- parameters used for multi-kernel computation, see compMultiKernel
%  recX     -- [bool] record the X data for later test runs? [Warning: uses a *lot* of RAM] (1)
%
opts=struct('dim',[],'kerType','linear','kerParm',{{}},'Z',[],'X',[],'grpDim',[],'grpIdx',[],'recX',0,'subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);
if(~iscell(opts.kerParm)) if(isempty(opts.kerParm)) opts.kerParm={}; else opts.kerParm={opts.kerParm}; end; end

dim=n2d(z.di,opts.dim); % get the dim to work along
if ( isempty(dim) || all(dim==0) )
   if ( isfield(z,'Ydi') ) % Infer dim from Ydi
      dim = n2d(z.di,{z.Ydi(1:end-1).name},0,0); dim(dim==0)=[]; % get dim to work along
   else
      dim = ndims(z.X); % default to last dim of X
   end
end
grpDim=n2d(z.di,opts.grpDim); grpDim(grpDim==0)=[];

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
mkDim=ndims(z.X)+1; % dim where the multi-kernels go...
if ( isempty(grpDim) )
   z.X = compKernel(X,Z,opts.kerType,'dim',dim,opts.kerParm{:},varargin{:});
else
   z.X = compMultiKernel(X,Z,opts.kerType,'dim',dim,'grpDim',grpDim,'grpIdx',opts.grpIdx,opts.kerParm{:},varargin{:});
end
% reshape back to n-d if it's not right
if ( ndims(z.X)==2 && numel(dim)>1 )
   if ( isempty(Z) ) szZ=szX; else szZ=size(Z); end;
   z.X = reshape(z.X,[szX(dim),szZ(dim)]);
end

% setup the diminfo
odi = z.di;
z.di = odi([dim(:)' dim(:)' 3:min(dim)-1 end]);
for di=numel(dim)+(1:numel(dim));
   z.di(di).name=[z.di(di).name '_ker'];
end
if( ~isequal(szX(dim),szzX(dim)) )
  if( numel(dim)==1 )
	 z.di(1).name=['X_' z.di(1).name];
	 z.di(1).vals=1:szX(dim);
  else
	 warning('Multi dim with subset not supported yet....');
  end
end
if( ~isempty(Z) ) % update the info for the 2nd kernel dim
  if( size(Z,dim) ~= size(X,dim) )
	 z.di(end-1).vals=1:size(Z,dim);
  else
	 warning('multi-dim not correctly supported yet....');
  end
end
%z.di(end).name = 'ker';
if ( ~isempty(grpDim) )
   mkDim = numel(dim)*2+1;    
   nK = size(z.X,mkDim);
   z.di = z.di([1:end-1 end-1:end]);
   z.di(mkDim)=odi(grpDim);
   z.di(mkDim).name=[z.di(mkDim).name '_grp'];
   if ( ~isempty(opts.grpIdx) ) % reset the vals if needed... 
      z.di(end-1).vals = 1:size(z.X,mkDim); % can I be clever here and give more sensible values?
      if ( numel(z.di(mkDim).extra)==size(opts.grpIdx,2) )
         [z.di(end-1).extra.grpIdx]=num2csl(opts.grpIdx);
      end
   end
end

if ( isempty(grpDim) ) 
   summary=sprintf('%s kernel over %s',opts.kerType,...
                   [sprintf('%s+',z.di(1:numel(dim)-1).name) z.di(numel(dim)).name]);
else
   summary=sprintf('%s multi-kernel over %s per %s',opts.kerType,...
                   [sprintf('%s+',z.di(1:numel(dim)-1).name) z.di(numel(dim)).name],odi(grpDim).name);
end
info=struct('odi',odi); % info to apply to new data
if ( opts.recX ) 
  info.X=oX;
  if( isempty(Z) ) info.Z=oX; else info.Z=Z; end;
  info.testFn={'jf_compKernel' opts 'Z' info.Z};
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
z = jf_mksfToy(ceil(rand(100,1)*2));
kz= jf_compKernel(z);

% try with a multi-kernel
kz = jf_compKernel(z,'grpDim','ch','grpIdx',eye(size(z.X,1))>0)

kc = jf_compKernel(z,'Z',[1:20])
