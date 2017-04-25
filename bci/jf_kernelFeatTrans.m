function [z]=jf_kernelFeatTrans(z,varargin);
% use nystrom method to transform to an approx kernel feature space
% Options:
%  dim      -- the dimension(s) which contains the trials ('epoch')
%  kerType  -- [str] or [function_handel] kernel function to use ('linear')
%  kerParm  -- [cell-array] of arguments to give to the kernel function ({})
%  rank     -- rank to use for the approximate feature transformation   (100)
%  recX     -- [bool] record the X data for later test runs? [Warning: uses a *lot* of RAM] (1)
%
opts=struct('dim',[],'kerType','linear','kerParm',{{}},'recX',1,'rank',100,'Z',[],'W',[],'subIdx',[],'verb',0);
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

% do the work
szX=size(z.X);
if ( isempty(opts.Z) && isempty(opts.W) ) % sample and compute
  [z.X,W,Xsamp]= nystromkerfeattrans(z.X,opts.kerType,opts.rank,'dim',dim,opts.kerParm{:},varargin{:});
else % use previously computed info
  Xsamp=opts.Z; W=opts.W;
  Kcm = compKernel(Xsamp,z.X,opts.kerType,'dim',opts.dim,opts.kerParm{:},varargin{:});
  z.X = W'*Kcm;
end
% reshape back to n-d if it's not right
if ( ndims(z.X)==2 && numel(dim)>1 )
   z.X = reshape(z.X,[rank,szX(dim)]);
end

% setup the diminfo
odi = z.di;
z.di = [mkDimInfo(size(z.X,1),1,'kerfeat',[],[]); odi([dim(:)' 3:min(dim)-1 end]) ];
z.di(1).info = struct('kerType',opts.kerType,'kerParm',{opts.kerParm{:} varargin{:}});

summary=sprintf('%s kernel over %s',opts.kerType,...
                [sprintf('%s+',z.di(1:numel(dim)-1).name) z.di(numel(dim)).name]);
info=struct('odi',odi,'W',W); % info to apply to new data
if ( opts.recX ) 
  info.X=Xsamp;
  info.W=W;
  info.testFn={'jf_kernelFeatTrans' opts 'Z' info.X 'W' info.W};
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------
function testCase()
  z=jf_mksfToy();
  zk=jf_kernelFeatTrans(z,'kerType','rbf','rank',10);

  zk=jf_kernelFeatTrans(jf_cov(z),'kerType','expndist','rank',10,'kerParm',{'covlogppDist' .5});

  % using the full kernel
jf_cvtrain(jf_compKernel(jf_compDist(jf_cov(jf_fftfilter(oz,'bands',[6 8 28 30])),'distType','covlogppDist'),'kerType','expn','kerParm',1),'objFn','klr_cg')
% using the nystrom approx
jf_cvtrain(jf_kernelFeatTrans(jf_cov(jf_fftfilter(oz,'bands',[6 8 28 30])),'rank',100,'kerType','expndist','kerParm',{'covlogppDist',1}),'objFn','lr_cg')



% test with multi-kernels...
% full kernel computation
zk=jf_compKernel(zpp,'kerType','expdist','kerParm',{'covlopppdist',5,.5,-.9});
zkm=jf_mean(zk,'dim','freq');
jf_cvtrain(zkm,'objFn','klr_cg');
% feat-extraction
zkf=jf_kernelFeatTrans(zpp,'rank',1000,'kerType','expdist','kerParm',{'covlogppDist',5,.5,-.9});
zkfm=jf_mean(zkf,'dim','freq');
jf_cvtrain(zkfm,'objFn','lr_cg')
% mean-first feature extraction..... Much better performance...
zkfmu=jf_kernelFeatTrans(zpp,'rank',1000,'kerType','expdist','kerParm',{'fmucovlogppDist',5,.5,-.9});
jf_cvtrain(zkfmu,'objFn','lr_cg')
