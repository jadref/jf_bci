function [z]=jf_AUC(z,varargin);
% compute class-dependent AUC
%
% Options:
%   Y   -- [size(z.X,dim) x 1] or [size(z.X,dim) x nSubProb] labeling to 
%           use for the plotting                                             (z.Y)
%   Ydi -- [di-struct ndims(Y)+1 x 1] dimension info for the Y               (z.Ydi)
%          OR
%          {str} list of dimension names which contain the trials
opts=struct('Y',[],'Ydi',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

% allow over-ride the data's labelling
Ydi=opts.Ydi; if ( isempty(Ydi) && isfield(z,'Ydi') ) Ydi=z.Ydi; end;
Y  =opts.Y;   if ( isempty(Y) && isfield(z,'Y') )     Y  =z.Y;   end;
keyY=unique(Y(:));  % convert to spMx if not already done
if( numel(keyY)>3 || ~( isequal(keyY,[-1 1]') || isequal(keyY,[-1 0 1]')) ) 
   [Y,Ydi]=addClassInfo(Y,'Ydi',{Ydi(1:ndims(Y)).name}); 
end;

oz=z;
dim=0; 
if ( isempty(opts.Ydi) && isfield(z,'Ydi') )
   dim = n2d(z.di,{Ydi(1:end-1).name},0,0); dim(dim==0)=[]; % get dim to work along   
   z.di(dim(1))=mkDimInfo([],1,'subProb',[],Ydi(n2d(Ydi,'subProb')).vals); 
elseif ( ~isempty(opts.Ydi) ) 
  if( iscell(opts.Ydi) || ischar(opts.Ydi)) dim=n2d(z.di,opts.Ydi,0,0); 
  else dim=n2d(z.di,{opts.Ydi.name},0,0); 
  end;
  z.di(dim(1))=mkDimInfo([],1,'subProb',[],1:size(Y,2));
else
   dim =ndims(z.X); 
   z.di(dim(1))=mkDimInfo([],1,'subProb',[],1:size(Y,2));
end
z.di(dim(2:end))=[];  % the result has a single dim even if input has multiple Y dims
if ( numel(dim)>1 && all(diff(dim)~=1) ) error('only for conseq dims'); end;

if ( numel(dim)>1 ) % convert multi-dim to single dim
   szX=size(z.X);
   z.X=reshape(z.X,[szX(1:dim(1)-1) prod(szX(dim)) szX(dim(end)+1:end)]); 
   Y=reshape(Y,[],size(Y,numel(dim)+1));
   dim=dim(1);
end

szX=size(z.X);
L=size(Y,ndims(Y));
if(L==2) 
   idx={}; for di=1:ndims(Y); idx{di}=1:size(Y,di); end;
   if( Y(idx{1:end-1},1)==-Y(idx{1:end-1},2) ) L=1; end;
end

auc =zeros([szX(1:dim-1) L szX(dim+1:end)],class(z.X));
aidx={}; for d=1:ndims(auc); aidx{d}=1:size(auc,d); end;
sidx=[];
for spi=1:L; % do the auc comp
   aidx{dim(1)}=spi;
   [auc(aidx{:}),sidx]=dv2auc(Y(:,spi),z.X,dim(1),sidx); 
end
z.X=auc;

% update the meta-info
if( L==1 ) 
   if ( isstruct(Ydi) )
      if( numel(Ydi(n2d(Ydi,'subProb')).vals)>L ) % re-label if just class names
         if(iscell(Ydi(end-1).vals)) z.di(end-1).vals={sprintf('%s v %s',Ydi(end-1).vals{:})}; 
         else                        z.di(end-1).vals={sprintf('%d v %d',Ydi(end-1).vals)}; 
         end
      end
   else 
      z.di(end-1).vals={'-1 v +1'};
   end
end;
z.di(end).name='auc';z.di(end).units='auc';

% remove invalid labeling info 
if ( isfield(z,'Y') ) z.Y=[];  end;
if ( isfield(z,'Ydi') ) z.Ydi=[]; end;
if ( isfield(z,'foldIdxs') ) z.foldIdxs=[]; end;
% record the prep
summary=sprintf('%ss -> %d subProbs',oz.di(dim).name,size(z.X,n2d(z,'subProb')));
info=[];
z=jf_addprep(z,mfilename,summary,opts,info);
return;

