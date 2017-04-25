function [z]=jf_cvtest(z,varargin);
% apply a classifier to this data set
%
% Options:
%  classifier - {nSp x ???} cell array of per-subproblem classifiers
%  objFn - [str] objective function used to train the classifier
%  dim   - [int] dimension of X which contains the trials
%  spDi  - [struct] sub-problem dimension info
%  Y     - [size(z.X,dim) x L] true labels for z's trials
%  Ydi   - [struct] dimInfo for Y
%  verb  - [bool] verbosity level
%  testFn- [str] function to use to apply the classifier to the data ('linClsfr')
%            function prototype is:  f = fn(X,classifiers,dim,varargin);
opts=struct('classifier',[],'objFn','klr_cg','dim',[],'spDi',[],'Y',[],'Ydi',[],'verb',0,'testFn','linClsfr');
[opts,varargin]=parseOpts(opts,varargin);
if ( isempty(opts.classifier) )
   error('Must specify a classifier to use on the new data!');
end

% extract subProblem bits from the inputs
nSp = numel(opts.classifier);
dim=opts.dim; dim=n2d(z.di,dim);
szX=size(z.X); szX(end+1:max(dim))=1;
rdim=setdiff(1:numel(szX),dim);
Y=opts.Y; Ydi=opts.Ydi; spDi=opts.spDi;
if ( isempty(Y) && isfield(z,'Y') ) 
   Y=z.Y; 
   if ( isfield(z,'Ydi') ) 
      Ydi=z.Ydi; 
      if ( isempty(spDi) && n2d(Ydi,'subProb') ) spDi=Ydi(n2d(Ydi,'subProb')); end;
   end;
end

% apply the classifier
if ( strcmp(opts.testFn,'linClsfr') ) % kernel method, or linear input space method
   f   = linClsfr(z.X,opts.classifier,dim,varargin{:});
else % fn is a method to call to apply this classifier
   f   = feval(opts.testFn,z.X,opts.classifier,dim,varargin{:});
end
% update the object state
z.X = f;
odi = z.di;
z.di= odi([dim(:)' rdim(1) end]); 
if ( ~isempty(spDi) ) z.di(numel(dim)+1)=spDi; 
else z.di(numel(dim)+1)=mkDimInfo(nSp,1,'subProb',[],[]); 
end;

% compute some performance information (if labels are available)
spD=numel(dim)+1;
if ( ~isempty(Y) )
   if ( size(Y,spD) ~= nSp || size(f,spD)~=nSp ) 
      error('Size of Y and predictions must match!'); 
   end;   
   Ytst=Y;   if ( spD>1 ) Ytst=reshape(Ytst,[],nSp); end;
   ftst=z.X; if ( spD>1 ) ftst=reshape(ftst,[],nSp); end;
   if ( opts.verb > -1 ) fprintf('\n'); end;
   for spi=1:nSp; % N.B. we need to loop as dv2conf etc. only work on 1 sub-prob at a time
      res.tstconf(:,spi)=dv2conf(Ytst(:,spi),ftst(:,spi));
      res.tstbin (:,spi)=conf2loss(res.tstconf(:,spi),1,'bin');
      res.tstauc (:,spi)=dv2auc(Ytst(:,spi),ftst(:,spi));
      % log the performance
      if ( opts.verb > -1 ) fprintf('(out/%2d)\t NA /%0.2f\n',spi,res.tstbin(:,spi)); end
   end
end
% multi-class decoding
if ( ~isempty(Ydi) && isfield(Ydi(n2d(Ydi,'subProb')).info,'spMx') && ...
     size(z.X,n2d(z.di,'subProb'))>1 ) % multi-class results recording
   spMx=Ydi(n2d(Ydi,'subProb')).info.spMx;
   spKey=Ydi(n2d(Ydi,'subProb')).info.spKey;
   if ( ~isempty(spMx) && numel(z.di(n2d(Ydi,'subProb')).vals)>1 )
      Yl   = dv2pred(Y,ndims(Y),spMx,'ml',1);%get back the true labels by decoding the sub-prob spec
      spD  = n2d(z.di,'subProb');
      mcres= cvmcPerf(Yl,z.X,[1 spD ndims(z.X)+1],1,spMx,spKey);
      res  = mergeStruct(res,mcres);
      if ( opts.verb> -1) fprintf('-----\n(out/mc)\t'); fprintf(' NA /%0.2f\n',mcres.tstcr(:,1)); end;
   end
end

% record the summary info
summary = sprintf('over [%s%s]',z.di(2).name,sprintf(' x %s',z.di(3:end).name));
info.res= res;
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------

function [f]=linClsfr(X,classifier,dim,varargin);
% apply linear classifer(s) to the input data
szX=size(X); szX(end+1:max(dim))=1;
rdim=setdiff(1:numel(szX),dim);
Wb  = cat(2,classifier{:});
if ( size(Wb,1)-1 ~= prod(szX(rdim)) ) error('Dont know how to use this type of classifier'); end;
W   = reshape(Wb(1:end-1,:),[szX(rdim) size(Wb,2)]); b=Wb(end,:);
xIdx= zeros(numel(szX),1); xIdx(rdim)=-rdim; xIdx(dim)=1:numel(dim);
wIdx= [-rdim numel(dim)+1];
f   = tprod(X,xIdx,W,wIdx);    % apply the weighting, [size(X,dim) x nSp]
f   = repop(f,'+',shiftdim(b(:),-numel(dim)));   % apply the bias
return;