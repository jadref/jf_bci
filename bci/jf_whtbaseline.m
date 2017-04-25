function [z,opts]=jf_reref(z,varargin)
% re-reference with a weighted mean over a subset of indicies
% Options:
%  dim -- dimension(s) to be re-referenced ('ch')
%  idx -- which elements along 'dim' to included in and re-referenced (1:size(z.X,dim))
%  wght-- weighting for the included elements, average over idx if empty. ([])
%         OR
%          [2/3/4x1] spec of which points to use for the base-line computation in format 
%                    as used for mkFilter, 
%                    e.g. [-100 -400] means use a smooth *gaussian* window from -100ms to -400ms
%  op  -- [str] operator to us for the referencing ('-')
%  smoothFactor - [float] factor to use for exp smoothing - 0=no-smoothing
opts=struct('dim',{{'ch' 'time'}},'wght',[],'smoothFactor',[],'smoothHalfLife',[],'tol',1e-7,'verb',1,'subIdx',[],'verb',0);
[opts,varargin]=parseOpts(opts,varargin);

dim=n2d(z,opts.dim); if( isempty(dim) ) dim=[1 2]; end;

wght = opts.wght;
if ( iscell(wght) || (~isempty(opts.wght) && numel(wght)<=4) )
  wght=mkFilter(size(z.X,dim(2)),wght,z.di(dim(2)).vals);
  wght=wght./sum(wght); % make it compute an average
  widx=find(wght>0); % indicies of used elements
elseif ( isempty(wght) )
  widx=1:size(z.X,dim); % indicies of used elements
end

smoothFactor=opts.smoothFactor;
if ( isempty(opts.smoothFactor) && ~isempty(opts.smoothHalfLife) )
  smoothFactor = exp(log(.5)./opts.smoothHalfLife);
end

% loop over epochs, computing the whitener and applying it
szX=size(z.X);
nEp=prod(szX(setdiff(1:end,dim)));
if ( ~isequal(dim(:),[1;2]) && ~isequal(dim(:),[2;1]) ) error('only for space x time x other at the moment'); end;
% cov computation idx
covidx=1:ndims(z.X); covidx(dim(2))=-dim(2);
covidx2=covidx;        covidx2(dim(1))=dim(2); % time -> space2
cov=zeros(size(z.X,dim(1)));
if ( opts.verb>0 ) fprintf('whtbase:'); end;
for ei=1:nEp;
  Xei=z.X(:,:,ei);
  covei=tprod(Xei,covidx,[],covidx2); % apply Lediot?
  % apply expionential smoothing
  if ( ei==1 ) 
    cov = covei;
  else
    cov = opts.smoothFactor*cov + (1-opts.smoothFactor)*covei;
  end
  % compute the symetric whitener
  [U,D]= eig(cov); D=diag(D);
  si=~(isinf(D) | isnan(D) | imag(D)>0 | D<=opts.tol );
  W=repop(U(:,si),'/',sqrt(D(si))')*U(:,si)'; % symetric whitener
  % apply to the data
  z.X(:,:,ei) = tprod(Xei,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(z.X)],W,[-dim(1) dim(1)]);
  if ( opts.verb>0 ) textprogressbar(ei,nEp); end;
end
if ( opts.verb>0 ) fprintf('\n'); end;

% update the meta-info
summary = sprintf('over %s',z.di(dim(1)).name);
if( numel(dim)>1 ) summary = [summary sprintf('+%s',z.di(dim(2:end)).name)]; end;
info=[];
if( ~isempty(wght) ) info.wght=wght; end;
z=jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_opt');
z=jf_load('external_data/mlsp2010/p300-comp','s1','trn');
z=jf_mksftoy();

jf_whtbaseline(z,'dim',{'ch' 'time'},'wght',[0 1000],'smoothFactor',exp(log(.5)./10))
