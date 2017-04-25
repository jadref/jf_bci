function [z,opts]=jf_reref(z,varargin)
% re-reference with a weighted mean over a subset of indicies
% Options:
%  dim -- dimension(s) to be re-referenced ('ch')
%  idx -- which elements along 'dim' to included in and re-referenced (1:size(z.X,dim))
%  wght-- weighting for the included elements, average over idx if empty. ([])
%         OR 
%          'robust' to use a robust median estimate
%         OR
%          [2/3/4x1] spec of which points to use for the base-line computation in format 
%                    as used for mkFilter, 
%                    e.g. [-100 -400] means use a smooth *gaussian* window from -100ms to -400ms
%         OR
%          {X x 1} cell array of values along dimension dim to match.  The average of these elements
%                  will be used as the reference value
%  robust-- [bool] flag if use robust estimate
%  op  -- [str] operator to us for the referencing ('-')
%  smoothFactor - [float] factor to use for exp smoothing - 0=no-smoothing
%  smoothDim    - [int] dimension(s) along which to smooth
%  summary -- additional summary string
%  subIdx  -- {cell} cell array of arguments to pass to jf_retain to specify the sub-set of X on 
%               which the re-reference will be performed. e.g. to only rereference in time the eeg channels use: 
%                 jf_reref(z,'dim','time',subIdx,{'dim','ch','idx',[z.di(1).extra.iseeg]})
%
% Examples:
%   z=jf_reref(z,'dim','ch');   % re-ref to common average reference
%   z=jf_reref(z,'dim','time'); % center in time, i.e. remove DC offest
%   z=jf_reref(z,'dim','ch','wght',{'MASTL' 'MASTR'}); % switch to linked mastiods reference
%   z=jf_reref(z,'dim','time','wght',[-400 -100]); % basline in time to average of -400 -> -100 ms
%   z=jf_reref(z,'dim','time','wght','robust'); % robust basline in time to median value
opts=struct('wght',[],'op','-','robust',0,...
            'summary','','smoothFactor',[],'smoothDim',[],'subIdx',[],'verb',0,'blockIdx',[]);
subsrefOpts=struct('dim','ch','vals',[],'idx',[],...
                   'range',[],'mode','retain','valmatch','exact');
[opts,subsrefOpts,varargin]=parseOpts({opts,subsrefOpts},varargin);

idx=subsrefDimInfo(z.di,subsrefOpts);
dim = n2d(z.di,subsrefOpts.dim);
if ( ~isempty(opts.subIdx) ) % sub-idx over-rides spec of elements to work on
  idx=subsrefDimInfo(z.di,opts.subIdx);
end
if( ~isempty(subsrefOpts.idx) ) % don't use index if it's not needed
   if(( islogical(subsrefOpts.idx) && all(subsrefOpts.idx) ) || ...
      ( isnumeric(subsrefOpts.idx) && isequal(subsrefOpts.idx,1:size(z.X,dim))) ) 
      subsrefOpts=[]; 
   end
end

wght = opts.wght;
if ( iscell(wght) || (~isempty(opts.wght) && numel(wght)<=4) )
  wght=mkFilter(size(z.X,dim),wght,z.di(dim).vals);
  wght=wght./sum(wght); % make it compute an average
elseif ( strcmp(opts.wght,'robust') ) 
   opts.robust=true; wght=[];
end

% compute the thing to subtract (multi-dim aware)
X=z.X; if ( ~isempty(subsrefOpts.idx) || ~isempty(subsrefOpts.vals) ) X = z.X(idx{:}); end;

mu=X; 
if ( any(isnan(mu(:))) ) mu(isnan(mu(:)))=mean(mu(~isnan(mu(:)))); end; % remove NaN's from mu

if ( isempty(wght) ) % simple average reference
   if( ~opts.robust ) % normal mean computation
      N=1;
      for di=1:numel(dim); N=N*size(mu,dim(di)); mu=sum(mu,dim(di));  end; 
      mu=mu./N;
   else % robust center computation
      mu= robustCenter(mu,dim); % do a robust centering
   end
else
  if( numel(wght)~=numel(idx{dim}) ) error('Wght should match num ref pos'); end;
  if( ~opts.robust ) % normal average estimate based on the weighted mean
     wIdx=[1:ndims(z.X)]; wIdx(dim)=-wIdx(dim); mu=tprod(z.X(idx{:}),wIdx,wght,-dim);
  else % weighted robust estimate....
       %BODGE: use the wght>0 to sub-set the data one more time...
     wIdx=idx; wIdx{dim}=(wght>eps); mu=robustCenter(z.X(idx{:}),dim); % do a robust centering
  end
end

% smoothing of the base-line over other dimensions
mu=expSmooth(mu,n2d(z,opts.smoothDim,0,0),opts.smoothFactor);

% do the re-ref, on the selected elements
if ( ~isempty(subsrefOpts.idx) || ~isempty(subsrefOpts.vals) )
   z.X(idx{:})=repop(X,opts.op,mu);
else
   z.X        =repop(X,opts.op,mu);
end

% update the meta-info
summary = sprintf('over %s',z.di(dim(1)).name);
if( numel(dim)>1 ) summary = [summary sprintf('+%s',z.di(dim(2:end)).name)]; end;
if( numel(idx{dim(1)})<5 ) 
   summary = sprintf('%s (%s)',summary,vec2str(z.di(dim(1)).vals(idx{dim(1)}),'+'));
end;
if(ischar(opts.wght)) summary = [summary ' ' opts.wght]; 
elseif( isnumeric(wght) ) summary=['weighted ' summary];
end;
if(~isempty(opts.summary)) summary=[summary  sprintf(' (%s)',opts.summary)];end
info.mu=mu;
if( ~isempty(wght) ) info.wght=wght; end;
z=jf_addprep(z,mfilename,summary,mergeStruct(opts,subsrefOpts),info);
return;
%--------------------------------------------------------------------------
function mu=expSmooth(mu,smoothDim,smoothFactor)
% expionentially smooth the input
if ( isempty(smoothFactor) || smoothDim==0 ) return; end;
idx={}; for d=1:ndims(mu); idx{d}=1:size(mu,d); end;     
idx{smoothDim}=1;
musi = mu(idx{:});
for si=2:size(mu,smoothDim);
  idx{smoothDim}=si;
  mu(idx{:}) = (smoothFactor)*musi + (1-smoothFactor)*mu(idx{:});
end
return;

%--------------------------------------------------------------------------
function testCase()
z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_opt');
z=jf_load('external_data/mlsp2010/p300-comp','s1','trn');
z=jf_mksftoy();

%z=jf_retain(z,'dim','time','idx',1:600);
zmu=jf_reref(z,'dim','ch','idx',[z.di(1).extra.iseeg]);
zrmu=jf_reref(z,'dim','ch','idx',[z.di(1).extra.iseeg],'wght','robust');
figure(1);clf;jf_plotEEG(z,'subIdx',{[] [] 1});    
figure(2);clf;jf_plotEEG(zmu,'subIdx',{[] [] 1});  saveaspdf('~/car');
figure(3);clf;jf_plotEEG(zrmu,'subIdx',{[] [] 1}); saveaspdf('~/rcar');

figure(1);clf;jf_plotERP(z,'subIdx',{[1:27 29:56 58:64] [] []});   saveaspdf('~/raw_ERP');
figure(2);clf;jf_plotERP(zmu,'subIdx',{[1:27 29:56 58:64] [] []}); saveaspdf('~/car_ERP');
figure(3);clf;jf_plotERP(zrmu,'subIdx',{[1:27 29:56 58:64] [] []});saveaspdf('~/rcar_ERP');

zz=jf_reref(z,'dim','ch','smoothDim','epoch','smoothFactor',.1);
