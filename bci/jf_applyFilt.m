function [z]=jf_applyFilt(z,varargin);
% apply a fir/iir filter to the data
%
% Options:
%  dim        -- the dimension along which to apply the filter
%  filtFn     -- {str} {function_handle}  a function to 'filter' the data
%                   [f,state]=func(f,state)
%                    where state is the internal state of the filter, e.g. the history of past values
%                      Examples, using function handles:
%                        'filtFn','avefilt',10             % moving average filter, length 10
%                        'filtFn',@(x,s) avefilt(x,s,10)   % moving average filter, length 10
%                        'filtFn',@(x,s) biasFilt(x,s,50)  % bias adaptation filter, length 50
%                        'filtFn',@(x,s) stdFilt(x,s,100)  % normalising filter (0-mean,1-std-dev), length 100
%                        'filtFn',@(x,s) avenFilt(x,s,10)  % send average f every 10 predictions
%                        'filtFn',@(x,s) marginFilt(x,s,3) % send if margin between best and worst prediction >=3  
%  resetIdx  -- 'str' OR [int] set of points at which the reset the filter state
%  summary    -- additional summary description info
opts=struct('dim','time','filtFn',[],'summary','','verb',0,'subIdx',[],'resetIdx',[],'filtOpts',{{}});
[opts,varargin]=parseOpts(opts,varargin);
if ( isempty(opts.filtOpts) ) opts.filtOpts={}; end; % ensure is empty cell...
if ( ~iscell(opts.filtOpts) ) opts.filtOpts={opts.filtOpts}; end;

% sortcut identify filters
if ( isequal(opts.filtFn,1) || (ischar(opts.filtFn) && any(strcmpi(opts.filtFn,{'eye','Id'}))) )
  return;
end

dim=n2d(z,opts.dim,1,0); dim(dim==0)=[];

resetIdx=opts.resetIdx;
if ( ischar(resetIdx) && strncmpi(resetIdx,'block',numel('block')) ) % filter in blocks
  resetIdx=round(getBlockIdx(z,resetIdx)); % N.B. use round to ensure only last block info is used
  if( numel(opts.resetIdx)>numel('block') ) % single-block whitening
	 bi=str2num(opts.resetIdx(numel('block')+1:end)); % get the blockID to use
	 blockIdx= (blockIdx==bi); % mark everything but this block as excluded
  end
  % reset state when blockIdx changes
  resetIdx=find(resetIdx(1:end-1)~=resetIdx(2:end));
end


										  % move along indicated dimension
if ( opts.verb>=0 ) fprintf('applyFilt:'); end;
state=[];
X=z.X;
szX=size(z.X);
idx={}; for di=1:ndims(z.X); idx{di}=1:size(z.X,di); end; 
for ti=1:prod(szX(dim));
  if ( numel(dim)==1 ) 
	 idx{dim}=ti;	 
  else
	 [idx{dim}]=ind2sub(szX(dim),ti);
  end
  tX = X(idx{:});
  if ( any(ti==resetIdx) )
	 state=[];
  end;
  [fX,state]=feval(opts.filtFn,tX,state,opts.filtOpts{:},varargin{:});
  if ( ~isempty(fX) ) 
	 z.X(idx{:})=fX;
  else
	 z.X(idx{:})=0; % mark with 0 if filter says ignore...
  end
  if ( opts.verb>=0) textprogressbar(ti,prod(szX(dim))); end;
end
if ( opts.verb>=0 ) fprintf('\n'); end;

info=struct('state',state); % preserve final filter state
if ( ischar(opts.filtFn) ) summary=opts.filtFn; else summary=disp(opts.filtFn); end;
summary=sprintf('%s along %s %s',summary,sprintf('%s',z.di(dim).name),opts.summary);
if ( ~isempty(resetIdx) )
  summary=[summary sprintf('in %d blocks',numel(resetIdx))];
end
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%------------------------------------------
function testCase();
  z = jf_mksfToy('Y',ceil(rand(100,1)*2));oz=z;
  z = jf_addFolding(z,'outfIdxs',1:50); % 1st half training
  z.X(:,:,51:end)=z.X(:,:,51:end)*100+10; % cov-shift for 2nd half
  z = jf_cvtrain(z);
  jf_recompPerf(z);
zb=jf_applyFilt(z,'dim','epoch','filtFn','biasFilt',exp(log(.5)/10));
zb=jf_applyFilt(z,'dim','epoch','filtFn','stdFilt',exp(log(.5)/10));
zc=jf_recompPerf(zb);
jf_disp(zb);  
clf;plot([z.X zb.X],'linewidth',2)

										  % with filter reset at block boundary...

zb=jf_applyFilt(z,'dim','epoch','resetIdx',51,'filtFn','biasFilt',exp(log(.5)/10));
