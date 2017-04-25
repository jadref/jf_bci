function [x2,idx]=butterdownsample(x,N,dim,forder,MAXEL)
% Simple function to sub-sample a time series using an bin-averaging technique
% [sx,idx]=subsample(x,N,dim)
% Note: we assume 
%  a) that input values are average over the *preceeding* 1 unit bin
%  b) output values are then the average over the preceeding bin width
%
% [x2,idx]=subsample(x,N[,dim,centbins,MAXEL])
% Inputs:
%  x    -- n-d matrix to sub-sample
%  N    -- output size along the dimension we want to sub-sample
%  dim  -- the dimension of x to subsample along
%  forder- [int] order of butterworth filter to use (6)
%  MAXEL-- max number of elements to process at a time for memory usage
% Outputs:
%  x2   -- the subsampled x, [ size(X) with size(x2,dim)==N ]
%  idx  -- position of the new bin center in the input linear index. 
%          N.B. this is not necessarially an integer!
if ( nargin < 3 || isempty(dim) ) dim=find(size(x)>1,1,'first'); end;
if ( dim < 0 ) dim=ndims(x)+dim+1; end;
if ( nargin < 4 || isempty(centbins) ) centbins=0; end;
if ( nargin < 5 || isempty(forder) ) forder=6; end;
if ( nargin < 6 || isempty(MAXEL) ) MAXEL=2e6; end;

if ( isscalar(N) )
   wdth = (size(x,dim))/N;
   idx  = wdth:wdth:size(x,dim);
   %idx  = linspace(0,size(x,dim),N+1); idx(1)=[];
   %wdth = (size(x,dim))/N; % width each bin
else % explicit locations -- N.B. *must* be equal spaced!
   wdth=diff(N);
   if(any(abs(diff(wdth))>1e-3)) 
      warning('sample boundarys not equally spaced!');
   else
      wdth=wdth(1);
   end
   idx  = N; centbins=0;
end

if ( wdth<= 1 ) x2=x; idx=1:size(x,dim); return; end;
if ( wdth< 1.05) warning('Butter is unstable at this low a decimation factor'); end;

% make the filter to spectrally filter
[B,A]=butter(forder,1./wdth,'low'); 

szx = size(x);
x2 = zeros([szx(1:dim-1) numel(idx) szx(dim+1:end)]); % pre-alloc
[ckidx,allStrides]=nextChunk([],size(x),dim,MAXEL);
while ( ~isempty(ckidx) ) 
   
   x2idx=ckidx; x2idx{dim}=1:size(x2,dim); % assign index      
   [x2(x2idx{:})]=buttersubsamp(x(ckidx{:}),idx,dim,B,A);
   
   ckidx=nextChunk(ckidx,size(x),allStrides);
end

if ( centbins ) idx=idx-wdth/2+.5; end;

return;

%---------------------------------------------------------------------------
% Inner fucntion to do the actual work
function [x2,idx]=buttersubsamp(x,idx,dim,B,A)
szx = size(x); 
for i=1:numel(szx); subs{i}=1:szx(i); end; 
subs{dim}=ceil(idx); % make indices

% remove constant terms to stop rounding issues
mu   = mean(x,dim);
x    = repop(x,'-',mu);

% Compute the whole sample contributions, i.e. assume bins end on sample bounds
x2   = filter(B,A,x,[],dim);
x2   = x2(subs{:});

% undo the centering
x2   = repop(x2,'+',mu);
return

   
%---------------------------------------------------------------------------
function testCase()
X=randn(2,100,4);X=cumsum(X,2); X=single(X);
clf;plot(1:size(X,2),X(:,:,1)','LineWidth',3);hold on;
[T,idx]=subsample_butter(X,10,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,15,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,20,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,40,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,50,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,60,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,80,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,90,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,99,2);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,100,2);plot(idx,T(:,:,1)',linecol);

% Test with chunking
[T,idx]=subsample_butter(X,10,2,[],200);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,15,2,[],200);plot(idx,T(:,:,1)',linecol);
[T,idx]=subsample_butter(X,20,2,[],200);plot(idx,T(:,:,1)',linecol);

% test with large offsets to cause rounding errors
X=randn(2,1000,4);X=cumsum(X,2); X=X+1e6; X=single(X);
clf;plot(1:size(X,2),X(:,:,1)','LineWidth',3);hold on;
[T,idx]=subsample_butter(X,250,2);plot(idx,T(:,:,1)',linecol);
