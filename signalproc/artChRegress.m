function [X,info]=artChRegress(X,dim,idx,varargin);
% remove any signal correlated with the input signals from the data
% 
%   [X,info]=artChRm(X,dim,idx,varargin)
%
% Inputs:
%  X   -- [n-d] the data to be deflated/art channel removed
%  dim -- dim(1) = the dimension along which to correlate/deflate ([1 2])
%         dim(2) = the time dimension for spectral filtering/detrending along
%         dim(3) = compute regression separately for each element on this dimension
%  idx -- the index/indicies along dim(1) to use as artifact channels ([])
% Options:
%  bands   -- spectral filter (as for fftfilter) to apply to the artifact signal ([])
%  fs  -- the sampling rate of this data for the time dimension ([])
%         N.B. fs is only needed if you are using a spectral filter, i.e. bands.
%  detrend -- detrend the artifact before removal                        (1)
%  center  -- center in time (0-mean) the artifact signal before removal (0)
opts=struct('detrend',0,'center',0,'bands',[],'fs',[],'maxIter',0,'tol',1e-2,'verb',0);
[opts]=parseOpts(opts,varargin);

dim(dim<0)=ndims(X)+1+dim(dim<0);

% compute the artifact signal and its forward propogation to the other channels
if ( ~isempty(opts.bands) ) % smoothing filter applied to art-sig before we use it
  artFilt = mkFilter(floor(size(X,dim(end))./2),opts.bands,opts.fs/size(X,dim(end)));
else
  artFilt=[];
end

% make a index expression to extract the artifact channels
artIdx=cell(ndims(X),1); for d=1:ndims(X); artIdx{d}=1:size(X,d); end; artIdx{dim(1)}=idx;
% Iteratively refine the artifact signal by propogating to the rest of the electrodes and back again
tpIdx  = -(1:ndims(X)); tpIdx(dim(1)) =1; if( numel(dim)>2)  tpIdx(dim(3))=3; end;
tpIdx2 = -(1:ndims(X)); tpIdx2(dim(1))=2; if( numel(dim)>2) tpIdx2(dim(3))=3; end;
sf=[];

artSig = X(artIdx{:});
if ( opts.center )       artSig = repop(artSig,'-',mean(artSig,dim(2))); end;
if ( opts.detrend )      artSig = detrend(artSig,dim(2)); end;
if ( ~isempty(artFilt) ) artSig = fftfilter(artSig,artFilt,[],dim(2),1); end % smooth the result
artCov = tprod(artSig,tpIdx,[],tpIdx2); % cov of the artifact signal: [nArt x nArt x nEp]
iartCov= zeros(size(artCov)); % [nArt x nArt x nEp]
for epi=1:size(artCov(:,:,:),3); iartCov(:,:,epi)= pinv(artCov(:,:,epi)); 
end 
artXcov= tprod(artSig,tpIdx,X,tpIdx2); % [nArt x nCh x nEp]
artW   = tprod(iartCov,[-1 1 3],artXcov,[-1 2 3]); % weighting over artCh for each Xch  [nArt x nCh x nEp]

% make a single spatial filter to remove the artifact signal in 1 step
%  X-w*X = X*(I-w)
sf     = eye(size(X,dim(1))); % [nCh x nCh x nEp]
if( numel(dim)>2 ) sf=repmat(sf,[1 1 size(X,dim(3))]); end;
sf(idx,:,:)=sf(idx,:,:)-artW; % [ nArt x nCh x nEp]

                                % apply the deflation
if( numel(dim)==2 ) % global regression mode
  X = tprod(sf,[-dim(1) dim(1)],X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(X)]);
else % per-epoch mode
  tpIdx=1:ndims(X); tpIdx(dim(1))=-dim(1); tpIdx(dim(3))=dim(3);
  X = tprod(sf,[-dim(1) dim(1) dim(3)],X,tpIdx);
end

info = struct('artSig',artSig,'artFilt',artFilt,'sf',sf,'artCov',artCov,'artW',artW,'artIdx',{artIdx});
return;
%--------------------------------------------------------------------------
function testCase()
S=randn(10,1000);% sources
sf=randn(10,2);% per-electrode spatial filter
X =S+sf*S(1:size(sf,2),:); % source+propogaged noise
Y =artChRegress(X,1,[1 2]);
clf;mimage(S,Y-S,'clim','cent0','colorbar',1); colormap ikelvin
Y2=artChRm(X,1,[1 2]);
clf;mimage(S,Y2-S,'clim','cent0','colorbar',1); colormap ikelvin

                                % regress per-epoch
X3d=cat(3,X,X,X);
[Y,info] =artChRegress(X3d,[1 2 3],[1 2]);

