function [FX]=fft_posfreq(X,len,dim,feat,taper,detrendp,centerp,corMag,MAXEL,verb)
% FFT computation, which only returns the *postive* frequency components
%
% [FX]=fft_posfreq(X,len,dim,taper,detrendp,centerp,corMag,MAXEL,verb)
%
% Inputs:
%  X    - n-d input matrix
%  len  - length of the maxtrix to use                (size(X,sampDim))
%  dim  - dimensions of X along which to compute fft. (1st non-singlenton) 
%         neg values count back from ndims(X)
%  feat - [str] type of feature to compute, one-of       ('complex')
%           'complex' - normal fourier coefficients
%           'l2','pow' - power, i.e. squared length of the complex coefficient
%           'abs'     - absolute length of the coefficients
%           'angle'   - angle of the complex coefficient
%           'real'    - real part
%           'imag'    - imaginary part
%           'db'      - power in decibles, ie. 10*log10(F(X).^2)
%  taper-  [size(X,dim) x1] time domain window to apply before the fft  ([])
%  detrendp - [bool] detrend before computing the fft? (0)
%  centerp  - [bool] center before computing the fft? (0)
%  corMag  - [2 x bool] correct power changes?
%             corMag(1) - correct for *sqrt(2) factor in 0Hz&fs/2Hz     (1)
%             corMag(2) - correct for *sqrt(size(X,dim)) factor due to non-unity fourier basis (0)
%  MAXEL- chunk size for chunking algorithm
%  verb - verbosity level
% Output:
%  FX   - Fourier transform of X for positive frequencies only
%
% N.B. to keep he power equal after and fft you must correct by #entries, i.e.
%   mad(X*X',fft(X,[],2)*fft(X,[],2)'./size(X,2)) == 0
%   fX=fft(X,[],2)./sqrt(size(X,2)); % pre-correct for # entries = re-scaling in fft
%   mad(X*X',real(fX*fX')) == 0 
% When using only the positive frequencies you should do the same
%   mad(X*X',real(fft_posfreq(X,[],2)*fft_posfreq(X,[],2)').*2./size(X,2)) == 0
%   fX=fft_posfreq(X,[],2).*sqrt(2./size(X,2)); % N.B. inc factor 2 for 1/2 spectrum
%   mad(X*X',real(fX*fX')) == 0
if ( nargin < 3 || isempty(dim) ) 
   dim = find(size(X)>1,1,'first'); if ( isempty(dim) ) dim=1; end;  
end;
if ( dim < 0 ) dim = ndims(X)-dim+1; end;
if ( nargin < 2 || isempty(len) ) len = size(X,dim); end;
if ( nargin < 4 || isempty(feat) ) feat='complex'; end;
if ( nargin < 5 ) taper=[]; end;
if ( nargin < 6 || isempty(detrendp) ) detrendp=false; end;
if ( nargin < 7 || isempty(centerp) ) centerp=false; end;
if ( nargin < 8 || isempty(corMag) ) corMag=true; end;
if ( nargin < 9 || isempty(MAXEL) ) MAXEL=2e6; end;
if ( nargin < 10 || isempty(verb) ) verb=0; end;
szX=size(X);
normp=false; if ( numel(corMag)>1 ) normp=corMag(2); corMag=corMag(1); end;

if ( len ~= size(X,dim) ) error('len isnt correctly implemented yet!'); end;
if ( ~isempty(taper) ) taper=shiftdim(taper,-dim+1); end;

switch lower(feat);
 case 'complex'; 
  FX = complex(zeros([szX(1:dim-1) ceil((szX(dim)-1)/2)+1 szX(dim+1:end)],class(X))); % pre-alloc
 case {'abs','real','imag','angle','pow','db','amp'};
  FX = zeros([szX(1:dim-1) ceil((szX(dim)-1)/2)+1 szX(dim+1:end)],class(X)); % pre-alloc
 otherwise;
  error('Unrec feature type to compute : %s',lower(feat) );
end

% N.B. norm of the fourier basis is sum(1.^2) = size(X,dim)
%nF=size(X,dim);if ( ~isempty(taper) ) nF=taper(:)'*taper(:); end;

[idx,allStrides,nchnks]=nextChunk([],size(X),dim,MAXEL);
ci=0; if ( verb >= 0 && nchnks>1 ) fprintf('FFT posfreq:'); end;
while ( ~isempty(idx) )

   tX = X(idx{:});
   if ( centerp )         tX=repop(tX,'-',mean(tX,dim)); end;
   if ( detrendp )        tX=detrend(tX,dim);     end;
   if ( ~isempty(taper) ) tX=repop(tX,'*',taper); end;

   tX=fft(tX,len,dim); % full FFT. N.B. inc pos & neg freq

   % Make some index expresssions
   FXidx=idx; FXidx{dim}=1:size(FX,dim); % to assign into result
   % to extract pos freq only
   tmpIdx={};for d=1:ndims(tX); tmpIdx{d}=1:size(tX,d); end; 
   tmpIdx{dim}=1:size(FX,dim);
   
   % Do the extraction
   tX=tX(tmpIdx{:});
	if ( normp ) tX=tX.*sqrt(2); end;
   if ( corMag ) % correct double mag of fs/2 entry
      if ( mod(szX(dim),2)==0 ) % Need to correct the fs/2 entry too
         corIdx=tmpIdx; corIdx{dim}=[1 size(FX,dim)];
      else % Only need to correct the 0Hz
         corIdx=tmpIdx; corIdx{dim}=1;
      end
      tX(corIdx{:})=tX(corIdx{:})/sqrt(2); % do the correction
   end
   switch lower(feat);
    case 'complex';     FX(FXidx{:}) = tX;
    case {'l2','pow'};  FX(FXidx{:}) = real(tX).^2 + imag(tX).^2;    
    case {'abs','amp'}; FX(FXidx{:}) = sqrt(real(tX).^2 + imag(tX).^2);
    case 'angle';       FX(FXidx{:}) = angle(tX);
    case 'real';        FX(FXidx{:}) = real(tX);
    case 'imag';        FX(FXidx{:}) = imag(tX);     
    case 'db';          tX=real(tX).^2 + imag(tX).^2; tX(tX==0)=eps; FX(FXidx{:}) = 10*log10(tX);
   end
        
   if( verb>=0 ) ci=ci+1; textprogressbar(ci,nchnks);  end
   idx=nextChunk(idx,size(X),allStrides);
end
if ( verb>=0 && nchnks>1 ) fprintf('done\n'); end;
return

%-----------------------------------------------------------------------
function testCase()
nCh=2; nSamp=100; N=10;
X=randn(nCh,nSamp,N);
FXp=fft_posfreq(X,[],2);

nCh=2; nSamp=100; N=10;
X=single(randn(nCh,nSamp,N));
FXp=fft_posfreq(X,[],2);

% Using the positive_freq only
mimage(tprod(X,[1 -2 3],conj(X),[2 -2 3]),...
       2*squeeze(tprod(real(FXp),[1 -3 4],[],[2 -3 4])+...
                 tprod(imag(FXp),[1 -3 4],[],[2 -3 4]))/nSamp,...
       'diff',1); % 1st Half

mimage(tprod(X,[1 -2 3],conj(X),[2 -2 3]),...
       2*squeeze(sum(tprod(real(FXp),[1 3 4],[],[2 3 4])+...
                     tprod(imag(FXp),[1 3 4],[],[2 3 4]),3))/nSamp,...
       'diff',1); % late summ + 1st Half


%------------------------------------------------------------------------
% Development test code
FX=fft(X,[],2);

clf;mimage(real(fft(X,[],2)*fft(X,[],2)'),real(fft_posfreq(X,[],2)*fft_posfreq(X,[],2)')*2,'diff',1,'colorbar',1,'clim',[])

X =randn(10,1000);
C =(X*X')./size(X,2);
fX=fft_posfreq(X,[],2)./sqrt(size(X,2)); % inc-normalization factor to keep the power equal
fX=fX.*sqrt(size(fX,2)./(floor(size(fX,2)/2)*2)); %N.B. extra correction for inclusion of the 0-hz bin
fC=(real(fX)*real(fX)'+imag(fX)*imag(fX)')./(size(fX,2));
clf;mimage(C,fC,'diff',1,'clim',[],'colorbar',1)

% Check it allows the correct computation of the covariance matrices
% Direct Computation
mimage(X(:,:,1)*X(:,:,1)',real(FX(:,:,1)*FX(:,:,1)')/nSamp,'diff',1);% fourier
mimage(X(:,:,1)*X(:,:,1)',...
       (real(FX(:,:,1))*real(FX(:,:,1)')+...
        imag(FX(:,:,1))*imag(FX(:,:,1))')/nSamp,'diff',1); % fourier in bits
mimage(X(:,:,1)*X(:,:,1)',...
       (FX(:,1,1)*FX(:,1,1)'+...
        2*(real(FX(:,2:ceil((end-1)/2),1)*real(FX(:,2:ceil((end-1)/2),1)'))+...
           imag(FX(:,2:ceil((end-1)/2),1))*imag(FX(:,2:ceil((end-1)/2),1))')+...
        FX(:,ceil((end-1)/2)+1,1)*FX(:,ceil((end-1)/2)+1,1)')/100,'diff',1); % 1st half only

mimage(X(:,:,1)*X(:,:,1)',... % late summation
       sum(real(tprod(FX(:,:,1),[1 3],conj(FX(:,:,1)),[2 3])),3)/100,'diff',1)

mimage(X(:,:,1)*X(:,:,1)',...
       tprod((tprod(real(FX(:,1:ceil((end-1)/2)+1,1)),[1 3],[],[2 3])+...
             tprod(imag(FX(:,1:ceil((end-1)/2)+1,1)),[1 3],[],[2 3])),[1 2 -3],...
    [1;2*ones(ceil((size(FX,2)-1)/2)-1,1);1],[-3])/100,'diff',1); % late summ + 1st Half


% Tprod computation
mimage(tprod(X(:,:,:),[1 -2 3],[],[2 -2 3]),...
       real(tprod(FX(:,:,:),[1 -2 3],conj(FX(:,:,:)),[2 -2 3]))/100,'diff',1);%fourier
mimage(tprod(X(:,:,:),[1 -2 3],[],[2 -2 3]),...
       (tprod(real(FX(:,:,:)),[1 -2 3],[],[2 -2 3])+...
        tprod(imag(FX(:,:,:)),[1 -2 3],[],[2 -2 3]))/100,'diff',1);%fourier

mimage(tprod(X(:,:,:),[1 -2 3],[],[2 -2 3]),...
       (tprod(FX(:,1,:),[1 -2 3],conj(FX(:,1,:)),[2 -2 3],'n')+...
        2*(tprod(real(FX(:,2:ceil((end-1)/2),:)),[1 -2 3],[],[2 -2 3])+... 
           tprod(imag(FX(:,2:ceil((end-1)/2),:)),[1 -2 3],[],[2 -2 3]))+...
        tprod(FX(:,ceil((end-1)/2)+1,:),[1 -2 3],conj(FX(:,ceil((end-1)/2)+1,:)),[2 -2 3],'n'))/nSamp,'diff',1); % 1st half only

mimage(tprod(X,[1 -2 3],conj(X),[2 -2 3]),... % late summation
       squeeze(sum(real(tprod(FX(:,:,:),[1 3 4],conj(FX(:,:,:)),[2 3 4])),3))/nSamp,'diff',1)

mimage(tprod(X,[1 -2 3],conj(X),[2 -2 3]),...
       tprod((tprod(real(FX(:,1:ceil((end-1)/2)+1,:)),[1 3 4],[],[2 3 4])+...
             tprod(imag(FX(:,1:ceil((end-1)/2)+1,:)),[1 3 4],[],[2 3 4])),[1 2 -3 3],...
    [1;2*ones(ceil((size(FX,2)-1)/2)-1,1);1],[-3])/nSamp,'diff',1); % late summ + 1st Half
