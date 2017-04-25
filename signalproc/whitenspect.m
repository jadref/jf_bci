function [X,muSpect,smth]=whitenspect(X,dim,smth,centerp,muReg,muCh,chD,verb)
if ( nargin<4 || isempty(centerp) ) centerp=1; end;
if ( nargin<5 || isempty(muReg) ) muReg=-.01; end;
if ( nargin<6 ) muCh=[]; end;
if ( nargin<7 ) chD=[]; end;
if ( nargin<8 || isempty(verb) ) verb=1; end;
szX=size(X); szX(end+1:max(dim))=1;
timeD=dim(1);
accDims=setdiff(1:numel(szX),dim);

% compute mean spectrum over channels/epochs
% N.B. we do it ourselves rather than via e.g. welch, to save RAM as we don't need the per-xxx values
% compute average spectrum over all channels/windows/time-points etc..

% loop over the data computing the grand average spectrum
[idx,chkStrides,nchnks]=nextChunk([],szX,[timeD;chD]);
ci=0; if ( verb >= 0 && nchnks>1 ) fprintf('%s:',mfilename); end;
muSpect=[];
while ( ~isempty(idx) )
   muIdx   = idx;
   if ( ~isempty(muCh) ) muIdx{chD}=muCh; end % select sub-set of channels of wanted
   Fx      = abs(fft(X(muIdx{:}),[],timeD));
   Fx      = msum(Fx,accDims);
   if ( isempty(muSpect) ) muSpect = Fx; else  muSpect = muSpect + Fx; end;    
   if ( verb>=0 && nchnks>1 ) textprogressbar(ci,nchnks); end;
   idx=nextChunk(idx,szX,chkStrides);
end
if ( verb>=0 && nchnks>1 ) fprintf('\n'); end;
muSpect = muSpect./prod(szX);

if ( centerp ) % remove the 0-hz
  idx={}; for di=1:ndims(muSpect); idx{di}=1:size(muSpect,di); end;
  idx{timeD}=1;
  muSpect(idx{:})=0;
end

%smooth muSpect
if ( ~isempty(smth) )
  if ( numel(smth)==1 ) % use a gaussian smoother of the given length, N.B. pass band is 1/2 this size!
	 if ( smth<0 ) smth = szX(timeD)*abs(smth); end; % fractional length 
    smth=floor(smth/2)*2+1; % ensure odd length and integer
	 smth=mkFilter(smth,'hanning');
  end
  smth=smth./sum(smth);
  smthMx=spdiags(repmat(smth(:)',size(muSpect,timeD),1),-floor(numel(smth)/2):floor(numel(smth)/2),...
                 size(muSpect,timeD),size(muSpect,timeD));
  % fix the startup effects
  smthMx=full(smthMx); smthMx=repop(smthMx,'/',sum(smthMx));
	% shift 0-hz to the middle so that smoothing works correctly
	muSpect=fftshift(muSpect,timeD);
   muSpect=tprod(muSpect,[1:timeD-1 -timeD timeD+1:ndims(muSpect)],full(smthMx),[-timeD timeD]);
	muSpect=ifftshift(muSpect,timeD);
end

% compute the normalising filter
normFilt=muSpect;
if ( muReg<0 )
  if ( muReg>-1 ) % fraction of max-spect-power
	 muReg=abs(muReg)*max(abs(muSpect(:)));
  else
	 ss=mmean(muReg,[1:timeD-1 timeD+1:ndims(muSpect)]);
	 ss=sort(abs(ss),'descend');
	 muReg=ss(min(abs(muReg),end));
  end
end;
if ( muReg>0 ) normFilt = normFilt+muReg; end % include regularizor to prevent to large magnification
normFilt(normFilt==0)=1; % stop divide by 0
normFilt = 1./muSpect; % invert the spectrum
%normFilt = repop(normFilt,'./',sum(normFilt,timeD)); % scale to unit total area

% use fftfilter to actually apply the spectral normalization filter
if ( numel(dim)==1 )
  X = fftfilter(X,normFilt,[],timeD,0,[],0);
elseif( numel(dim)==2 ) % apply for each element of dim(2)
  idx={}; filtidx={};
  for di=1:numel(szX);
	 idx{di}=1:szX(di);
	 filtidx{di}=1:size(normFilt,di);
  end;
	% loop over elements in this dim and applying the element specific filter
  for ei=1:szX(dim(2));
	 idx{dim(2)}=ei; filtidx{dim(2)}=ei;
	 Xei=X(idx{:});
	 Xei=fftfilter(Xei,normFilt(filtidx{:}),[],timeD,0,[],0);
	 X(idx{:})=Xei;
  end
else
  error('not supported yet');
end;

return;

								 %--------------------------------------------------
function testcase()
X  =cumsum(randn(10,100,20));
clf;
subplot(311);cla;plot(mean(welchpsd(X,2,'width_samp',100,'overlap',.5),3)','linewidth',1)

Xw =whitenspect(X,2,10);
subplot(312);cla;plot(mean(welchpsd(Xw,2,'width_samp',100,'overlap',.5),3)','linewidth',1)

Xcw=whitenspect(X,[2 1],10);
subplot(313);cla;plot(mean(welchpsd(Xcw,2,'width_samp',50,'overlap',.5),3)','linewidth',1)
