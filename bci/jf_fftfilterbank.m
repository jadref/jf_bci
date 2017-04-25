function [z]=jf_fftfilterbank(z,varargin);
% Spectrally filter (+up/down sample) with a set of filters using the fft technique
%
% Options:
%  dim        -- the time dimension
%  bands      -- {1 x nf} set of nf filter bands to make filters for
%              OR
%                [2 x nf] set of nf filter bands specified as [start stop]
%              OR
%                [4 x nf] set of nf filter bands specified as [lowcut lowpass highpass highcut]
%              OR
%                [1 x nf] set of nf filter bands specified as center-frequencies bands(i) and
%                         width is mean(diff(bands))
%  normfilt   -- [bool] normalize the set of filters to have unit norms = preserve power of the inputs
%  len        -- output size (use to up/downsample at the same time) (size(z.X,dim))
%  fs         -- output sampling rate (use to up/downsample)              ([])
%  detrend    -- [int] flag if we linearly deTrend before computing the ffts    (1)
%               0=no, 1=detrend, 2=center
%  win        -- time domain window to apply before filtering             ([])
%                e.g. 'hanning', 'hamming', [low-cut low-pass hi-pass hi-cut]
%                SEEALSO: mkFilter for types of win available
%  hilbert    -- [int] do a hilbert transform? (0)
%              0=non-hilbert, 1=hilbert with amp only, 2=hilbert with phase only, 3=hilbert with amp+phase
%  summary    -- additional summary description info
opts=struct('dim','time','len',[],'fs',[],'bands',[],'detrend',1,'win',[],'hilbert',0,'normfilt',1,'summary',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

% get dim to work along
dim=n2d(z.di,opts.dim);
% extract the sampling rate
fs=getSampRate(z);
nSamp=size(z.X,dim);

if(~isempty(opts.len) )  len=opts.len; if(any(len<0)) len=round(nSamp*abs(len)); end;
else                     len=nSamp;
end
if( ~isempty(opts.fs) )  len(2)=round(nSamp*opts.fs/fs); % downsample
elseif ( numel(len)<2 )  len(2)=nSamp; 
end;
if ( len(2) > nSamp )
   warning('Upsampling not implemented yet! ignored');
   len(2)=nSamp;
end

bands=opts.bands; if ( isempty(bands) ) bands = [0 fs]; end;

% temporal window to apply before computing the fft
win=opts.win;
if ( ~isempty(win) && (ischar(win) || iscell(win) || (numel(win)<=5 && numel(win)~=size(z.X,dim))) ) 
  win=mkFilter(size(z.X,dim),opts.win,z.di(dim).vals);
end

% compute the set of filters to apply
if ( iscell(bands) ) % cell-array of filter specs
  filt=[];
  for fi=1:numel(bands);
	 filti= mkFilter(floor(len(1)/2),bands{fi},fs/len(1));
	 if ( isempty(filt) ) filt=filti; else filt(:,fi)=filti; end;
  end
elseif ( isnumeric(bands) )  
  if ( size(bands,1)==1 ) % center-freq -> lo-high cut
	 bands = [bands-.5*mean(diff(bands)); bands+.5*mean(diff(bands))];
  end
  filt=[];
  for fi=1:size(bands,2);
	 filti = mkFilter(floor(len(1)/2),bands(:,fi),fs/len(1));
	 if ( isempty(filt) ) filt=filti; else filt(:,fi)=filti; end;
  end
else
  error('unrecog filter bank spec');
end
% unit total filter weight in each frequency band to preserve total power
if ( opts.normfilt )
  normfilt=sqrt(sum(filt.*filt,2));
  filt(normfilt>0)=repop(filt(normfilt>0),'./',normfilt(normfilt>0));
end

% do the actual filtering
if ( any(filt~=1) ) % don't bother if not asked to do anything
  z.X = fftfilter(z.X,filt,len,dim,opts.detrend,win,opts.hilbert,[],opts.verb);
end
% compute the acutal output sampling rate, may be different from what was requested
len  =size(z.X,dim);
outfs=fs*len./nSamp;


% recording the updating info
summary=sprintf(' %d filters',size(filt,2));
if ( outfs~=fs ) 
   z.di(dim).vals = single(z.di(dim).vals(1)):(1000./outfs):single(z.di(dim).vals(end));
   z.di(dim).info.fs = outfs;
   sidx = round((1:numel(z.di(dim).vals))*fs./outfs); % nearest old point to the new one
   if ( numel(z.di(dim).extra)>0 ) z.di(dim).extra = z.di(dim).extra(sidx); end;
   summary=sprintf('%s + resample %3.1fHz->%3.1fHz',summary,fs,outfs); 
end;
% add the filter-bank dimension if needed
if ( size(filt,2)>1 )
  z.di=[z.di(1:dim); mkDimInfo(size(z.X,dim+1),1,'freq_cent','hz'); z.di(dim+1:end)];
  for fi=1:size(filt,2); % save the filter info
	 z.di(dim+1).extra(fi).filt=filt(:,fi);
	 z.di(dim+1).vals(fi)      =linspace(0,fs/2,size(filt,1))*filt(:,fi)./sum(filt(:,fi)); %weighted-mean-freq = center-freq
	 if ( iscell(bands) )
		z.di(dim+1).extra(fi).bands=bands{fi};
	 elseif ( isnumeric(bands) )
		z.di(dim+1).extra(fi).bands=bands(:,fi);
		%z.di(dim+1).vals(fi) =mean(bands(:,fi));
	 end;
  end
end

if ( ~isempty(opts.summary) ) summary=sprintf('%s(%s)',summary,opts.summary); end;
switch (opts.hilbert);
 case 1; summary=sprintf('%s hilbert-amplitude',summary); 
 case 2; summary=sprintf('%s hilbert-phase',summary); z.di(end).units='rad'; 
 case 3; summary=sprintf('%s hilbert',summary);
 otherwise;
end
if ( opts.normfilt ) summary=sprintf('%s (norm)',summary); end;
info=struct('filt',filt);
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function []=testCase()

% make signal with power at 100Hz
oz = jf_import('test','test','test',shiftdim(mkSig(2000,'sin',10),-1),[],'fs',1000); z=oz;
z=oz;
z = jf_fftfilterbank(oz,'bands',{[10 20] [20 30] [40 60] [60 140]}); % explicit
z = jf_fftfilterbank(oz,'bands', [10 20; 20 30; 40 60; 60 140]'); % lo-hi
z = jf_fftfilterbank(oz,'bands', [10:10:100]); % centers


clf;imagesc('cdata',squeeze(z.X))
