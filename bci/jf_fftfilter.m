function [z]=jf_fftfilter(z,varargin);
% Spectrally filter (+up/down sample) using the fft technique
%
% Options:
%  dim        -- the time dimension
%  bands      -- set of bands to make filter for
%  len        -- output size (use to up/downsample at the same time) (size(z.X,dim))
%  fs         -- output sampling rate (use to up/downsample)              ([])
%  detrend    -- [int] flag if we linearly deTrend before computing the ffts    (1)
%               0=no, 1=detrend, 2=center
%  win        -- time domain window to apply before filtering             ([])
%  hilbert    -- [int] do a hilbert transform? (0)
%              0=non-hilbert, 1=hilbert with amp only, 2=hilbert with phase only, 3=hilbert with amp+phase
%  summary    -- additional summary description info
opts=struct('dim','time','len',[],'fs',[],'bands',[],'detrend',1,'win',[],'hilbert',0,'summary',[],'notStatusCh',1,'subIdx',[],'verb',-1);
opts=parseOpts(opts,varargin);

% get dim to work along
dim=n2d(z.di,opts.dim);
% extract the sampling rate
fs=getSampRate(z);
nSamp=size(z.X,dim);

if( ~isempty(opts.len) )     len=opts.len; if(any(len<0)) len=round(nSamp*abs(len)); end;
else                         len=nSamp;
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
if ( ~isempty(win) && ...
     (ischar(win) || iscell(win) || (numel(win)<=5 && numel(win)~=size(z.X,dim))) ) 
  win=mkFilter(size(z.X,dim),opts.win,z.di(dim).vals);
end

statCh=[];
if ( opts.notStatusCh && n2d(z,'ch',0,0) && any(strcmpi('status',z.di(n2d(z,'ch')).vals)) )
  % save status channel
  statIdx=subsrefDimInfo(z,'dim','ch','vals','status','valmatch','icase');
  statCh=z.X(statIdx{:});  
end

% do the actual filtering
filt=bands; if(~isnumeric(filt) || numel(filt)~=len(1) ) filt= mkFilter(floor(len(1)/2),filt,fs/len(1)); end;
if ( any(filt~=1) || len(1)~=len(2) ) % don't bother if not asked to do anything
  z.X = fftfilter(z.X,filt,len,dim,opts.detrend,win,opts.hilbert,[],opts.verb);
  if ( ~isempty(statCh) ) % put the non-filtered status channel back in
    statIdx{n2d(z,'time')}=1:size(z.X,dim);
    chIdx=statIdx; chIdx{n2d(z,'ch')}=1; chIdx{n2d(z,'time')}=round(linspace(1,size(statCh,2),size(z.X,dim)));
    z.X(statIdx{:})=statCh(chIdx{:}); 
  end
end
% compute the acutal output sampling rate, may be different from what was requested
len  =size(z.X,dim);
outfs=fs*len./nSamp;


% recording the updating info
summary='';
if ( ~iscell(bands) ) bands={bands}; end;
for bi=1:numel(bands); 
   if ( numel(bands{bi}) < 5 ) summary=[summary 'bands ']; else summary=[summary 'filt ']; end;
   summary=[summary sprintf('[%s%.1f',sprintf('%.1f ',bands{bi}(1:min(end-1,5))),bands{bi}(min(end,6)))];
   if ( numel(bands{bi}) < 5 ) summary=[summary ']']; else summary=[summary '...]']; end;
   if ( bi~=numel(bands) ) summary=[summary ' + ']; end;
end
if ( ~isempty(win) && ~isequal(win,1) ) summary=[summary ' wind']; end
if ( outfs~=fs ) 
   z.di(dim).vals = single(z.di(dim).vals(1)):(1000./outfs):single(z.di(dim).vals(end));
   z.di(dim).info.fs = outfs;
   sidx = round((1:numel(z.di(dim).vals))*fs./outfs); % nearest old point to the new one
   if ( numel(z.di(dim).extra)>0 ) z.di(dim).extra = z.di(dim).extra(sidx); end;
   summary=sprintf('%s + resample %3.1fHz->%3.1fHz',summary,fs,outfs); 
end;
if ( ~isempty(opts.summary) ) summary=sprintf('%s(%s)',summary,opts.summary); end;
switch (opts.hilbert);
 case {1,'abs'};     summary=sprintf('%s hilbert-amplitude',summary); 
 case {2,'angle'};   summary=sprintf('%s hilbert-phase',summary); z.di(end).units='rad'; 
 case {3,'complex'}; summary=sprintf('%s hilbert',summary);
 otherwise;
end
info=struct('filt',filt);
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%-----------------------------------------------------------------------
function []=testCase()

% make signal with power at 100Hz
oz = jf_import('test','test','test',shiftdim(mkSig(2000,'sin',10),-1),[],'fs',1000); z=oz;
w=jf_welchpsd(z,'width_ms',1000); jf_plotEEG(w);
% filter and downsample and test
z=oz;
z = jf_fftfilter(z,'bands',[40 101],'fs',500);
w=jf_welchpsd(z,'width_ms',1000); jf_plotEEG(w);
% filter to remove and test again
z = oz;
z = jf_fftfilter(z,'bands',[40 99],'fs',500);
w=jf_welchpsd(z,'width_ms',1000); jf_plotEEG(w);

