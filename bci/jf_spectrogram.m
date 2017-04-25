function [z]=jf_spectrogram(z,varargin);
% compute a spectrogram of the data
%
% Options:
%  dim -- time dimension
%  windowType -- [str] type window to use on each time window                   ('hamming')
%  nwindows   -- [int] number of windows to use                                 ([])
%  overlap    -- [float] fractional overlap of the windows                      (.5)
%  width_ms/width_samp -- [float] width of the time window in ms/samples        (500)
%  start_ms/start_samp -- [float nWin x 1] start position of each window
%  verb       -- [int] verbosity level
%  feat       -- [str] type of feature to compute, one-of:                      ('abs')
%                   'complex'    - normal fourier coefficients
%                   'l2','power' - squared length of the complex coefficient
%                   'abs'        - absolute length of the coefficients
%                   'angle'      - angle of the complex coefficient
%                   'real'       - real part
%                   'imag'       - imaginary part
%                   'db'         - decibles, ie. 10*log10(F(X).^2)
%  detrend     -- [bool] detrend data before computing spectrum                 ([])
%  center      -- [bool] center (0-mean over time) before computing spectrum    ([])
%
% See also: SPECTROGRAM
opts=struct('dim','time','subIdx',[],'verb',0);
specOpts=struct('windowType','hamming','overlap',[]); % spec windows as # and overlap
[opts,specOpts,varargin]=parseOpts({opts,specOpts},varargin);

dim=n2d(z.di,opts.dim);

% Get the sample rate
fs=getSampRate(z);
ms2samp = fs/1000; samp2ms = 1000/fs; 

if ( isempty(specOpts.overlap) ) % set the overlap based on the window type
  switch lower(specOpts.windowType);
	 case 'blackman'; specOpts.overlap=.3;
	 case 'kaiser';   specOpts.overlap=.45;
	 otherwise; specOpts.overlap=.5;
  end
end

% do the conversion
[z.X,start_samp,freqs,winFn,specOpts,width_samp] = spectrogram(z.X,dim,'fs',fs,specOpts,varargin{:});

% update the dimInfo
odi     = z.di;
z.di    = z.di([1:dim dim dim+1:end]); 
z.di(dim  ).name='freq'; z.di(dim  ).units='Hz'; z.di(dim).vals=freqs;
z.di(dim+1).name='time'; z.di(dim+1).units='ms'; z.di(dim+1).vals=z.di(dim+1).vals(max(1,min(start_samp+round(width_samp/2),end)));
z.di(dim+1).extra(:)=struct();
if ( all(diff(start_samp,2)==0) ) 
   z.di(dim+1).info.fs = single(1000.0./median(diff(z.di(dim+1).vals))); 
end;
if ( specOpts.log ) z.di(end).units='dB'; else z.di(end).units = 'uV'; end;

if ( isfield(z,'Y') && isfield(z,'Ydi') && n2d(z.Ydi,odi(dim).name,0,0) ) % need to sub-sample Y also
   z=subsampleY(z,start_samp,odi(dim).name); % subsample Y
end

info=struct('winFn',winFn,'freqs',freqs,'start_samp',start_samp,'width_samp',width_samp);
z =jf_addprep(z,mfilename,...
             sprintf('%d %s windows of %g ms',specOpts.nwindows,specOpts.windowType,...
                     specOpts.width_samp*samp2ms),...
             mergeStruct(opts,specOpts),info);
return;
%---------------------------------------------------------------------------
function testCase()
s=jf_spectrogram(z); % 12 win @ .5 overlap

s=jf_spectrogram(z,'nwindows',8); % 8 win @ .5 overlap

s=jf_spectrogram(z,'width_samp',100); % 100 samp wide @ .5 overlap

s=jf_spectrogram(z,'start_samp',1:100:size(z.X,1));%start 100 samp @ .5 overlap

s=jf_spectrogram(z,'start_samp',1:100:size(z.X,1),'width_samp',150);

s=jf_spectrogram(z,'width_ms',500,'feat','complex'); % complex outputs

% plot the result
clf;image3ddi(s.X(:,:,:,1:4),s.di,1,'clim',[],'ticklabs','SW','colorbar','ne');packplots('sizes','equal')
