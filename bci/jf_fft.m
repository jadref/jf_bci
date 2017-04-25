function [z]=jf_fft(z,varargin);
% fourier transform the data
%
% Options:
%  dim     -- spec the dimension to fft over ('time')
%  posfreq -- flag to only return the positive frequencies or all frequencies (1)
%  feat    -- [str] type of output to generate, one-of            ('complex')
%             'complex,'abs','angle','real','imag'
%  windowType - [str] the type of temporal window to use as given by mkFilter ([])
%  detrend -- [bool] detrend before fft'ing (0)
%  center  -- [bool] center before fft'ing (0)
%  norm    -- [bool] normalize output to be in input units, i.e. make the fourier basis unit length (1)
opts=struct('dim','time','posfreq',1,'outType','complex','feat',[],'win',[],'detrend',0,'center',0,'verb',0,'norm',1,'subIdx',[]);
[opts,varargin]=parseOpts(opts,varargin);
if ( isempty(opts.feat) ) opts.feat=opts.outType; end;

szX=size(z.X); nd=ndims(z.X);
% get the dim to work down
dim=n2d(z.di,opts.dim);

% Get the sample rate
fs=getSampRate(z);
ms2samp = fs/1000; samp2ms = 1000/fs; 

% data window to use
winFn=[]; nF=szX(dim);
if ( ~isempty(opts.win) ) 
  winFn = mkFilter(size(z.X,dim),opts.win,z.di(dim(1)).vals);
  nF=winFn(:)'*winFn(:);
end;

% do the conversion
if ( opts.posfreq )
  corMag=[0 0]; if ( opts.norm ) corMag=[1 1]; end;
  % pos freq only
  z.X  = fft_posfreq(z.X,[],dim,opts.outType,winFn,opts.detrend,opts.center,corMag,0,opts.verb); 
   freqs= [0:(size(z.X,dim)-1)]*1000/(szX(dim)*samp2ms);
else
   if ( opts.detrend || opts.center ) 
     warning('detrend/center not yet implementated for normal fft');
   end
   z.X  = fft(z.X,[],dim); % both pos and neg freq
	% normalise to unitary fourier basis
	if ( opts.norm ) z.X=z.X./sqrt(nF); end
   freqs= [0:(ceil((szX(dim)-1)/2)) -floor((szX(dim)-1)/2):-1]*1000/(szX(dim)*samp2ms);
   switch (lower(opts.outType));
    case 'complex'; 
    case {'l2','pow'};   z.X = 2*(real(X).^2 + imag(X).^2);    
    case {'abs','amp'};  z.X = abs(z.X);
    case 'angle';        z.X = angle(z.X);
    case 'real';         z.X = real(z.X);
    case 'imag';         z.X = imag(z.X);          
    case 'db';           z.X = 10*log10(z.X);
    otherwise; warning(sprintf('Unrecognised outType: %s',opts.outType));
   end
end

% update the dimInfo
odi=z.di;
z.di(dim)=mkDimInfo(size(z.X,dim),1,'freq','Hz',freqs);
summary=sprintf('over %s',odi(dim).name);
extrasum='';
if ( ~strcmp(opts.outType,'complex') ) extrasum=[opts.outType]; end;
if ( opts.posfreq )
  if(~isempty(extrasum))extrasum=[extrasum ',']; end;
  extrasum=[extrasum 'pos_freq'];
end;
if ( ~isempty(extrasum) ) summary=[summary ' (' extrasum ')']; end;
if ( ~strcmp(opts.outType,'complex') ) summary=[summary ' (' opts.outType ' output)']; end;
info=[];
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------
function testCase()
f=jf_fft(z);
jf_disp(f)
clf;image3ddi(abs(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
clf;image3ddi(real(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
clf;image3ddi(imag(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
