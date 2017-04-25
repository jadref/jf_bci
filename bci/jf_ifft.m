function [z]=ifft(z,varargin);
% inverse fourier transform the data
%
% Options:
%  dim     -- spec the dimension to fft over ('time')
opts=struct('dim','freq','subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

sz=size(z.X); nd=ndims(z.X);

dim=n2d(z,opts.dim);
% Get the sample rate
fs=2*max(z.di(dim).vals);
ms2samp = fs/1000; samp2ms = 1000/fs; 
   
% do the conversion
if ( all(z.di(dim).vals(:)>0) ) 
   error('Inverting a posfreq fft-transformed signal unsupported currently');
   z.X  = fft_posfreq(z.X,[],dim); % pos freq only
   times= [0:size(z.X,dim)-1]*samp2ms;
else
   z.X  = ifft(z.X,[],dim); % both pos and neg freq
   times= [0:size(z.X,dim)-1]*samp2ms;
end

% update the dimInfo
odi=z.di;
z.di(dim)=mkDimInfo(size(z.X,dim),1,'time','ms',times);
z.di(dim).info.fs=fs;
summary=sprintf('over %s',odi(dim).name);
if ( all(z.di(dim).vals(:)>0) ) summary=[summary ' (pos_freq)']; end;
info=[];
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------------
function testCase()
f=jf_ifft(z);
jf_disp(f)
clf;image3ddi(abs(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
clf;image3ddi(real(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
clf;image3ddi(imag(f.X),f.di,1,'colorbar','nw');packplots('sizes','equal')
