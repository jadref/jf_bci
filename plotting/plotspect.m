function [varargout]=plotspect(f,sf,dim,varargin);
% Plot the spectrum of the input data
%
% [hdl,spect,freqs]=plotspect(x[,sf,dim,...])
% 
% Inputs:
%  x   -- n-d data matrix
%  sf  -- sampling frequency of x
%  dim -- time dimension of x (1)
%  ... -- additional options to pass to plot
% Outputs:
%  spect -- spectrum of f
%  h     -- handle of the lines we plotted

if ( nargin < 2 || isempty(sf) ) sf=1; end;
if ( nargin < 3 || isempty(dim) ) dim=find(size(f)>1,1,'first'); end;

% Simple fourier trans to make the plot
ff=fft(f,[],dim);

% Spectrum is positive frequencies only
for i=1:ndims(f); idx{i}=1:size(f,i); end; idx{dim}=1:floor(size(f,dim)/2);
spect=abs(ff(idx{:}));

% Plot
fs=(0:floor(size(f,dim)/2)-1)/size(f,dim)*sf;
h=plot(fs,spect,varargin{:});
if ( nargout>0 ) varargout={h,spect,fs}; end;
return;

%--------------------------------------------------------------------------
function testCase()
f=cumsum(randn(1000,1000));
g=sin(linspace(0,100,10*100)*2*pi); % period 10 sin wave
plotspect(g,10)