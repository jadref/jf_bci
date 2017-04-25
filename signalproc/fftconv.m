function [fg,Ff,Fg]=fftconv(f,g,dim);
% Code to rapidly compute the convolution of f with the filter g along dim
% using the fourier transform theorem.
% Inputs:
%  f   -- signal 
%  g   -- filter [Lx1] vector.  L < size(f,dim)
%  dim -- dimension to filter along. (first non-singlenton dimension)
% Outputs:
%  fg  -- f convolved with g
%  Ff  -- fft(f,[],dim);
%  Fg  -- fft(g,size(f,dim),1);
MAXEL=2e6;
if ( nargin < 3 || isempty(dim) ) dim=find(size(f)>1,1); end;
if ( size(g,1)==1 ) g=g'; end; % ensure col vector
Fg=fft(g,size(f,dim),1);

% Chunking computation for memory savings
szf = size(f);
if ( islogical(f) ) f=single(f); end;
fg=zeros(szf,class(f));
[idx,allStrides]=nextChunk([],szf,dim,MAXEL);
while ( ~isempty(idx) )
   Ff=fft(f(idx{:}),[],dim);
   fg(idx{:})=ifft(repop(Ff,'.*',shiftdim(conj(Fg),-(dim-1))),[],dim);
   idx=nextChunk(idx,szf,allStrides);
end
return;

%---------------------------------------------------------------------------
function testcase()
f=cumsum(randn(1000,1000));
g=sin(linspace(0,1,10)*2*pi); % period 10 sin wave
fg_f=filter(g,1,f);
fg_F=fftconv(f,g,1);
fg_t=tdconv(f,g,1);
for i=1:size(f,1);
   clf;plot(fg_f(:,i),'LineWidth',2); hold on; plot(fg_F(:,i),'r');
   title(sprintf('%d',i));
   fprintf('Hit key\n');pause;
end

% check the phase shift this induces
ff(:,1)=[zeros(10,1);ones(10,1)];        fc(:,1)=ff(:,1);
ff(:,2)=fftconv(ff(:,1),ones(2,1)/2,1);  fc(:,2)=[zeros(floor(2/2),1);ff(1:end-floor(2/2),2)];
ff(:,3)=fftconv(ff(:,1),ones(3,1)/3,1);  fc(:,3)=[zeros(floor(3/2),1);ff(1:end-floor(3/2),3)];
ff(:,4)=fftconv(ff(:,1),ones(4,1)/4,1);  fc(:,4)=[zeros(floor(4/2),1);ff(1:end-floor(4/2),4)];
ff(:,5)=fftconv(ff(:,1),ones(5,1)/5,1);  fc(:,5)=[zeros(floor(5/2),1);ff(1:end-floor(5/2),5)];
subplot(211);plot(ff);grid on;           subplot(212);plot(fc);grid on;
