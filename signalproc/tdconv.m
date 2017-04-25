function [fg]=tdconv(f,g,dim);
% Code to convolve f with the filter g along dim by direct multiply and sum
%
% Inputs:
%  f   -- signal 
%  g   -- filter [Lx1] vector.  L < size(f,dim)
%  dim -- dimension to filter along. (first non-singlenton dimension)
% Outputs:
%  fg  -- f convolved with g
if ( nargin < 3 || isempty(dim) ) dim=find(size(f)>1,1); end;
if ( size(g,1)==1 ) g=g'; end; % ensure col vector
for i=1:ndims(f); idx{i}=1:size(f,i); end;
for t=1:size(f,dim)-size(g,1)+1;
   idx{dim}=t+(1:size(g,1))-1;
   tt = sum(repop(f(idx{:}),'.*',shiftdim(g,-(dim-1))),dim);
   idx{dim}=t; fg(idx{:}) = tt;
end
return;

%----------------------------------------------------------------------------
function testCase();
f=cumsum(randn(1000,1000));
g=sin(linspace(0,1,10)*2*pi); % period 10 sin wave
fg_f=filter(g,1,f,[],1);
fg_F=fftconv(f,g,1);
fg_t=tdconv(f,g,1);
for i=1:size(f,1);
   clf;plot(fg_f(numel(g):end,i),'LineWidth',2); 
   hold on; plot(fg_F(numel(g):end,i),'r');
   plot(fg_t(:,i),'g');
   title(sprintf('%d',i));
   fprintf('Hit key\n');pause;
end
