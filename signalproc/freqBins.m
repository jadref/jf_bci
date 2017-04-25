function [freqs]=freqBins(L,dur,fs,posFreqOnly)
% [freqs]=freqBins(L,dur,fs,pposFreqOnly)
if ( nargin < 2 ) dur=[]; end;
if ( nargin < 4 || isempty(posFreqOnly) ) posFreqOnly=0; end;
if ( isempty(dur) && ~isempty(fs) )
   if ( isempty(dur) ) dur   = L/fs;
   else warning('fs ignored');
   end
end
if ( isempty(L) && ~isempty(dur) && ~isempty(fs) )
  L=dur*fs;
end
if ( isempty(dur) ) dur=1; end;
if ( posFreqOnly ) 
   freqs = [0 1:floor(L/2)]/dur;
else
   freqs = [0 1:floor(L/2) -floor(L/2)+(mod(L+1,2)):-1]/dur;
end
return;
%--------------------------------------------------------------------
function testCase()
plot(freqBins(101,2))
plot(freqBins(100,2))

