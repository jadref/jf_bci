function [fs]=getSampRate(z,varargin)
opts=struct('dim','time','verb',1);
opts=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);
if ( isfield(z.di(dim(1)).info,'fs') ) fs=z.di(dim(1)).info.fs;
elseif ( isfield(z,'fs') ) fs=z.fs;
else
  if( opts.verb>0 ) warning('Couldnt find sample rate in z, estimating from vals'); end
  fs = 1000./mean(diff(z.di(dim(1)).vals));
  switch ( z.di(dim(1)).units )
   case {'s','s'}; fs=fs/1000; 
  end
end
% ensure output is of float type
if( ~isfloat(fs) ) fs=single(fs); end;
return;