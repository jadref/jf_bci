function [z]=jf_welchpsd(z,varargin);
% Compute a welch-style power spectral density estimate
%
% [z]=jf_welchpsd(z,varargin)
% Options: (N.B. just as passed to welchpsd)
%  windowType -- type of time windowing to use ('hanning')
%  nwindows   -- number of windows to use      (12)
%  overlap    -- fractional overlap of windows (.5)
%  width_{ms,samp} -- window width in millisec or samples
%  start_{ms,samp} -- window start locations in millisec or samples
%  log        -- [bool] flag if we report log power or just power
%  outType    -- 'str' type of output to produce.  one of:
%                'amp' - ave amp, 'power' - ave power, 'db' - ave db
%  center     -- [bool] center before spectrum computation?   (0)
%  detrend    -- [bool] remove linear trends before spectrum computation (1)
opts=struct('dim','time','log',[],'subIdx',[],'verb',0);
specOpts=struct('start_ms',[],'start_samp',[],'aveType','amp','outType',[]);
[opts,specOpts,varargin]=parseOpts({opts,specOpts},varargin);
if ( ~isempty(opts.log) && opts.log ) specOpts.outType='db'; end;
if ( isempty(specOpts.outType) ) specOpts.outType=specOpts.aveType; else specOpts.aveType=specOpts.outType; end;
dim=n2d(z,opts.dim);

% Get the sample rate
fs=getSampRate(z);
ms2samp = fs/1000; samp2ms = 1000/fs; 

% convert start points in ms to samples
if( ~isempty(specOpts.start_ms) && isempty(specOpts.start_samp) )
   specOpts.start_samp = floor((specOpts.start_ms-z.di(dim).vals(1))*ms2samp); % samples relative to 0-time
end
% do the conversion
[z.X,specOpts] = welchpsd(z.X,dim,'fs',fs,specOpts,varargin{:});
%if ( opts.log ) z.X(z.X==0)=min(z.X(z.X>0)); z.X=20*log10(z.X); z.di(end).units='db'; end;
switch lower(specOpts.outType)
 case 'db';                        z.di(end).units='db';
 case {'amp','abs','medianabs'};   z.di(end).units='uV';
 case {'power','pow','medianpow'}; z.di(end).units='uV^2';
 otherwise; warning('unrec ave type');
end

% update the dimInfo
freqs   = [0:(size(z.X,dim)-1)]*1000/(specOpts.width_samp*samp2ms);
z.di(dim).name='freq';   z.di(dim).units='Hz';   z.di(dim).vals=freqs;
z =jf_addprep(z,mfilename,...
             sprintf('%d windows of %g ms (%s)',specOpts.nwindows,specOpts.width_samp*samp2ms,specOpts.outType),...
             mergeStruct(opts,specOpts),[]);
return;
%---------------------------------------------------------------------------
function testCase()
s=jf_welchpsd(z); % 12 win @ .5 overlap

s=jf_welchpsd(z,'nwindows',8); % 8 win @ .5 overlap

s=jf_welchpsd(z,'width_samp',100); % 100 samp wide @ .5 overlap

s=jf_welchpsd(z,'start_samp',1:100:size(z.X,1));%start 100 samp @ .5 overlap

s=jf_welchpsd(z,'start_samp',1:100:size(z.X,1),'width_samp',150);

% plot the result
clf;image3ddi(s.X(:,:,:,1:4),s.di,1,'clim',[],'ticklabs','SW','colorbar','ne');packplots('sizes','equal')
