function [z]=jf_filter(z,varargin);
% apply a fir/iir filter to the data
%
% Options:
%  dim        -- the dimension along which to apply the filter
%  filter     -- the filter to use, either:
%                {'name' 'type' order cutoff} name and parameters for filter to use, one-of;
%                name is one-of: 
%                      'butter' -- butterworth, 'buttersos' -- butterworth with second-order-section imp
%                     'fir' -- finite impuluse response, 'firmin' -- min-phase finite impluse response
%                type is one of: 'low','high','stop','bandpass'
%                ord  is filter order : around fs for FIR or 6 for IIR (butter)
%                cuttoff is: [1x1] for low/high pass, or [lowcutoff highcutoff] for bandpass/stop
%               OR [nTaps x 1] a set of coefficients to use in a FIR filter
%               OR {[nTaps1 x 1] [nTaps2 x 1]} set of IIR coeffients
%  N.B. see FIR1 for how to make an FIR filter, e.g. B=fir1(30,fcutOff*2/fSample,'low');
%  N.B. see FIRMINPHASE for how to make a min-phase FIR or min-lag FIR filter
%  N.B. see BUTTER for how to make an IIR filter, e.g. [A,B]=butter(6,fcutOff*2/fSample,'low');
%  center     -- [bool] flag if we center along this dim before applying the filter  (1)
%  delayCompensate -- [bool] flag if we adjust dim-vals to compenstate for the delay induced by the filter (1) 
%  summary    -- additional summary description info

opts=struct('dim','time','filter',[],'center',1,'delayCompensate',1,'summary',[],'MAXEL',1e5,'verb',0,'subIdx',[]);
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim); 

gDelay=0;
if ( isnumeric(opts.filter) )
  filttype='fir';
  B=opts.filter(:); A=1;   
elseif ( iscell(opts.filter) && numel(opts.filter)==2 )
  filttype='iir';
  B=opts.filter{1}(:); A=opts.filter{2}(:);
elseif ( iscell(opts.filter) && isstr(opts.filter{1}) ) % filter name
  filttype=lower(opts.filter{1});
  fs=getSampRate(z);
  type=opts.filter{2};  ord=opts.filter{3};  bands=opts.filter{4};
  switch filttype;
    case {'butter','buttersos'}; [B,A]=butter(ord,bands*2/fs,type);
      if (strcmp(filttype,'buttersos')) [sos,sosg]=tf2sos(B,A); end;
    case {'fir','firmin'};        B   =fir1(ord,bands*2/fs,type); A=1;
    if ( isequal(opts.filter{1},'firmin') )B=firminphase(B); end
    [ans,gDelay]=max(abs(B));
   otherwise;
    error('Unrecognised filter design type');
  end
end

% force conversion to double precision to apply the filter to avoid numerical stabilty issues if needed
doubleFilter=false;
if ( ~isa(z.X,'double') && ~isempty(strmatch(filttype,{'butter','iir','buttersos'},'exact')) )
  doubleFilter=true;
end

szX=size(z.X);
[idx,allStrides,nchnks]=nextChunk([],szX,dim,opts.MAXEL);
ci=0; if ( opts.verb >= 0 && nchnks>1 ) fprintf('filter:'); end;
while ( ~isempty(idx) )
   tX = z.X(idx{:});
   if ( opts.center )     tX=repop(tX,'-',mean(tX,dim)); end;
   if ( strcmp(filttype,'buttersos') )
     tX=double(tX);
     if ( exist('sosfilt') ) 
        tX=sosfilt(sos,tX,dim);
     else
        for li=1:size(sos,1); % apply the filter cascade
           tX=filter(sos(li,1:3),sos(li,4:6),tX,[],dim);       
        end
        tX=sosg*tX;
     end
   else
     if ( doubleFilter ) 
       tX=filter(B,A,double(tX),[],dim);
       tX=single(tX); % covert back to single
     else
       tX=filter(B,A,tX,[],dim);
     end
   end
   z.X(idx{:})=tX;
   if( opts.verb>=0 ) ci=ci+1; textprogressbar(ci,nchnks);  end
   idx=nextChunk(idx,szX,allStrides);
end
if ( opts.verb>=0 && nchnks>1 ) fprintf('done\n'); end;

% discard the first part of the response as everything is shifted by the filter size
if ( opts.delayCompensate && any(strmatch(filttype,{'fir','firmin'})) )
   % shift the labels to deal with the time-shift induced by the filter
   offset=gDelay;
   if ( isnumeric(z.di(dim(1)).vals) )
     z.di(dim(1)).vals=z.di(dim(1)).vals - z.di(dim(1)).vals(offset);
   else % bodge the sample shift in
     z.di(dim(1)).vals(offset:end)=z.di(dim(1)).vals(1:end-offset+1);
   end
   % discard the startup corrupted data
   idx={}; for di=1:ndims(z.X); idx{di}=1:size(z.X,di); end; idx{dim(1)}=numel(B)+1:size(z.X,dim(1));
   z.X=z.X(idx{:});
   z.di(dim(1)).vals=z.di(dim(1)).vals(idx{dim(1)});
 end

summary=opts.summary;
info=struct('A',A,'B',B);
if ( strcmp(filttype,'buttersos') ) info.sos=sos; info.sosg=sosg; end;
z =jf_addprep(z,mfilename,summary,opts,info);
return;
%----------------------------------------------------
function testcase()
% make toy with pretty much a flat spectrum, 128hz samplerate and an ERP in most trials...
z=jf_mksfToy('sources',{{'prod' {'gaus' 100 20} {'sin' 15}}; {'coloredNoise' 1}}); 
% plot non-filtered
figure(1);clf;jf_plot(jf_ERP(jf_welchpsd(z)))
figure(1);clf;jf_plot(jf_ERP(z))
% plot fftfilterd
figure(2);clf;jf_plot(jf_ERP(jf_welchpsd(jf_fftfilter(z,'bands',[0 5 30 31]))))
% same again but with the butter-filter
figure(3);clf;jf_plot(jf_ERP(jf_welchpsd(jf_filter(z,'filter',{'butter' 'bandpass' 6 [5 30]}))))
% same again with fir bandpass
figure(4);clf;jf_plot(jf_ERP(jf_welchpsd(jf_filter(z,'filter',{'fir' 'bandpass' 20 [5 30]}))))
% same again with fir min-phase band pass
figure(5);clf;jf_plot(jf_ERP(jf_welchpsd(jf_filter(z,'filter',{'firmin' 'bandpass' 20 [5 30]}))))
% Time-domain comparsion
ch=1; ord=128; t0=z.di(2).vals(50);
clf;
[ans,t1]=min(abs(z.di(2).vals-t0));plot(shiftdim(mean(z.X(ch,t1+(1:100),:),3)),'b'); hold on;
zf=jf_filter(z,'filter',{'fir' 'bandpass' ord [5 30]});
[ans,t1]=min(abs(zf.di(2).vals-t0));plot(shiftdim(mean(zf.X(ch,t1+(1:100),:),3)),'g');
zfm=jf_filter(z,'filter',{'firmin' 'bandpass' ord [5 30]});
[ans,t1]=min(abs(zfm.di(2).vals-t0));plot(shiftdim(mean(zfm.X(ch,t1+(1:100),:),3)),'r');