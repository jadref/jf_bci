function [classifier,fsoln]=extractClassifier(z)
% Extract the necessary pre-processing parameters
% extract the bad-channel list
isbad=[];%false(size(c.X,n2d(c.di,'ch')),1);
% from direct rejection/retention
% badchpi=strmatch('jf_reject',{z.prep.method}); % find bad-xxx removal steps
% for i=1:numel(badchpi); 
%    if ( strmatch('ch',z.prep(badchpi(i)).opts.dim) ) badchpi=badchpi(i); break; end; 
% end; % find ch removal
% if( isfield(z.prep(badchpi).info,'src') )
%    isbad=~(z.prep(badchpi).info.src.info{1}.ind);
% else
%    isbad=~(z.prep(badchpi).info.ind); % get the bad ch list
% end
% from rmOutliers
badchpi=strmatch('jf_rmOutliers',{z.prep.method}); % find bad-xxx removal steps
for i=1:numel(badchpi); 
   if ( strmatch('ch',z.prep(badchpi(i)).opts.dim) ) badchpi=badchpi(i); break; end; 
end; % find ch removal
tisbad=[z.prep(badchpi).info.isbad];
if ( isempty(tisbad) ) isbad=tisbad;
else                   isbad(1:numel(tisbad))=isbad(1:numel(tisbad)) | tisbad; 
end
% from retain
% badchpi=strmatch('jf_retain',{z.prep.method}); % find bad-xxx removal steps
% for i=1:numel(badchpi); 
%    if ( strmatch('ch',z.prep(badchpi(i)).opts.dim) ) badchpi=badchpi(i); break; end; 
% end; % find ch removal
% if( isfield(z.prep(badchpi).info,'src') )
%    tisbad=~(z.prep(badchpi).info.src.info{1}.ind);
% else
%    tisbad=~(z.prep(badchpi).info.ind); % get the bad ch list
% end
% isbad(1:numel(tisbad))=isbad(1:numel(tisbad)) | tisbad; % get the bad ch list

% extract the spectral filter parameters
fftpi=strmatch('jf_fftfilter',{z.prep.method});
if ( ~isempty(fftpi) ) % only the last filtering really matters
   filtopts.bands=z.prep(fftpi(end)).opts.bands;  filtopts.detrend=z.prep(fftpi(end)).opts.detrend;
else
   filtopts=[];
end

% extract the whitening transformation parameters
whtpi=strmatch('jf_whiten',{z.prep.method}); 
if (~isempty(whtpi) ) 
   W_wht=z.prep(whtpi).info.W; 
else  % check if we just applied it to the data, and use that instead
   whtpi=strmatch('jf_linMapDim',{z.prep.method});
   if(~isempty(whtpi)) % from linMapDim
      W_wht=z.prep(whtpi).opts.mx;
   else
      W_wht=1; 
   end
end;
% Extract the classifier weight vector
clspi=m2p(z,'jf_cvtrain');
soln=z.prep(clspi).info.res.opt.soln;
if (numel(soln)>1) warning('Only 1st sub-problem solution extracted'); end;

% use the classifier objective function info to decide how to transform into a cov-weighting
switch ( z.prep(clspi).opts.objFn );
 
 case {'klr_cg','l2svm_cg'}; % normal linear-kernel method
  X = z.prep(m2p(z,'jf_compKernel')).info.X;
  W_cls=tprod(X,[1:ndims(X)-1 -1],soln{1}(1:end-1),[-1],'n'); b=soln(end);
 
 case {'LSigmaRegKLR','LSigmaRegKLR_als'};
  W_cls=parafac(soln{1}{1:end-1}); % put the factors back together
  b=soln{1}{end};  % get the bias
 
 otherwise;
  error('Unrecognised objective function');
end

% combine classification and whitening and bad-channel rejection in 1 step
if ( m2p(z,'jf_cov') ) % covariance classification
   W    =  single(double(W_wht) * double(W_cls) * double(W_wht)'); %ARGH! matlab bug needs double
   if ( any(isbad) ) 
      oW=W; wsz=size(W);
      W = zeros([numel(isbad),numel(isbad),wsz(3:end)],class(W));
      W(~isbad,~isbad,:)=oW(:,:,:);      
   end   
else % input classification
   W    =  single( double(W_wht) * double(W_cls) );
   if ( any(isbad) ) 
      oW=W; wsz=size(W);
      W = zeros([numel(isbad),wsz(2:end)],class(W));
      W(~isbad,:)=oW(:,:);
   end   
end

% put all the parameters into 1 structure
classifier = struct('W',W,'b',b,'isbad',isbad,'fftfilt',filtopts,'soln',soln,'W_wht',W_wht,'W_cls',W_cls);
return;
