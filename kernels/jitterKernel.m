function K = jitterKernel(X,Z,varargin)
% compute gaussian kernel allowing for time shifts of the data
%
%K = kernel(X,Z,...)
%
%  K_i,j = \sum_{tau\in taus} \exp(-|X(:,1:end-tau)-Z(:,tau:end)|^2./Sigma)*\exp(-tau.^2/Sigma_tau)
%        = \sum_{tau\in taus} \exp(-(|X(:,1:end-tau)-Z(:,tau:end)|^2./Sigma + tau.^2/Sigma_tau))
% N.B. Sigma_tau is a gaussian prior over shifts
%
%Inputs:
% X - [n-d] input objects
% Z - [n-d] input objects
%Options:
% dim -- dim(1)     - time dimension in X,Z  ([1 2])
%        dim(2:end) - the dimensions along which trials lie in X,Z
% taus - [1x1 int] max time step offset to use, (size(X,dim(1))/2)
%        OR
%        [nTau x1\ set of *postive* jitters to search
% Sigma- [1x1 or 1x2] scaling for the guassian map,      (size(X,dim(1)))
%          controls strength of bias towards best match
% mode - 'str': oneof: 'max','sum','logsum'
%Outputs:
% K: [size(X,dim(2:end)) x size(X,dim(2:end))] the computed kernel matrix. 
opts=struct('dim',[1 2],'taus',[],'sigma',[],'mode','sum','verb',0);
opts=parseOpts(opts,varargin);

if (nargin < 1) % check correct number of arguments
   error('Insufficient arguments'); return;
end
if ( nargin< 2 ) Z=[]; end;
if ( isinteger(X) ) X=single(X); end % need floats for products!
% empty Z means use X
isgram=false;
if ( isempty(Z) ) Z=X; isgram=true; elseif ( isinteger(Z) ) Z=single(Z); end; 

% extract dim info
dim=opts.dim; dim(dim<0)=dim(dim<0)+ndims(X)+1;
tdim=dim(1); dim=dim(2:end); % split into time and trials

szX=size(X); szZ=size(Z); nd=max([max(dim),numel(szX),numel(szZ)]); szX(end+1:nd)=1; szZ(end+1:nd)=1; % input sizes

sigma=opts.sigma; if ( isempty(sigma) ) sigma=szX(tdim); end;
if ( numel(sigma)<2 ) sigma(2)=eps; end; % if not given default to no prior over jitters

taus=opts.taus; if ( isempty(taus) ) taus=0:ceil(szX(tdim)/2); elseif ( numel(taus)==1 ) taus=1:round(taus); end; 
taus=taus(taus>0); taus=[-taus(end:-1:1) 0 taus]; % ensure set of time shifts is symetric about 0..

% Compute the appropriate indexing expressions
idx  =-(1:numel(szX)); idx(dim)=1:numel(dim);               % normal index
tidx =-(1:numel(szX)); tidx(dim)=numel(dim)+(1:numel(dim)); % transposed index

% Norms
X2   = tprod(X,idx,X,idx,'n');
if ( ~isgram ) Z2   = tprod(Z,idx,Z,idx,'n'); else Z2=X2; end;
% compute the kernel for each possible time-shift
% index expressions for x and z
% TODO: Rewrite to compute the log(sum_tau exp(x(t-tau),y)) by default as it's numerically more stable
K = zeros([szX(dim),szZ(dim),numel(taus)],class(X)); 
xidx={}; zidx={}; for d=1:numel(szX); xidx{d}=1:size(X,d); zidx{d}=1:size(Z,d); end;
for taui=1:numel(taus);
  tau=taus(taui);
  % subset to use for IP, works for both positive and negative taus
  if( tau>=0 ) % shift x forwards
     xidx{tdim}=    1:size(X,tdim)-tau; zidx{tdim}=1+tau:size(Z,tdim);
  else         % shift x backwards
     xidx{tdim}=1-tau:size(X,tdim);     zidx{tdim}=    1:size(Z,tdim)+tau;
  end
  Ktau = tprod(X(xidx{:}),idx,Z(zidx{:}),tidx,'n');      % inner-prod at this time shift  
  Ktau = repop(X2,'+',repop(-2*Ktau,'+',shiftdim(Z2,-numel(dim)))); % squared distance at this time shift
  Ktau = Ktau./sigma(1); % scaled squared distance
  if(sigma(2)>eps) Ktau = Ktau + tau.^2./sigma(2); end; % include the prior over tau
  K(:,:,taui)=Ktau;
  if (opts.verb>=0) textprogressbar(taui,numel(taus)); end;
end
% combine the time-shifts in the desired (numerically robust) way
if( size(K,3)==1 ) % fast path easy case
   K = exp(-K);
else
   minK=min(K,[],3); % get the correction factor
   if ( strcmp(opts.mode,'max') )
      K=exp(-minK);   % use a hard max. N.B. *not PD*
   else
      K   =repop(K,'-',minK);
      if(strcmp(opts.mode,'logsum'))
         K = log(sum(exp(-K),3))+log(exp(minK)); % numerically corrected version
      elseif( strcmp(opts.mode,'sum') )
         K = sum(exp(-K),3).*exp(-minK); %numerically better version
      end
   end
end
if(opts.verb>=0)fprintf('\n'); end;
% % test if this works as a similarity under jitter
% sig=1;
% K1=exp(-Kt(:,:,ceil(end/2))*sig); % linear
% K2=sum(exp(-Kt*sig),3);           % softmax
% K3=max(exp(-Kt*sig),[],3);        % hard-max
% var(K1(:))./mean(K1(:)),var(K2(:))./mean(K2(:)),var(K3(:))./mean(K3(:)),
% clf;mimage(K1,K2,K3);
return;
%----------------------------------------------
function testCase()
X=randn(100,100);
K=jitterKernel(X,[],'taus',[0 10]);
% test kernel with jitter
X=zeros(100,100); % pure jittered signal
for i=1:size(X,2); X(:,i)=mkSig(size(X,1),'gaussian',50,7,17); end;
clf;imagesc(X);
K=jitterKernel(X,[],'sigma',1,'taus',0:10);
clf;imagesc(K);colorbar

% make a jittered toy problem for testing
nSamp=100;
nEpoch=100;
sources={{'gaus' 50 20 10} ...  % signal is DC shift
         {'randn'} }; %{'coloredNoise' exp(-[inf(1,0) zeros(1,1) linspace(0,5,40)])} }; % noise signal
y2mix  =cat(3,[1 2],[0 2]); % per class ERP magnitude % [ nSrc x nSig x nLab ]
rand('state',0); randn('state',0); % ensure all have same labeling
Yl     =sign(randn(nEpoch,1));  % epoch labels
z=jf_mksfToy(Yl,'sources',sources,'y2mix',y2mix,'nSamp',nSamp);

k=jf_compKernel(z);
kj=jf_compKernel(z,'kerType','jitterKernel','kerParm',{'dim',[2 3],'mode','max'});%,'taus',1:2:60});
kh=jf_compKernel(z,'kerType','jitterKernel','kerParm',{'dim',[2 3],'mode','sum'});
kl=jf_compKernel(z,'kerType','jitterKernel','kerParm',{'dim',[2 3],'mode','logsum'});%,'taus',1:2:60});
kg=jf_compKernel(z,'kerType','rbf',100);
jf_cvtrain(k)
jf_cvtrain(kj)
jf_cvtrain(kh)
jf_cvtrain(kl)
jf_cvtrain(kg)
