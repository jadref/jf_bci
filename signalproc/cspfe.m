function [sf,d,Sigmai,Sigmacp,Sigmacn]=cspfe(X,Y,dim,cent,ridge)
% Common Spatial Filters feature extractor (assumes white cov input)
%
% [sf,d,Sigmai,Sigmacp,Sigmacn]=cspfe(X,Y,dim);
% N.B. if inputs are singular then d will contain 0 eigenvalues & sf==0
% Inputs:
%  X     -- n-d data matrix, e.g. [nCh x nSamp x nTrials] data set,
%        OR
%           [nCh x nCh x nTrials] set of *trial* covariance matrices,
%        OR
%           [nCh x nCh x nClass ] set of *class* covariance matrices
%  Y     -- [nTrials x 1] set of trial labels, with nClass unique labels, OR
%           N.B. in all cases a label of 0 indicates ignored trial
%  dim   -- [1 x 2] dimension of X which contains the trials, and
%           (optionally) the the one which contains the channels.  If
%           channel dim not given the next available dim is used. ([-1 1])
%  cent  -- [bool] center the data (0)
%  ridge -- [float] size of ridge (as fraction of mean eigenvalue) to add for numerical stability (1e-7)
% Outputs:
%  sf    -- [nCh x nCh x nClass] sets of 1-vs-rest spatial *filters*
%           sorted in order of increasing eigenvalue.
%           N.B. sf is normalised such that: mean_i sf'*cov(X_i)*sf = I
%           N.B. to obtain spatial *patterns* just use, sp = Sigma*sf ;
%  d     -- [nCh x nClass] spatial filter eigen values
if ( nargin < 3 || isempty(dim) ) dim=[-1]; end;
if ( nargin<4 || isempty(cent) ) cent=false; end;
if ( nargin<5 || isempty(ridge) ) ridge=1e-5; end;
dim(dim<0)=ndims(X)+dim(dim<0)+1; % convert negative dims
dim(end+1:2)=1; % default channel dim

nCh=size(X,dim(2)); N=size(X,dim(1));

% convert to per-trial covariances
if ( ~isequal(dim,[3 1]) || ndims(X)>3 || nCh ~= size(X,2) )         
   idx1=-(1:ndims(X)); idx2=-(1:ndims(X)); % sum out everything but ch, trials
   idx1(dim(1))=3;     idx2(dim(1))=3;     % linear over trial dimension
   idx1(dim(2))=1;     idx2(dim(2))=2;     % Outer product over ch dimension
   Sigmai = tprod(X,idx1,[],idx2,'n');
   if ( cent ) % center the co-variances, N.B. tprod to comp means for mem
      error('Unsupported -- numerically unsound, center before instead');
   end
else
   Sigmai = X;
end

if ( ndims(Y)==2 && min(size(Y))==1 && ~(all(Y(:)==-1 | Y(:)==0 | Y(:)==1)) ) 
  oY=Y;
  Y=lab2ind(Y,[],[],[],0); 
end;
nClass=size(Y,2);

sf    = zeros([nCh,nCh,nClass],class(Sigmai)); d=zeros(nCh,nClass,class(Sigmai));
for c=1:nClass; % generate sf's for each sub-problem
   Sigmacp(:,:,c) = sum(double(Sigmai(:,:,Y(:,c)>0)),3); % +class covariance 
   % N.B. use double to avoid rounding issues with the inv(Sigma) bit
   [W D]    =eig(Sigmacp(:,:,c)); D=diag(D)./sum(Y(:,c)>0); 
   [dcp,di] =sort(D,'descend'); Wp=W(:,di); % order in decreasing eigenvalue

   if ( 1 )
     Sigmacn(:,:,c) = sum(double(Sigmai(:,:,Y(:,c)<0)),3); % -class covariance 
     % N.B. use double to avoid rounding issues with the inv(Sigma) bit
     [W D]    =eig(Sigmacn(:,:,c)); D=diag(D)./sum(Y(:,c)<0);
     [dcn,di] =sort(-D,'descend'); Wn=W(:,di); % order in decreasing eigenvalue
   else
     dcn=dcp; Wn=Wp;
   end

   % Save the top half of the normalised filters & eigenvalues
   sf(:,1:round(nCh/2),c)    = Wp(:,1:round(nCh/2));
   d(1:round(nCh/2),c)       = dcp(1:round(nCh/2));
   sf(:,round(nCh/2)+1:end,c)= Wn(:,round(nCh/2)+1:end); % N.B. reverse order!
   d(round(nCh/2)+1:end,c)   = dcn(round(nCh/2)+1:end);   
 end
% Compute last class covariance if wanted
if ( nClass==1 & nargout>3 ) Sigmacp(:,:,2)=Sigmacp(:,:,1);end;
return;

%-----------------------------------------------------------------------------
function testCase()
% validate equivalence of the 2 pipelines
nCh = 10; nSamp = 100; N=300;
X=randn(nCh,nSamp,N);
Y=sign(randn(N,1));
z=jf_mksfToy(); z.X=z.X+randn(size(z.X))*norm(z.X(:))*1e-7;
X=z.X; Y=z.Y;
[sf1,d1]=csp(X,Y,3,0,0,0,0);
sf1=repop(sf1,'/',sqrt(sum(sf1.^2)));

% whiten then feat extract
[W,Dw,wX]=whiten(X,1,1,0,[],[],[],[],1); % whiten
[sf2,d2]=cspfe(wX,Y,3); % feature extract
sf2=W*sf2;
sf2=repop(sf2,'/',sqrt(sum(sf2.^2)));
%clf;plot([d1 d2])
%clf;mimage(sf1,sf2,'diff',1);
% component direction similarity!  should be identical up to sign changes
clf;imagesc(sf1'*sf2); 
clf;jplot([z.di(1).extra.pos2d],[sf1(:,1:min(3,size(sf2,2))) sf2(:,1:min(3,size(sf2,2)))],'clim','cent0');
