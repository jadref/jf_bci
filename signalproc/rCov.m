function [covXY,wght]=rCov(X,Y,est,param,wght,maxIter,convThresh)
%Compute a robust covariance estimate, using either the M-estimator (Mest)
%Sign covariance matrix (SCM), the MCD (sigmaFrac) or 
%sigma-Threshold (sigmaThresh) based re-weighting functions
%Inputs:
% X -- [d X N] matrix of d-dimensional samples
% Y -- (defaluts to == X) [l x N] matrix of other samples
% estType - {'wght','Mest','sigmaFrac0','sigmaFrac','sigmaThresh0','sigmaThresh','shrinkage'}
% param   - L-norm order for Mest, outlier threshold for others
%           N.B. param==0 means default to normal covariance estimate!
% wght    - initial per-point weighting
% maxIter - number times to iterate the re-weighting
% convThresh - min change in solution to stop  
%Output:
% covXY -- covariance matrix
% wght  -- final point weighting
%
% $Id: rCov.m,v 1.13 2006-11-17 12:51:52 jdrf Exp $
if ( nargin < 2 | isempty(Y) ) Y=X; end; % X=Y is default
if ( nargin < 3 ) est=[]; end;
if ( nargin > 2 && nargin < 4 ) error('Need to give a parameter'); end;
if ( nargin < 5 ) wght=[]; end;
if ( nargin < 6 ) maxIter=10; end;
if ( nargin < 7 ) convThresh=1e-5; end;
[dim,nEx]=size(X); [dimY,nExY]=size(Y);

if ( isempty(wght) )  % compute the initial covariance estimate
   wght=ones(nEx,1);
   covXY=X*Y'; % unit-weighted estimate
else % compute weighted cov est, with the given weights
   covXY=zeros(dim,dim);
   for ex=1:nEx; 
      covXY=covXY+wght(ex)*X(:,ex)*Y(:,ex)';
   end
   covXY=covXY*nEx./(sum(wght)+(sum(wght)==0));  % stop division by 0
end

if ( strcmp(est,'shrinkage') ) % shrinkage estimate
   if ( param < 0 ) % negative param means use the given shrinkage factor
      lambda=-param;
   
   else % compute shrinkage parameter as from:
      % J. Schäfer & K. Strimmer, 
      % "Shrinkage approach to Large-Scale Covariance Matrix Est", Stat Applic
      % in Genetics and Molecular Biology, 

      % noramlise the weight
      wght=wght./(sum(wght)+(sum(wght)==0));
      % compute the bias correction factors...
      h1=1/(1-wght'*wght);    % n/(n-1) for equal weight
      h3=(wght'*wght)*h1.^3;  % n^2/(n-1)^3 for equal weight
      
      covXY=covXY./nEx*h1;  % convert to bias corrected cov est
      % compute the variance of the covariance matrix entries.
      varcovXY=zeros(size(covXY));
      for ex=1:nEx;
         varcovXY=varcovXY+wght(ex)*(X(:,ex)*Y(:,ex)'-covXY).^2;
      end
      varcovXY=varcovXY*h3; % normalise & bias correct
      
      % compute the optimal shrinkage factor, using the equation for optimal
      % shrinkage towards a diagonal only covariance matrix:      
      % \lambda^* = \sum_{i!=j} var(cov(x_i,y_j)) / \sum{i!=j} cov(x_i,y_i)^2
      lambda=(sum(varcovXY(:))-sum(diag(varcovXY))) ./ ...
             (sum(covXY(:).^2)-sum(diag(covXY).^2));
   end

   % bound the shrinkage 0<lambda<1
   lambda=min(1,max(lambda,0));
   
   % compute the shrinkage estimator -- N.B. pretty much adaptive ridge!
   covXY=(1-lambda)*covXY+lambda*diag(diag(covXY));
   covXY=covXY*nEx./h1; % BODGE: scale back up to an X*X' equivalent...
end

% return if we don't want a robust estimate
% BODGE: param=0 means stop now!
if( isempty(est) || param==0 || ...
    any(strcmp(est,{'none','wght','shrinkage'})) || ...
   (any(strmatch(est,{'sigmaFrac','sigmaFrac0'})) && param==0) || ...
   (any(strmatch(est,{'sigmaThresh','sigmaThresh0'})) && param==inf) ) 
   return; 
end;

% Refine the initial estimate into a robust estimate
for iter=1:maxIter;       
  icovXY=pinv(covXY);                  % pinv to deal with rCov(X,Y) cases
  for ex=1:nEx;
     wght(ex)=X(:,ex)'*icovXY*Y(:,ex); % comp mal dis for each example
  end
   
  % comp the reweighting factor
  switch est
    case 'SCM';                          % Sign covariance matrix
      nwght=1./sum(X.*X)';               % map each pt to unit sphere
    case 'Mest';                         % M-estimator style re-weighting
      nwght=max(wght,eps).^(param-2)/2;           
    case {'sigmaFrac','sigmaFrac0'};     %Minimum covariance determinate
      [sortw]=sort(wght,'descend'); thresh=sortw(max(floor(param*nEx),1));
      nwght=single(wght<=thresh);
      if ( est(end)~='0' )               % 0 or map to bound?
         nwght(nwght==0) = thresh./wght(nwght==0);  % map to bound
      end
    case {'sigmaThresh','sigmaThresh0'}; % Sigma based thresholding
      % BODGE: *sqrt(nEx/dim) to normalise to unit var  
      thresh=param.^2/nEx*dim;     
      nwght=single(wght <= thresh);  % <=> sqrt(wght*nEx/dim) <= param     
      if ( est(end)~='0' )               % 0 or map to bound?
         nwght(nwght==0) = thresh./wght(nwght==0);  % map to bound
      end
    otherwise error(['Unrecognised weight update type: ' est]);
  end
  
  if ( sum(abs(nwght-wght))/sum(abs(wght)) < convThresh ) break; end;
  wght=nwght;
  
  covXY(:)=0;
  for ex=1:nEx; % compute weighted cov est
     covXY=covXY+wght(ex)*X(:,ex)*Y(:,ex)';
  end
  covXY=covXY*nEx./(sum(wght)+(sum(wght)==0));  

end
return;

%-----------------------------------------------------------------------------
function []=testCases()
X=loadIEEE2004('je');
z=loadprep(bci('je'),'c_dsd');
z=rCov(mxcat(z.x,3)',[],'sigmaFrac',.03);
z=rCov(mxcat(z.x,3)',[],'shrinkage',1);