function [alphab,f,J]=kFDA(K,Y,C,varargin)
% Binary Kernel Fisher discriminant analysis (KFDA).
%
%function [alphab,f,J]=kFDA(K,Y,C,varargin)
%
%INPUTS
% K     - [N x N] kernel matrix
% Y     - [N x 1] set of labels
% C     - [1 x 1] regularization parameter
% 
%OUTPUTS
% alphab- [N+1 x 1] classifier weight vector
% f     - [N x 1] The decision value for all the inputs
% J     - the final objective value

if ( nargin < 3 ) C(1)=0; end;

% Identify the training points
Y  = Y(:);
trnInd = (Y~=0); 
Ytrn = Y(trnInd);
if( ~all(trnInd) ) Ktrn = double(K(trnInd,trnInd)); else Ktrn=double(K); end % subset if needed

% lambda=gamma;

% ell = size(Ktrn,1);
% ellplus = (sum(Ytrn) + ell)/2;
% yplus = 0.5*(Ytrn + 1);
% ellminus = ell - ellplus;
% yminus = yplus - Ytrn;
% rescale = ones(ell,1)+Ytrn*((ellminus-ellplus)/ell);
% plusfactor = 2*ellminus/(ell*ellplus);
% minusfactor = 2*ellplus/(ell*ellminus);
% B = diag(rescale) - (plusfactor * yplus) * yplus' ...
%       - (minusfactor * yminus) * yminus';
% alpha = (B*Ktrn + C(1)*eye(ell,ell))\Ytrn;
% b = -0.25*(alpha'*Ktrn*rescale)/(ellplus*ellminus);

N  = sum(trnInd);
Np = sum(Ytrn>0);%(N+sum(Ytrn))/2;%
Yp = double(Ytrn>0) ; % convert Y to 0-1 range
Nn = sum(Ytrn<0);%(N-sum(Ytrn))/2;%
Yn = double(Ytrn<0);
rescale = ones(N,1)+Ytrn*((Nn-Np)/N);
Pfac    = 2*Nn/(N*Np);
Nfac    = 2*Np/(N*Nn);
Ktrn    = (diag(rescale) - (Pfac*Yp)*Yp' - (Nfac*Yn)*Yn')*Ktrn;
% Rescale and center the kernel. More efficient computation of the rescaled kernel as we want it
%Ktrn    = repop(rescale,'*',Ktrn) - Pfac*Yp*Yp'*Ktrn - Nfac*Yn*Yn'*Ktrn;
% add regulariser to the diagonal
if ( C(1)~=0 ) Ktrn(1:size(Ktrn,1)+1:end)=Ktrn(1:size(Ktrn,2)+1:end)+C(1);  end;
alpha   = Ktrn\Ytrn; % solve for alpha
b       = -0.25*(alpha'*Ktrn*rescale)/(Np*Nn);

% Compute the outputs
alphab  = zeros(size(K,1)+1,1); alphab(trnInd)=alpha; alphab(end)=b;
f       = alphab(1:end-1)'*K + alphab(end); f=f(:);
J       = 0;
return;

%--------------------------------------------------------------------------
function testCase()

[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);[dim,N]=size(X);

K=X'*X; % N.B. add 1 to give implicit bias term
fInds=gennFold(Y,10,'perm',1); trnInd=any(fInds(:,1:9),2); tstInd=fInds(:,10);

[alphab,f,J]=kFDA(K,Y,1);

[alphab,f,J]=kFDA(K,Y.*double(trnInd),1);
dv=K*alphab(1:end-1)+alphab(end);
dv2auc(Y(trnInd),dv(trnInd))
dv2auc(Y(tstInd),dv(tstInd))

% for linear kernel
plotLinDecisFn(X,Y,X*alphab(1:end-1),alphab(end),alphab(1:end-1));


[alpha]=kFDA(K(trnInd,trnInd),Y(trnInd),1)