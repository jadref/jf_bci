function [W]=lcmv(A,Cxx,lambda)
% compute lcmv beanformer
% 
% [W]=lcmv(A,Cxx,lambda)
%
% Inputs:
%  A    -- [nElect x nOri x nSrc] forward matrix
%  Cxx  -- [nElect x nElect x nTr] data covariance matrix
%  lambda--[float] regularisation parameter for the inverse covariance
% Outputs:
%  W    -- [nElect x nOri x nSrc x nTr] inverse matrix
if ( nargin < 3 ) lambda=0; end;
% regularised inverse data covariance
iCxx=zeros(size(Cxx));
W=zeros([size(A) size(Cxx,3)]);
for ci=1:size(Cxx,3);
   iCxx = pinv(Cxx(:,:,ci) + lambda * eye(size(Cxx)));
   for si=1:size(A,3);
      % van Veen eqn. 23, use PINV/SVD to cover rank deficient leadfield
      W(:,:,si,ci) = (pinv( A(:,:,si)' * iCxx * A(:,:,si) ) * A(:,:,si)' * iCxx)';  
   end
end
return;
%-------------------------------------------
function testCase()
fwdMx=load('temp/fwdMx_64ch');
A=fwdMx.A;

W=lcmv(A,eye(size(A,1)),0); %
% normalise the filters to get the neural-activity-index, i.e. give each source equal power
W=repop(W,'./',tprod(W,[-1 -2 3],[],[-1 -2 3],'n'));

% compute the max-power direction projection of the W


% apply the inversion to the signal
Sest = tprod(X,[-1 3],W,[-1 1 2]); % [ori x nSrc x time]

% plot a group of sources arround the true source
neari = find(sum(repop(srcPos,'-',srcPos(:,ai)).^2)<22.^2);
clf;mcplot(reshape(Sest(:,neari,:),[],size(Sest,3))');
