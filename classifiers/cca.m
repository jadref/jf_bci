function [Wx,lambdax,Wy,XX,YY,XY]=cca(X,Y,C,varargin)
% Primal canonical correlation analysis
%
% [Wx,lambda,Wy,XX,YY,XY]=cca(X,Y,C,varargin)
% OR
% [lambda]=cca(X,Y,C,varargin)
%
% options:
%  dim -- [int] dimension of X,Y which contain the trials
opts=struct('dim',[]);
opts=parseOpts(opts,varargin);
if ( nargin < 3 || isempty(C) ) C=eps(X)*1e6; end;
dim=opts.dim; if ( isempty(dim) ) dim=[ndims(X),ndims(Y)]; end; dim(end+1:2)=dim(end);

% info about inputs
nDims = max([ndims(X),ndims(Y),dim]);
szX   = size(X); szY   = size(Y);

% compute the covariance matrices
xIdx = 1:nDims; xIdx(dim(1))=-dim(1); xIdx2=nDims+xIdx; xIdx2(dim(1))=-dim(1);
yIdx = 1:nDims; yIdx(dim(2))=-dim(1); yIdx2=nDims+yIdx; yIdx2(dim(2))=-dim(1);
XX = tprod(X,xIdx,[],xIdx2,'n');
YY = tprod(Y,yIdx,[],yIdx2,'n');
XY = tprod(X,xIdx,Y,yIdx2,'n');

% compress the results to 2-d
XX = reshape(XX,[prod(szX([1:dim(1)-1 dim(1)+1:end])) prod(szX([1:dim(1)-1 dim(1)+1:end]))]);
YY = reshape(YY,[prod(szY([1:dim(2)-1 dim(2)+1:end])) prod(szY([1:dim(2)-1 dim(2)+1:end]))]);
XY = reshape(XY,[prod(szX([1:dim(1)-1 dim(1)+1:end])) prod(szY([1:dim(2)-1 dim(2)+1:end]))]);

% add the regulariser
if ( any(C~=0) )
   XX(1:size(XX,1)+1:end)=XX(1:size(XX,1)+1:end)+C(1);
   YY(1:size(YY,1)+1:end)=YY(1:size(YY,1)+1:end)+C(min(end,2));
end

% compute the solutions
Rxx = chol(XX); %(R'*R=X)
Ryy = chol(YY);
% numerical tests seem to say that this is the stabilist and comp cheapest solution
RxxXYRyy = Rxx'\XY/Ryy;
[Wx,lambdax] = eig(RxxXYRyy*RxxXYRyy');  lambdax=diag(lambdax); lambdax=sqrt(lambdax);
Wy = repop(Ryy\(RxxXYRyy'*Wx),'./',lambdax);
Wx = Rxx\Wx; 
%[Wy,lambday] = eig(RxxXYRyy'*RxxXYRyy); Wy = Ryy\Wy; 

%XYRyy = XY/Ryy;
%[Wx,lambdax] = eig(XX,XYRyy*XYRyy');
%RxxtXY = Rxx'\XY;
%[Wy,lambday] = eig(YY,RxxtXY'*RxxtXY);
%Wy = repop(YY\(XY'*Wx),'./',diag(lambdax));

%lambdax=diag(lambdax);
%lambday=diag(lambday);

% reshape the results back to input shape
Wx = reshape(Wx,[szX(1:dim(1)-1) 1 szX(dim(1)+1:end) size(Wx,2)]);
Wy = reshape(Wy,[szY(1:dim(2)-1) 1 szY(dim(2)+1:end) size(Wy,2)]);

% change output order when only 1 output
if (nargout<2) Wx=lambdax; end;
return;

%----------------------------------------------------------------------
function testCase()

% data-set with linear transformation between X and Y
X = randn(10,100);
A = randn(10,10);
Y = A*X;

[Wx,lambda,Wy,XX,YY]=cca(X,Y);
mimage(squeeze(Wy)'\squeeze(Wx)',A); % show we've found the mapping function
plot(squeeze(Wx)'*X,squeeze(Wy)'*Y,'.')