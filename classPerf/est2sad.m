function [cSp]=est2sad(Y,Yest)
% sum absolute deviation based measure of similarity of Y and it's estimate
%
%  [cSp]=est2corr(Y,Yest)
%
% sad = 1 - \sum_i abs(Y-Yest) ./ (sum_i abs(Y) + abs(Yest))
%
% Input
%  Y    - [N x nSp] target to fit with N examples
%  Yest - [nSp x N] estimate of the target with N examples
%        OR
%         [N x nSp]
% Output
%  c    - [float] correlation between Y and Yest
%  cSp  - [nSp x 1] correlation for each sub-problem independently
%
Y     = reshape(Y,[],size(Y,ndims(Y))); % [ N x nSp ]
if( size(Yest,1)~=size(Y,1) )
   if (size(Yest(:,:),2)==size(Y,2) ) Yest=Yest(:,:); % Yest=[nSp x N]
   elseif (size(Yest,1)==size(Y,2) )  Yest=Yest(:,:)';% Yest=[N x nSp] -> reshape to [N x nSp] 
   elseif (numel(Y)==size(Yest,1) )   Y   =Y(:);      % Y=[n-d x 1] -> [ N x 1 ]
   else warning('dv and Y dont seem compatiable');
   end
end
exInd = isnan(Y) | Y==0;% excluded points
Y(exInd(:))=0; Yest(exInd(:))=0; % ensure exclued points don't influence correlation compuation
% normalized sum absolute error
cSp   = sum(abs(Y(:,:)-Yest(:,:)),1);
cSp   = cSp./(sum(abs(Y(:,:)),1)+sum(abs(Yest(:,:)),1));
cSp   = 1 - cSp; % ensure larger is better...
%c     = mean(cSp);
%c     = corr(Y(~exInd(:)),Yest(~exInd(:)));
return;
%----------------------------------------------------------------------------
function testCase()
X = cumsum(randn(10,100,200));
A = mkSig(size(X,1),'exp',1); % strongest on 1st channel
Y = tprod(A,[-1],X,[-1 2 3]);
% add some outliers
Y = Y+(rand(size(Y))>.99)*20;
est2corr(Y(:),X(1,:,:))
est2sad(Y(:),X(1,:,:))
