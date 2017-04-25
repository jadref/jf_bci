function [c,cSp]=avedv(Y,Yest)
% average decision value
%
%  [c,cSp]=avedv(Y,Yest)
%
% Input
%  Y    - [nY x N] target to fit with N examples (ignored)
%  Yest - [N x nY] similarity of the estimated Y and the true Y
%        OR
%         [N x nSp]
% Output
%  c    - [float] correlation between Y and Yest
%  cSp  - [nY x 1] correlation for each sub-problem independently
Y     = reshape(Y,[],size(Y,ndims(Y))); % [ N x nSp ]
if( size(Yest,1)~=size(Y,1) )
   if (size(Yest(:,:),2)==size(Y,2) ) Yest=Yest(:,:); % Yest=[nSp x N]
   elseif (size(Yest,1)==size(Y,2) )  Yest=Yest(:,:)';% Yest=[N x nSp] -> reshape to [N x nSp] 
   elseif (numel(Y)==size(Yest,1) )   Y   =Y(:);      % Y=[n-d x 1] -> [ N x 1 ]
   else warning('dv and Y dont seem compatiable');
   end
end
exInd = isnan(Y) | Y==0;% excluded points
Y(exInd(:))=0; Yest(exInd(:))=0; % ensure exclued points don't influence computation
cSp   = sum(Yest,1)./sum(~exInd,1);
c     = mean(cSp);
return;
