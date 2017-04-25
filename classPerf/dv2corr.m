function [c]=est2corr(Y,Yest)
Y     = reshape(Y,[],size(Y,ndims(Y)));
Yest  = reshape(Yest,[],size(Yest,ndims(Yest)));
exInd = any(isnan(Y),1) || all(Y==0,1);% excluded points
Y     = Y(:,~exInd);
Yest  = Yest(:,~exInd);
c     = corrcoef(Y,Yest);
return;
