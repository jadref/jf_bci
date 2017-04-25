function [K,dK,ddK]=linKerFn(hp,fat,X,Z)
% Function to compute a linear kernel between the input X
% !!!!N.B. assumes X has examples as the *LAST* input dimension!!!!
if ( nargin < 4 ) Z=[]; end;
K=tprod(X,Z,[-[1:ndims(X)-1] 1],[-[1:ndims(X)-1] 2]);
if ( nargout > 1 )
   dK=zeros([size(K),numel(hp)]);
   if ( nargout > 2 ) ddK=dK; end;
end
return;
%-----------------------------------------------------------------------------
function testCases()
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 20],[.3 .3; .3 .3; .2 .2],[],[1 -1 -1]);
labScatPlot(X,Y);

[f,df,ddf]=linKerFn(1,[],X,@idFeatFn);
jf_checkgrad(@(x) linKerFn(x,[],X,@idFeatFn),1,1e-5);

