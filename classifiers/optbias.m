function [mub]=optbias(Y,dv)
% compute the optimal chang in the bias from a given classifiers output
%
%  [mub]=optbias(Y,dv)
ind=Y~=0; Y=Y(ind); dv=dv(ind); % ignore 0 labelled points
[dv,si]=sort(dv); Y=Y(si); 
fp=cumsum(Y>0); fn=sum(Y<0)-cumsum(Y<0); % false pos/neg at given threshold
[ans,bi]=min(fp+fn); % min-error bias
if ( bi==1 || bi==numel(Y) ) % inverted labels?
   [ans,bi]=max(fp+fn); 
   if ( bi==1 || bi==numel(Y) ) mub=0; return; end;
   warning('It looks like the labels are inverted!');
end; 
% N.B. mean because this is max margin (in hard margin case!)
mub=-mean([dv(bi) dv(bi+1)]); 
return;
%---------------------------------------------------------------
function testCase();
Y = sign(randn(1000,1));
dv= Y+randn(size(Y)); % simulate LR style classifier
mub=optbias(Y,dv);