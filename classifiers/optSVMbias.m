function [b]=optSVMbias(Y,dv)
% compute the optimal bais from a given classifiers output
ind=Y~=0; Y=Y(ind); dv=dv(ind); Y=reshape(Y,size(dv));

% % iterative refinment optimiser
b=optbias(Y,dv); mJ=inf;
for i=1:10;
   err= 1-Y.*(dv+b); 
   J  = sum(err(err>0).^2);
   %fprintf('b=%g J=%g dJ=%g\n',b,J,b+sum(dv(err>0)-Y(err>0)));
   if ( J > mJ ) 
      b=(2*mb+b)/3;  % approx Golden Ratio value
   else 
      mb=b; mJ=J; b =-sum(dv(err>0)-Y(err>0)); 
   end;   
   if ( abs(b-mb)<1e-3 | mJ==0 ) break; end;
end
b=mb;

return;
%---------------------------------------------------------------
function testCase();
Y = sign(randn(1000,1));
dv= Y+randn(size(Y)); % simulate LR style classifier
mub=optbias(Y,dv);

% Another way of computing the opt bias
for i=1:numel(Y);
   b=-dv(i);
   err  = 1-Y.*(dv+b);
   J(i) = sum(err(err>0).^2);
   dJ(i)= b+sum(dv(err>0)-Y(err>0));
end
[mJ,bi]=min(J); % min-error bias
% % N.B. mean because this is max margin (in hard margin case!)
b=-mean(dv(bi)); 
