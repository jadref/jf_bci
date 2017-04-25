function [x]=unitVec(x);
x=repop(x,'./',sqrt(sum(x.^2,1)));
return;