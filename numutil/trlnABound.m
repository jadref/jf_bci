function [a,lbtrlna,ubtrlna]=trlnBound(A)
% upper bound on the trace of the inverse of A using the 
% Robinson Wathen bound: 
% Variational bounds on the entries of the inverse of a mtrix, IMA Journal of Numerical Analysis, 12 (1992) 463-486
[N,dim]=size(A);
[x,beta]=largestEig(A);
alpha=eigs(A,1,'SM'); % [x,alpha]=smallestEig(A);
mu1=trace(A);
mu2=sum(sum(A.*A));
tlb=(alpha*mu1-mu2)/(alpha*N-mu1);
tub=(beta *mu1-mu2)/(beta *N-mu1);
lbtrlna=[log(alpha) log(tlb)]*inv([alpha tlb; alpha^2 tlb^2])*[mu1 mu2]';
ubtrlna=[log(beta)  log(tub)]*inv([beta  tub; beta^2  tub^2])*[mu1 mu2]';

a=mean([lbtrlna ubtrlna]);