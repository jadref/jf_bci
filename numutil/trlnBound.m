function [a,lbtrlna,ubtrlna,alpha,beta]=trlnBound(A,alpha,beta)
% Compute bounds on the trace(ln(A)) using the bounds in the paper:
%  
% article{ bai97bounds,
%     author = "Zhaojun Bai and Gene H. Golub",
%     title = "Bounds for the trace of the inverse and the determinant of symmetric positive matrices. The heritage of {P. L. Chebyshev}: a {Festschrift} in honor of the 70th birthday of {T. J. Rivlin}",
%     journal = "Ann. Numer. Math.",
%     volume = "4",
%     number = "1-4",
%     pages = "29--38",
%     year = "1997",
%     url = "citeseer.ist.psu.edu/bai96bounds.html" }
[N,dim]=size(A);
if ( nargin < 2 ) alpha=[]; end  % [x,alpha]=smallestEig(A);
if ( nargin < 3 ) beta =largestEig(A); end;
if ( isempty(alpha) ) alpha=smallestEig(A,[],[],beta); end;
mu1=trace(A);
mu2=sum(sum(A.*A));
tlb=(alpha*mu1-mu2)/(alpha*N-mu1);
tub=(beta *mu1-mu2)/(beta *N-mu1);
lbtrlna=[log(alpha) log(tlb)]*inv([alpha tlb; alpha^2 tlb^2])*[mu1 mu2]';
ubtrlna=[log(beta)  log(tub)]*inv([beta  tub; beta^2  tub^2])*[mu1 mu2]';

a=mean([lbtrlna ubtrlna]);