function [a,lbaii,ubaii,alpha,beta]=trinvBound(A,alpha,beta)
% upper bound on the trace of the inverse of A using the 
% Robinson Wathen bound: 
% Variational bounds on the entries of the inverse of a mtrix, IMA Journal of Numerical Analysis, 12 (1992) 463-486
[N,dim]=size(A);
if ( nargin < 2 ) alpha=[]; end  % [x,alpha]=smallestEig(A);
if ( nargin < 3 ) beta =largestEig(A); end;
if ( isempty(alpha) ) alpha=smallestEig(A,[],[],beta); end;
for i=1:size(A,1);
   sii(i)=sum(A(i,:).^2);
   lbaii(i)= 1/beta  + (A(i,i)-beta).^2 /(beta* (beta*A(i,i) -sii(i)));
   ubaii(i)= 1/alpha + (A(i,i)-alpha).^2/(alpha*(alpha*A(i,i)-sii(i)));   
end
a=sum(mean([lbaii;ubaii]));

% Alt method of -- seems to be worse
% article{ bai97bounds,
%     author = "Zhaojun Bai and Gene H. Golub",
%     title = "Bounds for the trace of the inverse and the determinant of symmetric positive matrices. The heritage of {P. L. Chebyshev}: a {Festschrift} in honor of the 70th birthday of {T. J. Rivlin}",
%     journal = "Ann. Numer. Math.",
%     volume = "4",
%     number = "1-4",
%     pages = "29--38",
%     year = "1997",
%     url = "citeseer.ist.psu.edu/bai96bounds.html" }

% mu1=trace(A);
% mu2=sum(sum(A.*A));
% lbtrinva=[mu1 N]*inv([mu1 mu2; beta^2 beta])*[N 1]';
% ubtrinva=[mu1 N]*inv([mu1 mu2; alpha^2 alpha])*[N 1]';
% a=mean([lbtrinva ubtrinva]);
