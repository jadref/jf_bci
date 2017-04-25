function [w,b,alpha,neglogpost]=evidenceLRCG(X,Y,maxIter,alpha,verb)
% evidence based Logistic regression classification optimisation
cgOpts= struct('maxIter',inf,'maxEval',1000,'curveTol',1e-12,'plot',0, ...
               'verb',1,'pCond','hess');
if ( nargin < 4 ) alpha=0; end;
if ( nargin < 5 ) verb=0; end;
labels=unique(Y); oY=Y; Y(Y==labels(1))=0; Y(oY~=labels(1))=1;

persistent tcalled; if (isempty(tcalled)) tcalled=0; end;

% convert Y to P(Y==1)
Y=(Y>0); 

[N,dim]=size(X);
wb=ones(dim+1,1); 
% N.B. need the decision function to be near the center otherwise 
% everything buggers up!
% LS
sX=sum(X)';sY=sum(Y);w0=(X'*X - sX*sX'/N)\(X'*Y -sY*sX/N); b0=-sX'*w0/N;
wb=[w0;b0];
for i=1:maxIter;
   [wb f fs]=nonLinConjGrad(@(w) primalLRFn(w,X,Y,alpha),wb,cgOpts);
   [f df ddf H obj]=primalLRFn(wb,X,Y,alpha);
   alpha=dim/(2*obj(2)+sum(1./abs(ddf(1:dim)))); 
   neglogevidence=-(obj(1)-alpha*obj(2) + dim/2*log(max(alpha,eps))-log(det(H))/2);
   if(verb) 
      fprintf('%d) alpha=%g, logpost=%g, logevid=%g\n',i,alpha,f,neglogevidence); 
   end;
end
[wb f fs]=nonLinConjGrad(@(w) primalLRFn(w,X,Y,alpha),wb,cgOpts);

w=wb(1:end-1);
b=wb(end);
neglogpost=f;

return;

%-----------------------------------------------------------------------------
function []=testCase()
%TESTCASE 1) hinge vs. logistic (unregularised)
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
[w,b]=evidenceLRCG(X,Y,10);

[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5; -.2 -.5],[400 400 20 20],[.3 .3; .3 .3; .2 .2; .2 .2],[],[-1 1 1 -1]);
