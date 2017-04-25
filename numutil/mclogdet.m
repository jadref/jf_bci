function logdet=mcLogDet(C,pp,kk,beta)
% input arguments: covariance matrix C,
% (optional) the number of trace seeds selection pp,
% (optional) the number of power-series terms kk,
% (optional) the ensemble adjusting factor beta.
%
% from the paper: Zhang, Y., and W. E. Leithead. “Approximate Implementation of the Logarithm of the Matrix Determinant in Gaussian Process Regression.” Journal of Statistical Computation and Simulation 77, no. 4 (April 1, 2007): 329–48. doi:10.1080/10629360600569279.

if nargin < 2
    pp = 20;
end
if nargin < 3
    kk = 50;
end
if nargin < 4
    beta = 1;
end
 
N = size(C,1);
Cn=norm(C,inf);
A=C/Cn;
if issparse(C)
    B=speye(N)-A;
else
    B=eye(N)-A;
end
 
clear C A;
traceBt=trace(B);
traceB2t=traceAxB(B,B); % a simple routine to get trace(B 2 ) with N 2 operations
deltaB122=inf;
for iii=1:pp % trace seeds selection for pp times
    xxt=randn(N,1);
    cct=B*xxt;
    xxBNxx=xxt'*cct;
    xxTxxt=xxt'*xxt;
    traceBe=N*xxBNxx/xxTxxt;
    deltaBt=traceBt-traceBe;
    cct=B*cct;
    xxBNxx=xxt'*cct;
    traceB2ee=N*xxBNxx/xxTxxt;
    deltaB2e=traceB2t-traceB2ee;
    deltaB122new=deltaBt+deltaB2e/2;
    if abs(deltaB122new)<=abs(deltaB122), % have found a better seed
        xx=xxt;
        cc=cct;
        xxTxx=xxTxxt; % save the workspace related to the new seed
        traceB1e=traceBe;
        traceB2e=traceB2ee;
        deltaB=deltaBt;
        deltaB2=deltaB2e;
        deltaB122=deltaB122new;
    end
end
clear xxt cct;
rhotr=deltaB2/deltaB;
traceB=traceB1e; traceB2=traceB2e;
if abs(rhotr)>=1, % disp(‘trace rho greater than or equal to 1?');
    compens2=deltaB/(1-rhotr/2); % a more robust yet loose trace-seed compensation asin Remark 3
elseif rhotr==0,
    compens2=0; % no need compensation as i = 0
else
    compens2=-deltaB*log(1-rhotr)/rhotr; % the standard trace-seed compensation as in Remark 3
end;
vv=traceB+traceB2/2;
preadd=traceB2; % save the computation of the first two power-series terms
compens1=0;
wthla1=0;
wthla2=0;
wthla3=0;
for jjj=3:kk, % start the computation from the third power-series term to the kth term
    cc=B*cc;
    xxBNxx=xx'*cc;
    newadd=N*xxBNxx/xxTxx;
    if preadd==0, % only for numerical-robustness consideration
        compens1=0; wthl=vv-compens1-compens2;
        logdet2=N*log(Cn); logdet=-wthl+logdet2;
        return
    end
    lambda=newadd/preadd;
    if lambda<1,
        sumlambdaik=0; lambdai=1; ndt5=0;
        for iii=1:jjj, % computing the second term of the truncation-error compensation as in Proposition 5
            lambdai=lambdai*lambda;
            ndt5=ndt5+lambdai/(jjj+iii);
            sumlambdaik=sumlambdaik+lambdai/iii;
        end
        if lambdai==0,
            compens1=-newadd*ndt5; % truncation-error compensation via (8)
        else
            compens1=newadd*(log(1-lambda)+sumlambdaik)/lambdai;
            % the final truncation-error compensation as in Proposition 5
        end
    end;
    preadd=newadd;
    newadd=newadd/jjj;
    vv=newadd+vv;
    wthl=vv-compens1-compens2; % for the finite sequence with the ensemble effect as in Proposition 6
    wthla1=wthla2; wthla2=wthla3; wthla3=wthl;
    % save the last three terms of such a sequence, i.e, k-2 , k-1 and k
end;
if lambda>=1,
% disp(‘final lambda greater than or equal to 1?');
    compens1=0; % only for numerical-robustness consideration
end
    aaa=wthla1;
    bbb=wthla2-wthla1;
    qqq=(wthla3-wthla2)/bbb;
if bbb==0,
    % disp(‘The last three terms of Gamma-sequence are equal: converge well');
else
    if abs(qqq)>= 1, % only for numerical-robustness consideration
        % disp(‘Gamma-sequence coefficient greater than or equal to 1?');
    else
        wthl=aaa+bbb*qqq/(1-qqq); % Geometric series based re-estimation as in Proposition 6
    end
end
logdet2=N*log(Cn);
logdet=-wthl+logdet2;
if logdet/max(abs([wthl,logdet2]))<1e-2
    wthl=wthl*1;
elseif bbb/aaa>1e-4 % use beta to adjust the final ensemble effect if necessary
    wthl=wthl*beta; % in general, beta:=1
end
logdet=-wthl+logdet2; % return the final logdet approximation
clear B; % the routine is thus complete
end
 
function r = traceAxB(A,B)
    r = sum(sum(A'.*B,1),2);
end