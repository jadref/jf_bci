function WXXW = taucovWXXW(X,WB,bias)
% compute est power given set of time-shifted single-trial covariance matrices and spatio-temporal filter weights
%
%   WXXW = taucovWXXW(X,WB,bias)
%
% N.B.
%  [w0 w1 w2 ...] [XX0 XX1 XX2 ...] [w0]  = w0 XX0 w0 + w1 XX0 w1 + w2 XX0 w2 ...
%                 [XX1 XX0 XX1 ...] [w1]          + 2*( w0 XX1 w1 + w1 XX1 w2 ...)
%                 [XX2 XX1 XX0 ...] [w2]                      + 2*( w0 XX2 w2 ...)
%  = \sum_tau \sum_1 ((XX_tau * [w0 w1 w2 ...w_(end-tau)]) .* [w_tau w_tau+1 w_tau+2 ...])
if ( nargin<3 ) bias=0; end;
xch=1:size(X,1)-(bias>0);
bch=[]; if ( bias ) bch=size(X,1); end;
ych=(size(X,1)-(bias>0)+1):size(X,2);
% Compute the norm of the prediction for every trial
W3d  = reshape(WB(1:end-(bias>0),:),numel(xch),size(X,3),size(WB,2)); % [ ch_x x tau x ch_y ]
% w0XX_0w0 + w1XX_ow1 + ... = sum(XX_0*[w0 w1 w2...] .* [w0 w1 w2...],1)
WXXW =tprod(tprod(X(xch,xch,1,:),[1 -1 0 4],W3d,[-1 2 3]),[-1 -2 1 2],W3d,[-1 -2 1]);
for t=2:size(X,3); 
  WXXW=WXXW+2*tprod(tprod(X(xch,xch,t,:),[1 -1 0 4],W3d(:,1:end-t+1,:),[-1 2 3]),[-1 -2 1 2],W3d(:,t:end,:),[-1 -2 1]); 
end;
return;
