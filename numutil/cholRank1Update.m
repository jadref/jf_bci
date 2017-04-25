function [chol,err]=cholRank1Update(chol,v)
% Compute the new cholesky factorisation after updating with v*v'

% Compute the factors needed to compute the new cholesky factorisation of
% the inner bits, i.e. \hat{L}\hat{L'}=eye+L^-1*v v'*L^-T
p=chol\v;                   % compute the inner vector, p=L^-1*v
ti=1;                       % rescaling factor = \prod_{j<i} \hat{L}_{i,i}
nLii=ones(size(chol,1),1);  % squared diag elements of \hat{L}_{i,i}^2
beta=zeros(size(chol,1),1); % p(i)/(nLii(i)*\prod_{j<i} \hat{L}_{i,i}(j))
nL=zeros(size(chol));       % the actual inner factorisation.
for i=1:size(chol,1);
   nLii2=1+p(i)*p(i)/ti; nLii(i)=sqrt(nLii2);
   beta(i)=p(i)/(nLii(i)*ti); 
   nL(i:end,i)=[nLii(i);beta(i)*p(i+1:end)]; % construct the inner chol fact
   ti=ti*nLii2;
end
%nL=tril(p*beta').*diag(nLii)+eye(size(nL));

% compute the new factorisation as: L*\hat{L}
% TODO: this can be done more efficiently w/o explicitly computing nL and 
% updating chol inplace
chol=chol*nL;
err=[];
