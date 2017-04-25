function [wb,J]=l2svm(K,Y,C,varargin);
% [alphab,J]=l2svm(K,Y,C,varargin)
% Quadratic Loss Support Vector machine using a pre-conditioned conjugate
% gradient solver so extends to large input kernels.
%
% J = C(1) w' K w + sum_i max(0 , 1 - y_i ( w'*K_i + b ) ).^2 
% 
% Inputs:
%  K       - NxN kernel matrix
%  Y       - Nx1 matrix of +1,-1 labels
%  C       - the regularisation parameter
%
% Outputs:
%  alphab  - (N+1)x1 matrix of the kernel weights and the bias b
%  J       - the final objective value
%
% Options:
%  alphab  - initial guess at the kernel parameters and bias
%  maxIter - max number of CG steps to do
%  tol     - absolute error tolerance
%  tol0    - relative error tolerance, w.r.t. initial gradient
%  verb    - verbosity
%  step    - initial step size guess
%  wght    - point weights [Nx1] vector of label accuracy probabilities
%            [2x1] for per class weightings
if ( nargin < 3 ) C=0; end;
opts=struct('alphab',[],'maxIter',inf,'tol',1e-6,'tol0',eps,'verb',0,...
            'step',0,'wght',[],'X',[],'ridge',1e-9,'nobias',0);
for i=1:2:numel(varargin); % process the options 
   if ( ~isstr(varargin{i}) || ~isfield(opts,varargin{i}) )
      error('Unrecognised option');
   else
      opts.(varargin{i})=varargin{i+1};
   end
end

[N,dim]=size(K);

% If the set of support vectors has changed, we need to
% reiterate.
iter = 0; sv=1:N; old_sv=[];
while ~isempty(setxor(sv,old_sv)) & (iter<opts.maxIter)
   old_sv = sv;
   H = K(sv,sv) + C(1)*eye(length(sv));
   H(end+1,:) = 1;                 % To take the bias into account
   H(:,end+1) = 1;
   H(end,end) = 0;
   
   % Find the parameters
   par = H\[Y(sv);0];
   beta = zeros(length(Y),1);
   beta(sv) = par(1:end-1);
   b = par(end);
   
   out = Y' .* (beta(sv)'*K(sv,:)+b); 
   sv = find(out < 1);

   wK  = beta(sv)'*K(sv,:);
   grad= 2*C(1)*wK + (Y(sv)'.*(1-out(sv)))*K(sv,:);
   
   obj = C(1)*wK(:,sv)*beta(sv) + sum(max(0,1-out).^2);
   iter = iter + 1;
   fprintf('%3d) J=%6g |dJ|=%6g nb of sv = %d\n',iter,obj,norm(grad),length(sv));
end;
fprintf('\n');
if iter == opts.maxIter
   warning(sprintf(['Maximum number of Newton steps reached. C(1) ' ...
                    'might be too small. Suggested value = %f'], ...
                   .1*(mean(diag(K))-mean(K(:)))));
elseif any(beta.*Y<0)
   warning('Bug');
end;
wb = [beta;b];
J  = obj;
%----------------------------------------------------------------------
function testCase();
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);

K=X'*X; % Simple linear kernel
fIdxs=gennFold(Y,10,'perm',1); trnIdx=any(fIdxs(:,1:9),2); tstIdx=fIdxs(:,10);
trnSet=find(trnIdx);
[alphab,J]=l2psvm(K(trnIdx,trnIdx),Y(trnIdx),1,'verb',2);
dv=K(tstIdx,trnIdx)*alphab(1:end-1)+alphab(end);
dv2conf(dv,Y(tstIdx))

% for linear kernel
alpha=zeros(N,1);alpha(trnSet)=alphab(1:end-1); % equiv alpha
w=X(:,trnIdx)*alphab(1:end-1); b=alphab(end); plotLinDecisFn(X,Y,w,b,alpha);
