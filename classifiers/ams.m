function [sig,C,alpha,b] = ams(X,Y,method,sphere,opt)
%   AMS  Automatic Model Selection for Kernel Methods with
%            - RBF Gaussian kernel
%            - L2 penalization of the errors
%          For classification: Support Vector Machines
%          For regression: Kernel Ridge Regression
%
%  Usage: [sig,C,alpha,b] = AMS(X,Y,method,[sphere],[opt])
%
%    X: is an n times d matrix containing n training points in d dimensions  
%    Y: is a n dimensional vector containing either the class labels
%       (-1 or 1) in classification or the targets values in regression.  
%    method is either:
%       'loo': minimizes the leave-one-out error. The LOO is an accurate
%              estimate of the generalization error but might be difficult
%              to minimize (not smooth). Also, do not use with if sphere = 0
%              and large d (long time for gradient calculations).
%       'rm':  radius / margin bound. Easier to minimize. That can be the
%              first thing to try
%       'val': minimizes a validation error. Good if there are a lot
%              a training points. The validation points are the
%              nv points of the training set [nv specified in the options]
%       'ev':  evidence maximization.
%    sphere: if 1, spherical RBF kernel = exp(-0.5*||x-y||^2 / sigma^2)
%            if 0, one sigma per component:
%               exp(-0.5*sum (x_i-y_i)^2/sigma_i^2)
%            [default 1]
%    opt: structure containing the options (in brackets default value)
%       itermax: maximum number of conjugate gradient steps [100]    
%       tolX: gradient norm stopping criterion [1e-5]
%       hregul: regularization on the hyperparameters (to avoid extreme values) [0.01]
%       nonorm: if 1, do no not normalize the inputs [0]
%       verb: verbosity [1]
%       eta: smoothing parameter used in classif by 'ev' and 'loo'
%            larger is smoother [0.1]
%       sigmoid: slope of the sigmoid used by 'val' and 'loo' in classif [5] 
%       nv: number of validation points for 'val' [n/2]
%
%
%    sig: the sigma(s) found by the model selection. Scalar if 
%         sphere = 1, d dimensional vector otherwise
%    C: the constant penalizing the training errors
%    alpha,b: the parameters of the prediction function
%      f(x) = sum alpha_i K(x,x_i) + b
  
%    Copyright Olivier Chapelle
%              olivier.chapelle@tuebingen.mpg.de
%              last modified 21.04.2005


  % Initialization
  [n,d] = size(X);
  if (size(Y,1)~=n) | (size(Y,2)~=1)
    error('Dimension error');
  end;
    
  if all(abs(Y)==1)
    classif = 1;      % We are in classification
  else
    classif = 0;      % We are in regression
  end;
  
  if nargin < 5       % Assign the options to their default values
    opt = [];
  end;
  if ~isfield(opt,'itermax'), opt.itermax = 100;                end;
  if ~isfield(opt,'length'),  opt.length  = opt.itermax;        end;
  if ~isfield(opt,'tolX'),    opt.tolX    = 1e-5;               end;
  if ~isfield(opt,'hregul'),  opt.hregul  = 1e-2;               end;
  if ~isfield(opt,'nonorm'),  opt.nonorm  = 0;                  end;
  if ~isfield(opt,'verb'),    opt.verb    = 0;                  end;
  if ~isfield(opt,'eta'),     opt.eta     = 0.1;                end;
  if ~isfield(opt,'sigmoid'), opt.sigmoid = 5;                  end;
  if ~isfield(opt,'nv'),      opt.nv      = round(n/2);         end;
  
  
  if nargin<4
    sphere = 1;
  end;
  
  % Normalization
  if ~opt.nonorm & (norm(std(X)-1) > 1e-5*d)
    X = X./repmat(std(X),size(X,1),1);
    fprintf(['The data have been normalized (each component is divided by ' ...
             'its standard deviation).\n Don''t forget to do the same for the ' ...
             'test set.\n']);
  end;
  
  % Default values of the hyperparameters (in log scale)
  C = 0;
  sig = .3*var(X)';
  if ~sphere,
    sig = log(d*sig) / 2;
  else
    sig = log(sum(sig)) / 2;
  end;
  
  % Do the optimization on the hyperparameters
  param = [sig; C];
  default_param = param;
%  check_all_grad(param,X,Y,classif,default_param,opt); return;
%  plot_criterion(X,Y,param,[-3:0.1:3],classif,default_param,opt);
%  return;
  [param,fX] = minimize(param,@obj_fun,opt,X,Y,method,classif,default_param,opt);
  
  % Prepare the outputs
  [alpha, b] = learn(X,Y,param,classif);
  C = exp(param(end));
  sig = exp(param(1:end-1));

function plot_criterion(X,Y,param,range,classif,opt)
  for i=1:length(range)
    fprintf('%f\r',range(i));
    for j=1:2
      param2 = param;
      param2(end-j+1) = param2(end-j+1)+range(i);
      obj(1,i,j) = obj_fun(param2,X,Y,'loo',classif,opt);
      obj(2,i,j) = obj_fun(param2,X,Y,'rm', classif,opt);
      obj(3,i,j) = obj_fun(param2,X,Y,'val',classif,opt);
      obj(4,i,j) = exp(obj_fun(param2,X,Y,'ev', classif,opt));
    end;
  end;
  figure; plot(obj(:,:,1)');
  figure; plot(obj(:,:,2)');

function  check_all_grad(param,X,Y,classif,opt)
  param = randn(size(param));
  checkgrad(@obj_fun,param,1e-6,X,Y,'loo',classif,opt)
  checkgrad(@obj_fun,param,1e-6,X,Y,'rm', classif,opt)
  checkgrad(@obj_fun,param,1e-6,X,Y,'val',classif,opt)
  checkgrad(@obj_fun,param,1e-6,X,Y,'ev', classif,opt)

  
function [obj, grad] = obj_fun(param,X,Y,meth,classif,default_param,opt)
% Compute the model selection criterion and its derivatives with
% respect to param which is a vector containing in log scale sigma
% and C

  % Add a regularization on the hyperparameters to avoid that
  % they take extreme values
  obj0 = opt.hregul * mean((param-default_param).^2);
  grad0 = 2*opt.hregul * (param-default_param) / length(param);

  if obj0 > 1               % Extreme value of hyperparamaters. The rest
    obj = obj0;             % might not even work because of numercial
    grad = grad0;           % instabilities -> exit
    return;
  end
    
  if strcmp(meth,'val')
    % Remove the validation set for the learning
    [alpha,b] = learn(X(1:end-opt.nv,:),Y(1:end-opt.nv),param,classif);
  else
    [alpha,b] = learn(X,Y,param,classif);
  end;

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Compute the objective function %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  sv = find(alpha~=0);            % Find the set of support vectors
  
  switch lower(meth)
    %
    % Leave-one-out 
    %
   case 'loo'
    K = compute_kernel(X(sv,:),[],param);
    K(end+1,:) = 1;                 % To take the bias into account
    K(:,end+1) = 1;
    K(end,end) = 0;
    D = opt.eta./abs(alpha(sv));
    invKD = inv(K+diag([D;0]));
    invKD(:,end) = []; invKD(end,:) = [];
    span = 1./diag(invKD) - D;      % "Span" of the support vectors
    % Estimated change of the output of each point when it's
    % removed from the training set.
    change = alpha(sv).*span;
    if classif
      out = 1-Y(sv).*change;
      obj = sum(tanh(-opt.sigmoid*out)+1)/2/length(Y);
    else
      obj = mean(change.^2);
    end;
    invK = inv(K); invK(:,end) = []; invK(end,:) = [];
    
    %
    % Radius margin
    %
   case 'rm'
    K = compute_kernel(X,[],param);
    w2 = sum(alpha.*Y);
    if classif
      R2 = mean(diag(K)) - mean(mean(K)); % Radius estimated by variance
    else
      K(end+1,:) = 1;                 % Radius = mean span
      K(:,end+1) = 1;
      K(end,end) = 0;
      invK = inv(K);
      span = 1./diag(invK);        
      R2 = mean(span(1:end-1));
    end;
    obj = R2*w2 / length(Y);
    
    %
    % Validation set
    %
   case 'val'
    K = compute_kernel(X(1:end-opt.nv,:),X(end-opt.nv+1:end,:),param);
    Yv = Y(end-opt.nv+1:end);
    out = K(end-opt.nv+1:end,:)*alpha+b;   % Output on the validation points
    if classif
      % Goes through sigmoid because step function is not differentiable
      obj = mean(tanh(-opt.sigmoid*out.*Yv)+1)/2;
    else
      obj = mean((out-Yv).^2);
    end;
    Kl = K(sv,sv);
    Kl(end+1,:) = 1;               
    Kl(:,end+1) = 1;
    Kl(end,end) = 0;
    invK = inv(Kl);
 
    
    %
    % Evidence maximization
    %
   case 'ev'
    K = compute_kernel(X,[],param);
    if classif
      % Normally, K should be computed only on the support vectors. But
      % then, the evidence wouldn't be continuous. We thus make a soft
      % transition for small alphas.
      weight = min(0, (alpha.*Y)/(mean(alpha.*Y))/opt.eta - 1);
      weight2 = 1 - weight.^2;
      K0 = K;
      K = K.*(max(weight2*weight2',eye(length(weight2))));
    
      Kl = K0(sv,sv);                    
      Kl(end+1,:) = 1;               
      Kl(:,end+1) = 1;
      Kl(end,end) = 0;
      invKl = inv(Kl); invKl(:,end) = []; invKl(end,:) = [];
    end;
    [invK, logdetK] = inv_logdet_pd_(K);
    obj = logdetK/length(K) + log(mean(alpha.*Y));
        
   otherwise
    error('Unknown model selection criterion');
  end;
  

  %%%%%%%%%%%%%%%%%%%%%%%%%
  % Compute the gradients %
  %%%%%%%%%%%%%%%%%%%%%%%%%
  
  for i=1:length(param)
    
    switch lower(meth)
      %
      % Leave-one-out 
      %
     case 'loo'
      dK = compute_kernel(X(sv,:),[],param,i);
      dalpha = -invK * (dK*alpha(sv));
      dD = - opt.eta * dalpha.*sign(alpha(sv))./alpha(sv).^2;
      dspan = diag(invKD).^(-2).* ...
              diag(invKD*(dK+diag(dD))*invKD) - dD;
      dchange = dalpha.*span + alpha(sv).*dspan;
      if classif
        dout = -Y(sv).*dchange;
        grad(i) = -0.5*opt.sigmoid * ...
                  sum(cosh(-opt.sigmoid*out).^(-2).*dout)/length(Y);
      else
        grad(i) = 2*mean(change.*dchange);
      end;
    
      %
      % Radius margin
      %
     case 'rm'
      dK = compute_kernel(X,[],param,i);
      dw2 = -alpha'*dK*alpha;
      if classif
        dR2 = mean(diag(dK)) - mean(mean(dK));
      else
        dspan = diag(invK).^(-2).*diag(invK(:,1:end-1)*dK*invK(1:end-1,:));
        dR2 = mean(dspan(1:end-1));
      end;
      grad(i) = (dR2*w2 + R2*dw2) / length(Y);
    
      %
      % Validation set
      %
     case 'val'
      dK = compute_kernel(X(1:end-opt.nv,:),X(end-opt.nv+1:end,:),param,i);
      dalpha = -invK(:,1:end-1) * (dK(sv,:)*alpha);
      db = dalpha(end);
      dalpha = dalpha(1:end-1);
      dout =  K(end-opt.nv+1:end,sv)*dalpha + db + ...
             dK(end-opt.nv+1:end,:) * alpha; 
      if classif
        grad(i) = -0.5*opt.sigmoid * ...
                  mean(cosh(-opt.sigmoid*out.*Yv).^(-2).*dout.*Yv);
      else
        grad(i) = 2*mean((out-Yv).*dout);
      end;
    
      %
      % Evidence maximization
      %
     case 'ev'
      dK = compute_kernel(X,[],param,i);
      dK0 = dK;
      if classif 
        dalpha = zeros(size(alpha));
        dalpha(sv) = -invKl * (dK(sv,:)*alpha);
        dweight = -2/opt.eta * weight.* ...
            (dalpha.*Y/mean(alpha.*Y) - (alpha.*Y) * mean(dalpha.*Y)/mean(alpha.*Y)^2);
        dK = dK .* max(weight2*weight2',eye(length(weight))) + ...
             (K0-diag(diag(K0))) .* (weight2*dweight' + dweight*weight2');
  
      end;
      grad(i) = invK(:)'*dK(:)/length(K) - ...
                alpha'*dK0*alpha / sum(alpha.*Y);
    end;
  end;
  obj = obj0 + obj;
  grad = grad0 + grad';
  
    
function [alpha,b] = learn(X,Y,param,classif)
% Do the actual learning (SVM or kernel ridge regression)
  K = compute_kernel(X,[],param);
  svset = 1:length(Y);
  old_svset = [];
  iter = 0;
  itermax = 20;
  
  % If the set of support vectors has changed, we need to
  % reiterate. Note that for regression, all points are support
  % vectors and there is only one iteration.
  while ~isempty(setxor(svset,old_svset)) & (iter<itermax)
    old_svset = svset;
    H = K(svset,svset);
    H(end+1,:) = 1;                 % To take the bias into account
    H(:,end+1) = 1;
    H(end,end) = 0;
    % Find the parameters
    par = H\[Y(svset);0];
    alpha = zeros(length(Y),1);
    alpha(svset) = par(1:end-1)';
    b = par(end);
    % Compute the new set of support vectors
    if classif
      % Compute the outputs by removing the ridge in K
      out = K*alpha+b - alpha/exp(param(end)); 
      svset = find(Y.*out < 1);
    end;
    iter = iter + 1;
  end;
  if iter == itermax
    warning('Maximum number of Newton steps reached');
  elseif classif & any(alpha.*Y<0)
    warning('Bug in learn');
  end;
    
function K = compute_kernel(X,Y,param,diff)
% Compute the RBF kernel matrix between  [X;Y] and X.
% if diff is given, compute the derivate of the kernel matrix
%  with respect to the diff-th parameter of param.  
  
  global K0   % Keep the kernel matrix in memory; we'll need to
              % compute the derivatives.
  
  Z = [X; Y];
  Y = X; X = Z; clear Z;
  ridge = exp(-param(end));
  
  if nargin<4    % Compute the kernel matrix
    
    % Divide by the length scales
    if length(param)==2
      X = X * exp(-param(1));
      Y = Y * exp(-param(1));
    else
      X = X .* repmat(exp(-param(1:end-1)'),size(X,1),1);
      Y = Y .* repmat(exp(-param(1:end-1)'),size(Y,1),1);
    end;
    
    normX = sum(X.^2,2);
    normY = sum(Y.^2,2);
    
    K0 = exp(-0.5*(repmat(normX ,1,size(Y,1)) + ...
                  repmat(normY',size(X,1),1) - 2*X*Y'));
    K = K0 + eye(size(K0))*ridge;

  else  % Compute the derivate
    if diff == length(param)    % With respect to log(C)
      K = - ridge*eye(size(K0));
    elseif length(param) == 2   % With respect to spherical log(sig)
      K = -2*K0.*log(K0+eps);
    else                        % With respect to log(sig_i)
      dist = repmat(X(:,diff),1,size(Y,1)) - ...
             repmat(Y(:,diff)',size(X,1),1);
      K = K0 .* (dist/exp(param(diff))).^2;
    end;
  end;

function [invA, logdetA] = inv_logdet_pd_(A)
  % For those who don't have the inv_logdet_pd written by Carl
  if exist('inv_logdet_pd')
    [invA, logdetA] = inv_logdet_pd(A);
  else
    invA = inv(A);
    C = chol(A);
    logdetA = 2*sum(log(diag(C)));
  end;
  
    
