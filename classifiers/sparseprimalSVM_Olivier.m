function [beta,b,S,obj,te,time] = sparse_primal_svm(Y,Yt,ker,kerparam,C,opt)
% [BETA,B,S] = SPARSE_PRIMAL_SVM(Y,YT,KER,KERPARAM,C,OPT)
% Approximates the SVM solution by expanding it on a small set of basis functions  
%
% Y is the target vecor (+1 or -1, length n)
% YT is a test vector (length nt)
% KER is a function handle to a kernel function of the form
%    K = MY_KERNEL(IND1,IND2,KERPARAM) computing the kernel submatrix
%    between the points with indices IND1 and IND2. KERPARAM is an
%    additional argument containing for instance kernel
%    parameters. Indices are supposed to be between 1 and n+nt, an indice
%    larger than n corresponding to a test point.
% KERPARAM: see above
% C is the constant penalizing the training errors
% OPT is a structure containing the following optional fields
%    nb_cand (aka kappa): the number of candidates at each iteration
%    set_size (aka dmax): the final number of basis functions
%    maxiter: the maximum number of iterations for Newton
%    base_recomp: the solution is recomputed every base_recomp^p
%    verb: verbosity
%
% BETA is the vector of expansion coefficients
% B is the bias
% S contains the indices if the expansion (same size as BETA)
%  
% [BETA,B,S,OBJ,TE,TIME] = SPARSE_PRIMAL_SVM(Y,YT,KER,KERPARAM,C,OPT)
% OBJ,TE contains the objective function and the test error after
%    each retraining
% TIME contains the time spent between each retraining (it does not
%    include the time for computing the kernel test matrix).  
  
  global K hess sv
  
  if nargin<6
    opt = []; 
  end;
  n = length(Y);
  
  % Set the parameters to their default value
  if ~isfield(opt,'nb_cand'),     opt.nb_cand     = 10;           end;
  if ~isfield(opt,'set_size'),    opt.set_size    = round(n/100); end;
  if ~isfield(opt,'maxiter'),     opt.maxiter     = 20;           end;
  if ~isfield(opt,'base_recomp'), opt.base_recomp = 2^0.25;       end; 
  if ~isfield(opt,'verb'),        opt.verb        = 1;            end; 

  % Memory allocation for K (size n times dmax) and Kt (size nt times dmax). 
  % The signed outputs are computed as K*[b; beta]. That's why the first
  % column is Y (and just 1 for Kt)
  K = zeros(n,opt.set_size+1); 
  K(:,1)=Y;
  Kt = zeros(length(Yt),opt.set_size+1);
  Kt(:,1)=1;
  
  % hess is the Cholesky decompostion of the Hessian
  hess = sqrt(C*n)*(1+1e-10);
  
  % At the beginning x (which contains [b; beta]) equal 0 and all points
  % are training errors. The set sv is set of points for which y_i f(x_i) < 1
  x = 0;
  sv = 1:n;
  
  S = [];
  te = [];
  time = [];
  obj = [];
  tic;
  
    
  while 1  % Loops until all the basis functions are selected

    if retrain(length(S),opt) % It's time to retrain ...
      d0 = size(hess);
      
      update_hess(S,Y,C,ker,kerparam); % First, compute the news columns
                                       % and K and update the Hessian and
                                       % its Cholesky decomposition
      [x,out,obj2] = train_subset(C,opt); % And then do a Newton optimization
      out = out(sv);
      obj = [obj obj2];
      time = [time toc];
      
      if ~isempty(Yt) % Compute the new test error
        nnt = length(S)-d0;
        if nnt>=0
          Kt(:,d0+1:d0+nnt+1) = feval(ker,n+[1:length(Yt)],S(end-nnt:end),kerparam);
        end;
        te = [te mean(Yt.*(Kt(:,1:length(x))*x)<0)];
        if opt.verb > 0, fprintf('  Test error = %.4f',te(end)); end;
      end;
      tic;
    end;
    
    if length(S)==opt.set_size % We're done !
      break;
    end;
    
    % Chooses a random subset for new candidates (exclude the points
    % which are already in the expansion)
    candidates = 1:n; 
    candidates(S) = [];
    candidates = candidates(randperm(length(candidates)));
    candidates = candidates(1:opt.nb_cand);
    
    [ind,x,out] = choose_next_point(candidates,S,x,out,Y,ker,kerparam,C,opt);
    S = [S candidates(ind)];
  end;

  beta = x(2:end);
  b = x(1);
  if opt.verb>0, fprintf('\n'); end;
    
function [x,out,obj] = train_subset(C,opt)
  global K hess sv
  
  if opt.verb>0, fprintf('\n'); end;
  iter = 0;
  d = length(hess);
  while iter < opt.maxiter
    iter = iter + 1;
    % Take a few Newton step (no line search). By writing out the
    % equations, this simplifies to following equation:
    x = C*(hess \ (hess' \ sum(K(sv,1:d),1)'));
    out = K(:,1:d)*x;      % Recompute the outputs...

    new_sv = find(out'<1); % ... and identify the errors
    
    % The set of errors has changed (and so the Hessian). We update the
    % Cholesky decomposition of the Hessian
    for i=setdiff(new_sv,sv)
      hess = cholupdate(hess,sqrt(C)*K(i,1:d)','+');
    end;
    for i=setdiff(sv,new_sv)
      hess = cholupdate(hess,sqrt(C)*K(i,1:d)','-');
    end;
    
    % Compute the objective function (just for debugging)
    obj = 0.5* (norm(hess*x)^2 - 2*C*sum(out(sv)) + C*length(sv));
    
    if opt.verb>0
      fprintf(['\rNb basis = %d, iter Newton = %d, Obj = %.2f, ' ...
               'Nb errors = %d   '],length(hess)-1,iter,obj,length(sv));
    end;
    if isempty(setxor(sv,new_sv)) % No more changes -> stop
      break;
    end;
    sv = new_sv;
  end;
 
function update_hess(S,Y,C,ker,kerparam)
  global K hess sv
  
  d  = length(S);
  d0 = length(hess) - 1; 
  if d==d0, return; end;

  % Compute the new rows of K corresponding to the basis that have been added
  K(:,d0+2:d+1) = feval(ker,1:length(Y),S(end-(d-d0-1):end),kerparam) ...
      .* repmat(Y,1,d-d0);
  
  h = [zeros(1,d-d0); K(S,d0+2:d+1).*repmat(Y(S),1,d-d0)] + ...
      C * K(sv,1:d+1)' * K(sv,d0+2:d+1);

  % The new Hessian would be [[old_hessian h2]; [h2' h3]]
  h2 = h(d0+2:end,:);
  h2 = h2 + 1e-10*mean(diag(h2))*eye(size(h,2)); % Ridge is only for numerical reason
  h3 = hess' \ h(1:d0+1,:);
  h4 = chol(h2-h3'*h3);
  % New Cholesky decomposition of the augmented Hessian
  hess = [[hess h3]; [zeros(d-d0,d0+1) h4]];  
  
function [select,x,out] = choose_next_point(candidates,S,x,out,Y,ker,kerparam,C,opt)
  global K hess sv
  % When we choose the next basis function, we don't do any retraining
  % and assume that everyting is quadratic and that the other weights are fixed
  
  n = length(Y);
  K2 = feval(ker,sv,candidates,kerparam).*repmat(Y(sv),1,length(candidates));
  K3 = feval(ker,S, candidates,kerparam);
  Kd = feval(ker,candidates,[],kerparam);
  % If the point candidate(i) would be added as a basis function, the
  % first and second derivative with respect to its weight would be g(i) and h(i)
  h = Kd + C*sum(K2.^2,1)';
  g = K3'*x(2:end,:) + C*K2'*(out-1);
  score = g.^2./h; % Newton decrement
  [max_score, select] = max(score); % The larger the better
  if max_score<1e-8
    warning('No good basis function');
  end;
  x = [x; -g(select)/h(select)]; % Still assuming that the other weights
                                 % are fixed, the estimated weight of the
                                 % new basis function is g/h 
  
  out = out + K2(:,select)*x(end); % Update the outputs
  
function r = retrain(d,opt)
% Check if we should retrain  
 b = opt.base_recomp;
 if d<2, r=1; return; end;
 r = (floor(log(d)/log(b)) ~= floor(log(d-1)/log(b)));

% We always retrain at the end
 r = r | (d==opt.set_size);
