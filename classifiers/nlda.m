function [alpha, b, xi, err, act] = nlda(y,K,C,type);
%  [alpha, b, xi, err, act] = nlda(y,K,C);
%
% Compute [[linear] sparse] kernel Fisher's discriminants.
%
% input:   y      column vector of +1/-1 labels (length m)
%          K      m x m kernel matrix for training examples
%          C      regularization constant (C>0)
%          type   specify which type of KFD to perform:
%                 'kfd'   - quadratic loss, quadratic regularizer
%                 'skfd'  - quadratic loss, linear    regularizer
%                 'lskfd' - linear    loss, linear    regularizer
%
% output:  alpha  column vector with coefficients for w (length m)
%          b      threshold from QP (should not be useed !!!)
%          xi     slack on each example
%          err    number of tr. errors using suboptimal threshold b
%          act    output of decision function, i.e. K*alpha+b
%
% As noted above, the returned threshold is not optimal and should not be
% used for classification (even if it is used in this small program to
% give an estimate of the generalization error). 
%
% To be compatible this implementation uses the matlab qp/lp
% optimizer. They are not very fast. Hence, replecing them with e.g. CPLEX
% (if you have a licence) will improve things. This implementation should
% work for about 100-500 samples, not more.
%
% Author: Sebastian Mika, Fraunhofer FIRST, (c) 2002
%
% LICENSE TERMS:
% --------------------------------------------------------------------
% This program is granted free of charge for research and education
% purposes. However you must obtain a license from Fraunhofer to use
% it for commercial purposes.
% Scientific results produced using the software provided shall
% acknowledge the use of optimization software provided by
% Sebastian Mika.
% 
% NO WARRANTY
% --------------------------------------------------------------------
% BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
% FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT
% WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER
% PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND,
% EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
% PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME
% THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  ell   = size(K,1);
  
  switch lower(type)
   case {'kfd'}
    
    obj   = eye(2*ell+1);
    for i=1:ell
      obj(i,i) = C;
    end
    obj(end,end) = 0 ;
    
    constr  = [K, -speye(ell), ones(ell,1)];
    c       = zeros(2*ell+1,1);
    l       = -inf * ones(2*ell+1,1);
    u       = inf  * ones(2*ell+1,1);
    
    [res,FVAL,EXITFLAG,OUTPUT,lambda] = ...
        quadprog(obj,c,[],[],constr,y,l,u);
    
    alpha       = res(1:ell);
    xi          = res(ell+1:2*ell);
    b           = res(end);
    
   case {'skfd'}
    
    obj   = zeros(3*ell+1);
    obj(2*ell+1:3*ell, 2*ell+1:3*ell) = speye(ell);
    
    constr  = [K, -K, -eye(ell), ones(ell,1)];
    c       = C * [ones(2*ell,1); zeros(ell,1); 0];
    l       = [zeros(2*ell,1); -inf * ones(ell+1,1)];
    u       = inf  * ones(3*ell+1,1);
    x0      = [];
    
    [res,FVAL,EXITFLAG,OUTPUT,lambda] = ...
	quadprog(obj,c,[],[],constr,y,l,u,x0);
    
    alpha       = res(1:ell);
    xi          = res(ell+1:2*ell);
    b           = res(end);
    
   case {'lskfd'}
    
    constr  = [K, -K, -speye(ell), speye(ell), ones(ell,1)];
    c       = [C * ones(2*ell,1); ones(2*ell,1); 0];
    l       = [zeros(4*ell,1); -inf];
    u       = inf  * ones(4*ell+1,1);
    
    [res,FVAL,EXITFLAG,OUTPUT,lambda] = ...
	linprog(c,[],[],constr,y,l,u);
    
    alpha  = res(1:ell) - res(ell+1:2*ell);
    xi     = res(2*ell+1:3*ell) - res(3*ell+1:4*ell);
    b      = res(end);

   otherwise
    error('Unknown type of kfd');
  end

  act    = K*alpha+b;
  err    = sum(sign(act) ~= y);



