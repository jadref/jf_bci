function [wb,f,J]=libsvm(K,Y,C,varargin);
% wrapper code for call to libSVM to train classifier
%
% [alphab,f,J]=libsvm(K,Y,C,varargin)
% 
% Options:
%  libSVMoptstr -- additional options to pass to libsvm ('-t 4 -s 0')
% libsvm options (v. 2.83-1)
% 
% -s svm_type : set type of SVM (default 0)
%         0 -- C-SVC
%         1 -- nu-SVC
%         2 -- one-class SVM
%         3 -- epsilon-SVR
%         4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
%         0 -- linear: u'*v
%         1 -- polynomial: (gamma*u'*v + coef0)^degree
%         2 -- radial basis function: exp(-gamma*|u-v|^2)
%         3 -- sigmoid: tanh(gamma*u'*v + coef0)
%         4 -- precomputed kernel (kernel values in training_set_file)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/k)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train an SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C in C-SVC (default 1)
% -v n: n-fold cross validation mode
% 
% The k in the -g option means the number of attributes in the input data.
% 
% option -v randomly splits the data into n parts and calculates cross
% validation accuracy/mean squared error on them.

opts=struct('libSVMoptstr','-t 4 -s 0');

eqreg = [nan nan];
N = numel(Y);
wb = zeros(N+1, 1);
trnInd = (Y~=0); trnIdx=find(trnInd);
Ktrn = K(trnInd,trnInd); Ytrn = Y(trnInd);
if ( isa(K,'single') ) Ktrn=double(Ktrn); end;
if ( isa(Y,'single') ) Ytrn=double(Ytrn); end;
%try
	model = svmtrain(Ytrn, [(1:sum(trnInd))' Ktrn], sprintf('%s -c %.8g',opts.libSVMoptstr,1/max(C,eps)));
	wb(trnIdx(model.SVs)) = model.sv_coef * Ytrn(1);
   wb(end)               = -model.rho * Ytrn(1);
%end
equiv.nu = eqreg(1);
equiv.C = eqreg(2);

% compute final decision values.
wK= wb(1:end-1)'*K;
err  = 1-Y'.*(wK+wb(end)); svs=err>=0 & Y'~=0;

f = wK + wb(end); f = reshape(f,size(Y));
J = C*wK*wb(1:end-1) + err(svs)*err(svs)';



