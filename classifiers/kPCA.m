function [alpha,L,Knew,Ktestnew,Ktestvstest] = dualpca(K,Ktest,k)

%function [alpha,L] = dualpca(K,Ktest,k)
%
% Performs dual PCA
%
%INPUTS
% K = the kernel matrix of the training set (ell x ell)
% Ktest = the test kernel matrix ((ell+1) x t), containing
%         the kernel evaluations between test sample j and
%         training sample i at position (i,j), and where the
%         last row contains the kernel evaluations of the
%         samples with themselves
% k = the number of components
%
%OUTPUTS
% alpha = the k dual vectors (i.e., the training features)
% L = a column vector containing the corresponding variances
% Knew = the new kernel matrix based on these features
% Ktestnew = the new test kernel matrix based on these features
% Ktestvstest = the test versus test kernel matrix (t x t)
%               based on these features
%
%
%For more info, see www.kernel-methods.net

% K is the kernel matrix of the training points
% inner products between ell training and t test points 
%   are stored in matrix Ktest of dimension (ell + 1) x t
%   last entry in each column is inner product with self 
% k gives dimension of projection space 
% V is ell x k matrix storing the first k eigenvectors
% L is k x k diagonal matrix with eigenvalues
%
%
%For more info, see www.support-vector.net


ell = size(K,1);
D = sum(K) / ell;
E = sum(D) / ell;
J = ones(ell,1) * D;
K = K - J - J' + E * ones(ell, ell);
[V, L] = eigs(K, k, 'LM');
invL = diag(1./diag(L));         % inverse of L
sqrtL = diag(sqrt(diag(L)));     % sqrt of eigenvalues
invsqrtL = diag(1./diag(sqrtL)); % inverse of sqrtL
TestFeat = invsqrtL * V' * Ktest(1:end-1,:);
TrainFeat = sqrtL * V'; % = invsqrtL * V' * K;

% Note that norm(TrainFeat, 'fro') = sum-squares of 
%    norms of projections = sum(diag(L)).
% Hence, average squared norm not captured (residual) = 
%    (sum(diag(K)) - sum(diag(L)))/ell
% If we need the new inner product information:
Knew = V * L * V'; % = TrainFeat' * TrainFeat;

% between training and test
Ktestnew = V * V' * Ktest(1:end-1,:); 

% and between test and test
Ktestvstest = Ktest(1:end-1,:)'*V*invL*V'*Ktest(1:end-1,:);

% The average sum-squared residual of the test points is
(sum(Ktest(ell + 1,:) - diag(Ktestvstest)')/t

alpha=V;
L = diag(L);