function [perm,allperm]=randpermrows(N,M)
% generate a random permutation of the rows only of a NxM matrix
perm=allpermutations(N);
pi  =ceil(rand(M,1)*size(perm,2)); % permutation per entry
perm=perm(:,pi); % index into with pi


