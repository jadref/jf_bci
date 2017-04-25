function [perm]=allpermutations(N)
%1) generate the set of all possible permutations of N rows
nPerm = factorial(N);
perm=ones(N,nPerm,'int32');
idx=ones(N,1);
for i=1:nPerm;
  % convert from idx to perm
  pset=1:N;
  for d=1:N;
    perm(d,i)=pset(idx(d)); pset(idx(d))=[];
  end
  % increment the idx counter
  for d=1:N;
    idx(d)=idx(d)+1;
    if(idx(d)<=N-d+1) break; else idx(d)=1; end;
  end
end
