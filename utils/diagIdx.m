function [idx]=diagIdx(N,k)
% function [idx]=diagIdx(N,k)
% Function to compute the indices of the k'th diagonal of a NxM matrix
% N.B. positive is above the main diagonal, -ve below

% get the matrix size we want
if ( max(size(N))==1 ) M=N;  % scalar (therefore square matrix)
elseif ( min(size(N))==1 ) M=N(2); N=N(1); % vector of sizes
else M=size(N,2); N=size(N,1); 
end
if ( nargin < 2 ) k=0; end;
k=-k;
if ( k>=0 ) idx=k+1    + (N+1)*[0:min(N-k,M)-1];
else        idx=-k*N+1 + (N+1)*[0:min(N,M+k)-1];
end
idx=int32(idx); % effic hack
return;
%---------------------------------------------------------------------------
function []=testCases;
R=zeros(10,10);R(diagIdx(size(R)))=1;imagesc(R);
R=zeros(10,10);R(diagIdx(size(R),-1))=1;imagesc(R);
R=zeros(10,10);R(diagIdx(size(R),+1))=1;imagesc(R);
R=zeros(10,4);R(diagIdx(size(R),+1))=1;imagesc(R);
R=zeros(10,4);R(diagIdx(size(R),-1))=1;imagesc(R);