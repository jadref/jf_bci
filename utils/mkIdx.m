function [varargout]=mkIdx(sz,varargin)
% make indexing expresssion which indexs an entire matrix of size sz
%
% [idx]=mkIdx(sz[,dim1,idxrange1,dim2,idxrange2,...])
% Inputs:
%  sz  -- size of the matrix to index
%  dim?      -- dimension to use a sub-index along
%  idxrange? -- set of indices for this dimension
% Output
%  idx -- {1 x numel(sz)} cell array of index lists to use to index into
%         X. By default: idx = { 1:sz(1) 1:sz(2) ... } unless overridden. 
for i=1:numel(sz);
   idx{i}=1:sz(i);   
end
for i=1:2:numel(varargin);
   idx{varargin{i}}=varargin{i+1};
end
if ( nargout <= 1 ) 
   varargout{1}=idx; 
else 
   varargout=idx;
end
return;
%-------------------------------------------------------
function testCase()
nCh = 4; nSamp = 100, N = 10; sampDim=2;
X = randn(nCh,nSamp,N);
aa=mkIdx(size(X),sampDim,1:size(X,sampDim)/2);
X2 = X(aa{:}); size(X2);