function [foldIdxs]=loofold(Y,dim)
% generate a leave one out folding of the Y
% 
%  [foldIdxs]=loofold(Y,dim)
%
% Inputs:
%  Y -- [n-d x nSp] set of labels and sub-problems
%  dim -- [int] set of dimension indicators
%         dim(1:end-1) -- dimensions to loo over
%         dim(end)     -- sub-problem dim
% Outputs:
%  foldIdxs -- [size(Y,dim(1:end-1)) nFold] set of fold indicators
if( nargin < 2 || isempty(dim) ) dim=ndims(Y); end;
szY=size(Y); if(dim(end)>numel(szY)) szY(end+1:dim(end))=1; end; % empty sub-prob
nFold=prod(szY(dim(1:end-1)));
fsz=szY; fsz(setdiff(1:ndims(Y),dim(1:end-1)))=1; fsz(end+1)=nFold;
foldIdxs=-ones(fsz,'int32');
idx={};for d=1:ndims(Y); idx{d}=1:size(foldIdxs,d); end; %idx{dim(2:end)}=1;
for i=1:nFold; 
   [idx{dim}]=ind2sub(fsz(dim),i); idx{end}=i;
   foldIdxs(idx{:})=1; 
end;
return;
%-------------------------------------------------------------------
function testCase()
t=loofold(rand(10,10,10),[1 3])
t=loofold(rand(10,10,10),[1 2 3]);