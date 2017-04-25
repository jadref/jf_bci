function [minA,I]=mmin(A,dims)
% Multi-dimensional min function
%
% [minA,I]=mmin(A[,dims])
% Given an n-d input matrix compute the minimum value along the given set
% of dimensions and return the output
% e.g.  A=randn(10,10,10);  
%       [minA,I]=mmin(A,[1 2]); % for each value in dim 3 gets the
%                               % minvalue over the first 2 dimensions
%
%Inputs:
% A    -- n-d matrix of values to minimise over
% dims -- set of *consequetive* dimension indices over which to compute
%         the minimum, (N.B. negative values index from the last dimension)
%Outputs:
% minA -- [size(A) with size(dims)==1] matrix of minimum values over the
%         (dims) dimensions for the non-dims dimensions
% I   --  [size(A) with size(dims)==1] matrix of *linear* indicies into A
%         for the location of the minimum value over dims

% check dims is small enough consequetive sets inputs.
if ( nargin < 2 || isempty(dims) )  dims=find(size(A)>1,1); end;
dims=dims(:); % ensure is col vec
dims(dims<0)=dims(dims<0)+ndims(A)+1; % convert neg dims into +ve
szA=size(A); szA(end+1:max(dims))=1;
singD=(szA(dims)==1); dims(singD)=[]; % remove singlenton dims
if ( isempty(dims) || numel(dims)==ndims(A) ) 
   minA=A; I=reshape(1:numel(A),size(A)); return; 
end;
if ( min(size(dims)) > 1 || ...
     min(dims) < 1 || max(dims) > ndims(A) || any(diff(dims)~=1) )
   error('Dims must be vector of consequetive dims to minimise over');
end
% Reshape A into a [pre dims post] shape so we can min over dims
presz=szA(1:dims(1)-1); Adsz=szA(dims); postsz=szA(dims(end)+1:end);
% get the indexs of the optimal performance, over dims.
[minA,I]=min(reshape(A,[presz prod(Adsz) postsz 1]),[],numel(presz)+1);
% convert from index into (dims) into a direct index into A
preIs  =reshape(repmat([1:prod(presz)]',1,prod(postsz)),size(I));
postIs =reshape(repmat([1:prod(postsz)],prod(presz),1),size(I));
I      =sub2ind([prod(presz) prod(Adsz) prod(postsz)],preIs,I,postIs);
% convert into correctly sized output, which is size(A) with size(dims)=1
I   =reshape(I,   [presz ones(1,numel(dims)) postsz]);
minA=reshape(minA,[presz ones(1,numel(dims)) postsz]);

return

%-----------------------------------------------------------------------------
% Testcase
function []=testCase()
cv=cell2mat(agetfield(z.results.info.cv,'train'));meancv=msqueeze(mean(cv,2),1);
tst=cell2mat(agetfield(z.results.info.outer,'test'));
[idx,err]=cvPerf(meancv,2);
err,meancv(idx)  % check it worked
