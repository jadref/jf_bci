function varargout = ind2nsub(siz,ndx)
% ind2nsub
% 
% Modified version of ind2sub such that if given a single output argument
% returns an array containing the dimension indicies for each dim in ndx.
% N.B. input ndx must be a vector!
nout = max(nargout,1);
siz = double(siz);

if nout > 1 
   if length(siz)<=nout, % size of nargout indicates how many dims to compute
      siz = [siz ones(1,nout-length(siz))];
   else
      siz = [siz(1:nout-1) prod(siz(nout:end))];
   end
end
k = repmat([1 cumprod(siz)],[numel(ndx),1]);
res=repmat(ndx(:),[1,numel(siz)]);
res=ceil((rem(res-1,k(:,2:end))+1)./k(:,1:end-1));
if ( nout == 1 ) 
   varargout={res};
else
   varargout=num2cell(res,1);
end
