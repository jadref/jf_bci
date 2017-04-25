function varargout = ind2nsub(siz,ndx)
% ind2subv
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
k = [1 cumprod(siz)];%,[numel(ndx),1]);
ondx=ndx(:); ndx=ndx(:);
res=zeros(numel(ndx),numel(siz));
for id=numel(siz):-1:1;
   res(:,id)=floor((ndx-1)./k(id))+1; ndx=ndx-(res(:,id)-1)*k(id);
end
if ( nout == 1 ) 
   varargout={res};
else
   varargout=num2cell(res,1);
end
return;
%-----------------------------------------------------------
