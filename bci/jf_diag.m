function z=jf_diag(z,varargin)
% extract the diagonal entries from the first 2 dimensions of z.X
% 
%  z=jf_diag(z,varargin)
%
% Options:
%  dim -- dimensions to extract diag from
%  verb
%  subIdxx
opts=struct('dim',[1 2],'verb',0,'subIdx',[]);
opts=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);
if ( ~isequal(dim(:),[1;2]) )
   error('Only supported for first 2 dims at the moment');
end

szC=size(z.X);
X=zeros(szC([1 3:end]),class(z.X));
if ( opts.verb>0 ) fprintf('diag:'); end;
for ci=1:prod(szC(3:end));
   if (opts.verb>0 ) textprogressbar(ci,prod(szC(3:end))); end;
   Cci=z.X(:,:,ci);
   X(:,ci) = Cci(1:szC(1)+1:szC(1)*szC(2));
end
if ( opts.verb>0 ) fprintf('\n'); end;
z.X=X;
z.di=z.di([1 3:end]);
z =jf_addprep(z,mfilename,'',opts,[]);
return;
