function [z]=jf_imag2cart(z,varargin);
% convert complex features to a real-axis + imag-axis representation
% 
% [z]=jf_imag2cart(z,...)
%
% Options:
%  dim -- the dimension to cat the imaginary part along (ndims(X)+1)
opts=struct('dim',[],'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);

dim=opts.dim; if ( isempty(opts.dim) ) dim=ndims(z.X)+1; end;
dim=n2d(z.di,dim); % get the dim to work along

sz=size(z.X);
z.X = cat(dim,real(z.X),imag(z.X)); % cat along dim

odi=z.di;
if( dim<numel(sz) ) 
   z.di(dim).name = [z.di(dim).name '_reIm'];
   z.di(dim).vals = cat(1,z.di(dim).vals,z.di(dim).vals);
   summary = sprintf('into %s',sprintf('%ss ',odi(dim).name));
else
   z.di = z.di([1:end end]);
   z.di(dim) = mkDimInfo(2,1,'reIm',[],{'real','imag'});
   summary = sprintf('into dim #%d',dim);
end


info=[];
z = jf_addprep(z,mfilename,summary,opts,info);
return
%-------------------------------------------------------------------------
function testCase()
