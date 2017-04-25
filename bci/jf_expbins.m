function z=jf_expbins(z,varargin);
% map a dimension of z from linear to linearly increasing bin sizes
opts=struct('dim','freq','divfactor',5,'avebin',0);
opts=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);
binvals=z.di(dim(1)).vals;
fact=opts.divfactor;
fIs=false(numel(binvals)); idx=1; 
while (idx<numel(binvals)) 
  fIs(idx)=true; 
  idx = idx + find(binvals(idx+1:end)>binvals(idx)+max(1,round(binvals(idx)/fact)),1); 
end;
fIs= find(fIs); if(fIs(end)<numel(binvals))fIs(end)=numel(binvals); end;
mx = zeros(numel(binvals),numel(fIs)-1); 
vals={}; 
for j=1:numel(fIs)-1;  
  mx(fIs(j):fIs(j+1)-1,j)=1;  
  vals{j}=sprintf('%g-%g',binvals(fIs(j:j+1))); 
end
if ( opts.avebin ) mx=repop(mx,'./',sum(mx,1)); end;
z.X=tprod(z.X,[1:dim(1)-1 -dim(1) dim(1)+1:ndims(z.X)],mx,[-dim(1) dim(1)]);
z.di(dim(1)).vals=vals;
z=jf_addprep(z,mfilename,sprintf('over %s',z.di(dim(1)).name),opts,struct('mx',mx));
return;
