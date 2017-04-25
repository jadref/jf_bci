function [z]=jf_tauERP(z,varargin);
opts=struct('dim',[],'taus',[],'taus_ms',[],'subIdx',[],'verb',[]);
opts=parseOpts(opts,varargin);

dim=opts.dim;
if ( isempty(dim) && isfield(z,'Ydi') ) % use meta-info to identify trial dims
   dim=n2d(z,{z.Ydi.name},0,0); dim(dim==0)=[];
end   

fs=getSampRate(z);
if( isempty(opts.taus) && ~isempty(opts.taus_ms) ) % allow spec of tau in milli-seconds
   if ( numel(opts.taus_ms)==2 ) % start-end
      opts.taus = ceil(opts.taus_ms*fs/1000); 
      opts.taus = opts.taus(1):opts.taus(2);
   else % exact set of relative times to use
      opts.taus = unique(ceil(opts.taus_ms*fs/1000));
   end
end

[Xmu,dim,featD,spD]=tauERP(z.X,z.Y,dim,opts.taus);

z.X=Xmu;
odi=z.di(1);
spDi=z.Ydi(spD); if (numel(spDi.vals)==1 && size(z.X,numel(featD)+1)==2) spDi.vals=[1 -1];end;
z.di=[z.di(featD); spDi; z.di(end)];
z.di(n2d(z,dim(1)))=mkDimInfo(size(z.X,n2d(z,dim(1))),1,'tau','ms',opts.taus*1000/fs);
z=jf_addprep(z,mfilename(),'',opts,info);
return;