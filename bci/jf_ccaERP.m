function [z]=jf_ccaERP(z,varargin)
% transform the data
opts=struct('dim',[],'ydim','symb','subIdx',[]); % ignore subIdx as we ignore Y==0
[opts,varargin]=parseOpts(opts,varargin);
dim=opts.dim;
if ( isempty(dim) && isfield(z,'Ydi') ) dim=z.Ydi(1).name; 
else dim=-1;
end
Y=z.Y;
if ( ~isempty(opts.subIdx) ) % enforce subIdx
   yidx=n2d(z,{z.Ydi.name},0,0); yidx(yidx==0)=[];
   Y=zeros(size(z.Y));
   Y(opts.subIdx{yidx},:)=z.Y(opts.subIdx{yidx},:);% force ONLY these elements to have no label info
end
[R,Xr,Xmu,opts]=ccaERP(z.X,Y,'dim',n2d(z,dim),'ydim',n2d(z,opts.ydim),varargin{:});
z.X=Xr;
z.di(n2d(z,'ch')).name='ch_cca';
z.di(n2d(z,'ch')).vals={};
for ci=1:size(Xr,n2d(z,'ch'));
  z.di(n2d(z,'ch')).vals{ci}=sprintf('cca_%02d',ci);
end
z=jf_addprep(z,mfilename,'',opts,struct('R',R,'Xmu',Xmu));
return;
%--------------------
function testCase();
z=jf_mkoverlapToy();
zd=jf_deconv(z,'irflen_samp',64);
zc=jf_ccaERP(zd);
clf;jf_plotERP(zc)
