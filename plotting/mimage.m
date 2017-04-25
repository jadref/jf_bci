function []=mimage(varargin)
% multiple imagesc plots
% Options:
% 'layout' -- [w h]
% 'clim'   -- 'limits' or [cmin,cmax]  ('minmax')
% 'title',[],'xlabel',[],'ylabel',[]
% 'diff' -- [bool] compute the diff between inputs
% 'divide' -- [bool] compute the ratio between inputs
% 'transpose' -- [bool] transpose the matrices
% 'dispType' -- one-of: 'image','plot','mcplot'  ('image')
opts=struct('layout',[0 0],'clim','minmax','disptype','image','dispType',[],...
            'title',[],'xlabel',[],'ylabel',[],...
            'colorbar',0,'diff',0,'absdiff',0,'divide',0,'transpose',0);
[opts,varargin]=parseOpts(opts,varargin);
if( numel(varargin)==1 && ~iscell(varargin{1}) && isnumeric(varargin{1}) ...
    && ndims(varargin{1})>2 )
  varargin=num2cell(varargin{1},[1 2]);
end
if (~isempty(opts.dispType)) opts.disptype=opts.dispType; end;
if ( opts.diff ) varargin{end+1}=varargin{end}-varargin{end-1}; end;
if ( opts.absdiff ) varargin{end+1}=abs(varargin{end}-varargin{end-1}); end;
if ( opts.divide ) varargin{end+1}= varargin{end}./varargin{end-1}; end;
N=numel(varargin);

if ( ~iscell(opts.xlabel) ) opts.xlabel={opts.xlabel}; end;
if ( ~iscell(opts.ylabel) ) opts.ylabel={opts.ylabel}; end;
if ( opts.layout(2) > 0 ) w=opts.layout(2); else w=floor(sqrt(N)); end;
if ( opts.layout(1) > 0 ) h=opts.layout(1); else h=ceil(N/w); end;
if ( ischar(opts.clim) ) %isequal(opts.clim,'limits') || isequal(opts.clim,'minmax') || isequal(opts.clim,'cent0') ) 
   rng=[inf -inf];
   for i=1:N; 
      if ( ~isempty(varargin{i}(:)) )
         rng=[min(min(real(varargin{i}(:))),rng(1))...
              max(max(real(varargin{i}(:))),rng(2))];
      end
   end
   switch opts.clim;
    case {'limits','minmax'}; opts.clim=rng;
    case 'cent0'; opts.clim = max(abs(rng))*[-1 1]; 
  end
end
for i=1:N;
   if(isempty(varargin{i})) continue; end;
   subplot(h,w,i);
   mx=[squeeze(real(varargin{i}))];
   if( opts.transpose ) mx=mx'; end;
   if ( ~isreal(varargin{i}) )  mx=[real(mx) squeeze(imag(varargin{i}))]; end
   switch lower(opts.disptype);
    case {'image','imagesc'};   
     imagesc('cdata',mx(:,:));
     if(~isempty(opts.clim))    set(gca,'clim',opts.clim); end;
    case {'imaget','imagesct'};   
     imagesc('cdata',mx(:,:)');
     if(~isempty(opts.clim))    set(gca,'clim',opts.clim); end;
    case 'plot'; 
     plot(mx(:,:));
     if(~isempty(opts.clim))    set(gca,'ylim',opts.clim); end;
    case 'mcplot'; mcplot(mx(:,:));     
   end
   if(~isempty(opts.title) )  title(opts.title{i}); end;
   if(~isempty(opts.xlabel) ) xlabel(opts.xlabel{min(i,end)}); end;
   if(~isempty(opts.ylabel) ) ylabel(opts.ylabel{min(i,end)}); end;
   if( opts.colorbar ) colorbar; end;
end
