function c = kelvin(m,basemap,alphamap,alpha)
% generate an alpha-blended colormap with m colors

if(nargin < 1 | isempty(m) ) m = size(get(gcf,'colormap'),1); end

if( nargin < 2 | isempty(basemap) ) basemap='gray'; end;
if( nargin < 3 | isempty(alphamap) ) alphamap='jet'; end;
if( nargin < 4 ) alpha=[-7 10]; end;

if( isstr(basemap) )  basemap=feval(basemap); end;
if( isstr(alphamap) ) alphamap=feval(alphamap); end;

if( numel(alpha)==2 ) 
   alpha=((1+exp(-linspace(alpha(1),alpha(2),m))).^-1)';
elseif ( numel(alpha)==3 ) 
   alpha=[linspace(alpha(1),alpha(2),floor(m/2)) ...
          linspace(alpha(2),alpha(3),m-floor(m/2))]';
end
c = repop(repop(basemap.^(.5),'*',1-alpha),'+',repop(alphamap,'*',alpha));
c = c./max(c(:));