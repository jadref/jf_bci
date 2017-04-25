function [I,xs,ys]=ffill(chull,xs,ys,padding)
% flood fill 
if ( nargin < 2 || isempty(xs) ) 
   xs=linspace(min(chull(1,:)),max(chull(1,:)),100);
end;
if ( nargin < 3 || isempty(ys) ) 
   ys=linspace(min(chull(2,:)),max(chull(2,:)),100);
end;
if ( nargin < 4 ) padding=0; end;

N=size(chull,2); if ( chull(:,end)==chull(:,1) ) N=N-1; end;
miny=min(chull(2,:)); ms=find(chull(2,:)==miny);
[ans,tmp]=min(chull(1,ms)); oil=ms(tmp); il=mod(oil-2,N)+1;   % bot-left
ml=(chull(1,il)-chull(1,oil))./(chull(2,il)-chull(2,oil));    % grad-left
[ans,tmp]=max(chull(1,ms)); oir=ms(tmp); ir=mod(oir,N)+1;     % bot-right
mr=(chull(1,ir)-chull(1,oir))./(chull(2,ir)-chull(2,oir));    % grad right
I=false(numel(xs),numel(ys));
for yi=find(ys>min(chull(2,:)),1,'first'):find(ys<max(chull(2,:)),1,'last');
   while ( ys(yi) > chull(2,il) ) % find next left crossing line
      oil=il; il=mod(il-2,N)+1; 
      ml=(chull(1,il)-chull(1,oil))./(chull(2,il)-chull(2,oil)); 
   end;
   while ( ys(yi) > chull(2,ir) ) % find next right crossing line
      oir=ir; ir=mod(ir,N)+1;
      mr=(chull(1,ir)-chull(1,oir))./(chull(2,ir)-chull(2,oir));
   end;
   xl = (chull(1,il)+(ys(yi)-chull(2,il))*ml);
   xli= find(xl>=xs,1,'last'); xli=max(1,xli-padding); % padding left
   xr = (chull(1,ir)+(ys(yi)-chull(2,ir))*mr);
   xri= find(xr<=xs,1,'first');xri=min(numel(xs),xri+padding); % padding right
   I(yi,xli:xri)=true; 
end
return;
%---------------------------------------------------------------------------
function testCase();
pts=rand(2,10);
C=convhull(pts(1,:),pts(2,:));
xs=linspace(0,1,100); ys=linspace(0,1,100);
I=ffill(pts(:,C),xs,ys);