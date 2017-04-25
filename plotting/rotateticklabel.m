function th=rotateticklabel(h,rot,varargin)
%ROTATETICKLABEL rotates tick labels
%   TH=ROTATETICKLABEL(H,ROT) is the calling form where H is a handle to
%   the axis that contains the XTickLabels that are to be rotated. ROT is
%   an optional parameter that specifies the angle of rotation. The default
%   angle is 90. TH is a handle to the text objects created. For long
%   strings such as those produced by datetick, you may have to adjust the
%   position of the axes so the labels don't get cut off.
%
%   Of course, GCA can be substituted for H if desired.
%
%   Known deficiencies: if tick labels are raised to a power, the power
%   will be lost after rotation.
%
%   See also datetick.

%   Written Oct 14, 2005 by Andy Bliss
%   Copyright 2005 by Andy Bliss

%set the default rotation if user doesn't specify
if( nargin<2 || isempty(rot) ) rot=90; end
if( nargin<1 || isempty(h) ) h=gca; end;
rot=mod(rot,360);

for ai=1:numel(h);
   axes(h(ai)); % make this plot current
   %get current tick labels
   a=get(h(ai),'XTickLabel');
   if ( isempty(a) ) continue; end;
   %erase current tick labels from figure
   set(h(ai),'XTickLabel',[]);
   xlabel(h(ai),[]);
   %get tick label positions
   b=get(h(ai),'XTick');
   c=get(h(ai),'YTick');
   %make new tick labels
   if rot<180
      th=text(b',repmat(c(1)-.1*(c(2)-c(1)),length(b),1),a,'HorizontalAlignment','right','rotation',rot,varargin{:});
   else
      th=text(b',repmat(c(1)-.1*(c(2)-c(1)),length(b),1),a,'HorizontalAlignment','left','rotation',rot,varargin{:});
   end
end
