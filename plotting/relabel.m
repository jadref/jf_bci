function []=relabel(labels,dir,ax)
if ( nargin < 3 ) fig=gca; end;
if ( isequal(dir,'X') )
   tickIdx=floor(get(gca,'XTick')); 
   tickIdx=tickIdx(tickIdx>0 & tickIdx<numel(labels));
   set(gca,'XTick',tickIdx,'XTickLabel',labels(tickIdx),...
           'XTickMode','manual','XTickLabelMode','manual');
else
   tickIdx=floor(get(gca,'YTick'));
   tickIdx=tickIdx(tickIdx>0 & tickIdx<numel(labels));
   set(gca,'YTick',tickIdx,'YTickLabel',labels(tickIdx),...
           'YTickMode','manual','YTickLabelMode','manual');
end
