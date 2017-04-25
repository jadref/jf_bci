function axOut = packsubplots(ax, each_lrud, global_lrud, layout)
% AX = PACKSUBPLOTS(AX, LRUD_EACH, LRUD_GLOBAL)
% 
% LRUD_EACH is a four-element vector indicating the proportions of space
% (in the order left, right, up, down) that should surround each subplot.
% 
% LRUD_GLOBAL is a four-element vector indicating the proportions of space
% (in the order left, right, up, down) that should surround the whole group
% of subplots.
if nargin < 1, ax=findobj(gcf,'type','axes'); end;
if nargin < 3, global_lrud = []; end
if nargin < 2, each_lrud = []; end
if nargin < 4, layout = []; end


if isempty(each_lrud), each_lrud = [0]; end
if numel(each_lrud)==1, each_lrud = each_lrud([1 1 1 1]); end
if numel(each_lrud)==2, each_lrud = each_lrud([1 1 2 2]); end

if isempty(global_lrud), global_lrud = [0.05]; end
if numel(global_lrud)==1, global_lrud = global_lrud([1 1 1 1]); end
if numel(global_lrud)==2, global_lrud = global_lrud([1 1 2 2]); end

gw = 1 - sum(global_lrud(1:2));
gxo = global_lrud(1);

gh = 1 - sum(global_lrud(3:4));
gyo = global_lrud(4);

if isempty(layout) [nrows ncols] = size(ax); 
else nrows=layout(1);ncols=layout(2); end;
aw = gw/ncols;
w = (1 - sum(each_lrud(1:2))) * aw;
xo = each_lrud(1) * aw;

ah = gh/nrows;
h = (1 - sum(each_lrud(3:4))) * ah;
yo = each_lrud(4) * ah;

p = cell(size(h));
for i = 1:nrows
	y = gyo + (nrows-i) * ah + yo;
	for j = 1:ncols
		x = gxo + (j-1) * aw + xo;
		p{i, j} = [x y w h];
	end
end
p=p(:);

oldu = get(ax, {'units'});
set(ax, 'units', 'normalized')
set(ax(:), {'position'}, p(1:numel(ax)))
set(ax(:), {'units'}, oldu(:))
if nargout, axOut = ax; else figure(gcf), drawnow, end
