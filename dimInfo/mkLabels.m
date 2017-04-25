function [labels]=mkLabels(di)
% make element labels from dim-info structure
%
% [labels]=mkLabels(di)
%
for d=1:numel(di); sz(d)=numel(di(d).vals); end;
if( isempty(di(end).name) && isequal(di(end).vals,1) ) sz(end)=[]; end;
labels=cell([sz 1]);
ii=ones(numel(sz),1);
for i=1:numel(labels);
  li=''; 
  for d=1:numel(sz); % generate the label
    vals =di(d).vals;
    units=di(d).units;
    if (numel(vals)==1 ) continue; end; % don't bother for fixed parameters
    if(iscell(vals))           val=vals{ii(d)}; 
    elseif ( isnumeric(vals) ) val=sprintf('%g',vals(ii(d))); 
    else                       val=disp(vals(ii(d))); 
    end    
    if ( d>1 ) li=[li ',']; end;
    li=[li ' ' val units];
  end
  labels{i}=li;
  % update the index
  for d=1:numel(sz);
    if( ii(d)<sz(d) ) ii(d)=ii(d)+1; break; else ii(d)=1; end; 
  end
end
return;
%-----------------------------------------------
function testCase()
di=mkDimInfo([3 10 10],'ch',[],{'a' 'b' 'c'},'time','ms',1:10,'epoch',[],1:10);
mkLabels(di)