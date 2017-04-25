function [labels]=mkLabels(di)
sz=size(di);
labels=cell([sz 1]);
for i=1:numel(labels);
  li=''; 
  vals =di(i);
  if(iscell(vals) && ischar(vals{1})) val=vals{1};
  elseif ( isnumeric(vals) ) val=sprintf('%g',vals); 
  else                       val=disp(vals); 
  end    
  labels{i}=val;
end
return;
