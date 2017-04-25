function [yn]=relabel(y,classes)
yn=zeros(size(y,ndims(y)),1);
for c=1:numel(classes);
   for i=1:size(classes{c},2)
      yn(all(repop(y,classes{c}(:,i),'=='),1))=c;
   end
end