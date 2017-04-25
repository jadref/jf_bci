function [szstr]=dispDimInfo(di)
% Generate a simplified display of the information in the dim-info struct
if ( numel(di)==1 && isfield(di,'di') )di=di.di; end;
szstr=sprintf('%d %s',numel(di(1).vals),di(1).name);
if ( numel(di(1).vals) > 1 ) szstr=sprintf('%ss',szstr); end;
nDim=numel(di); if ( isempty(di(end).vals) || isempty(di(end).name) ) nDim=nDim-1; end;
for d=2:nDim;
   szstr=sprintf('%s x %d %s',szstr,numel(di(d).vals),di(d).name); 
   if ( numel(di(d).vals) > 1 ) szstr=sprintf('%ss',szstr); end;
end
szstr=sprintf('[%s]',szstr);
if( isempty(di(end).vals) )
   if(~isempty(di(end).units)) szstr=sprintf('%s of %s',szstr,di(end).units);
   elseif( ~isempty(di(end).name) ) szstr=sprintf('%s of %s',szstr,di(end).name); end
end