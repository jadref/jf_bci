function [z]=jf_merge(zz,varargin)
% merge a pair of results sets
%  [z]=jf_merge(zz,varargin)
z=zz{1};
for zi=2:numel(zz); % loop over data-sets
   for eli=1:numel(zz{zi}.X);
      [pos{1:numel(z.di)-1}]=ind2sub(size(zz{zi}.X),eli); growp=false(1,ndims(z.X));
      for d=1:numel(zz{zi}.di)-1; % get the values for this array elm and try to match to z1
         val=zz{zi}.di(d).vals(pos{d});
         if ( isnumeric(val) ) mi=find(val==z.di(d).vals);
         elseif( iscell(val) && ischar(val{1}) ) mi=strmatch(val{:},z.di(d).vals,'exact');
         else 
            mi=[];for j=1:numel(z.di(d).vals);if(isequal(val,z.di(d).vals(j)))mi=j;break; end; end
         end
         if( ~isempty(mi) ) npos{d}=mi; else npos{d}=size(z.X,d)+1; growp(d)=true; end
      end
      % add the new elemenent and update the dim-info
      z.X(npos{:})=zz{zi}.X(eli);
      for d=find(growp); 
         z.di(d).vals(npos{d})=zz{zi}.di(d).vals(pos{d}); 
%          if ( ~isempty(z.di(d).extra) && ~isempty(zz{zi}.di(d).extra) ) 
%             z.di(d).extra(npos{d})=mergestruct(z.di(d).extra(npos{d}),zz{zi}.di(d).extra(pos{d})); 
%          end;
      end;
   end
end
