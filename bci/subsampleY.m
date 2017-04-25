function [z]=subsampleY(z,idx,ydim)
% [z]=subsampleY(z,idx,ydim)
% 
% idx = **start** of the range to sub-sample from, i.e. y(i) = median(y(idx(i):idx(i+1)-1))
if( nargin<3 || isempty(ydim) ) ydim=-1; end;
if( isstruct(z) ) 
   ydim=n2d(z.Ydi,ydim);
   oY  = z.Y;
else
   oY  = z;
   if( ydim<0 ) ydim=ndims(oY)+ydim; end;
end
minmax=false;
if( sum(oY(:)==0) > .2*numel(oY) ) 
   minmax=true;
   warning('Need to sub-sample Y also... doing by taking extreeem values, events may be lost!');
end
% get integer version of the sample idxs
yidx=repmat({':'},ndims(oY),1); yidx{ydim}=idx;
Y   =oY(yidx{:});
for ei=1:numel(idx); % loop over the range getting the extreem values in each block      
   if(ei<numel(idx)) tr=idx(ei):idx(ei+1)-1; else tr=idx(ei):size(oY,ydim); end; % get range samples
   yidx{ydim}=tr; 
   Yei = oY(yidx{:}); 
   if( minmax )
      nYei= max(Yei,[],ydim); % extract extreem value
      minYei=min(Yei,[],ydim); 
      nYei(nYei==0|isnan(nYei))=minYei(nYei==0|isnan(nYei)); %replace 0/NaN's with min-value
   else % mean these values
      Yei(isnan(Yei))==0;
            nYei= sum(Yei,ydim)./sum(Yei~=0,ydim);
   end
   yidx{ydim}=ei; Y(yidx{:})=nYei; % insert back into labels
end
if( isstruct(z) )
   z.Y              = Y;
   z.Ydi(ydim).vals = z.Ydi(ydim).vals(idx);
else
   z = Y;
end
return;
function testCase();
Y=repmat(sign(randn(1,10)),10,1); % Y is in blocks of 10
Y=Y(:);
idx=1:10:numel(Y);
sY=subsampleY(Y,idx);
clf;plot(Y);hold on; plot(idx+median(diff(idx))/2,sY,'r');