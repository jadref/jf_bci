function [outFold,inFold]=ind2fold(Ys,balanceInp,balanceOutp)
% construct the fold information
balYs=balanceYs(Ys); oYs=Ys;
if ( balanceInp )  Ys=balYs; end;
for i=1:size(Ys,2);
   if ( balanceInp ) 
      inInd=balYs(:,i)~=0 ;
   else
      inInd=oYs(:,i)~=0;
   end
   if ( balanceOutp ) 
      outInd = any(balYs(:,[1:i-1 i+1:end])~=0,2);
   else
      outInd = any(oYs(:,[1:i-1 i+1:end])~=0,2);
   end
   
   outFold{i}=find(outInd); % train Set
   inFold{i} =find(inInd);  % test  Set
end
outFold=outFold'; inFold=inFold'; % so is over folds
return;
%----------------------------------------------------------------------------
function []=testCase()