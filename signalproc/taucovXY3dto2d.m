function cov=taucovXY3dto2d(cov,bias,taus)
% convert from [ch x ch_2 x tau x ep] to block matrix
% [ch x ch_2 x tau x ep] = [  [ch x ch_2 x 1]   ] 
%                          [  [ch x ch_2 x 2]'  ] 
%                          [  [ch x ch_2 x 3]'  ] 
%                          [         .          ]
%                          [ [ch x ch_2 x tau]' ]
%                          [ [ch x ch_s x 1]' ] (optional bias)
if(nargin<2||isempty(bias)) bias=false; end;
if ( nargin<3 || isempty(taus) ) taus=0:size(cov,3)-1; end;
if(bias<=0)
  cov=reshape(permute(cov,[1 3 2 4]),[size(cov,1)*size(cov,3),size(cov,2),size(cov,4)]);
else
  ti=1; if ( ~isempty(taus) ) [ans,ti]=min(abs(taus-0)); end;
  biascov=cov(end,:,ti,:); % only take the tau=0 bias term
  cov    =cov(1:end-1,:,:,:);
  cov    =reshape(permute(cov,[1 3 2 4]),[size(cov,1)*size(cov,3),size(cov,2),size(cov,4)]);
  cov    =cat(1,cov,biascov);
end
return;
