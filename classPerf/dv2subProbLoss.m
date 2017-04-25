function [bin,pc]=dv2subProbLoss(Y,dv)
% test the resulting classifiers on each others sub-problems
%
% [bin,pc]=dv2subProbLoss(Y,dv)
%
% Inputs:
%  Y  - [n-d] set of true labels
%  dv - [n-d] set of decision values
%
% Outputs:
%  bin  [L x L] matrix of binary losses
%  pc - [2 x L x L] matrix of per class losses.
%        pc(1,:,i) = positive class error-rate for classifer i on all subProbs
%        pc(2,:,i) = negative class error-rate for classifer i on all subProbs 
if ( size(Y,1)~=size(dv,1) ) 
   error('decision values and targets must be the same size');
end
for i=1:size(Y,2); % loop over different sub-probs (Y's)
   for j=1:size(dv,2); % loop over classifiers (dv's)
      conf(:,i,j)= dv2conf(Y(:,i),dv(:,j));
   end
end
