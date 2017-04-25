function [Xs Ys]=multiclass2subProb(X,Y,subProb)
% convert from a multi-class sub-problem specification in terms of the
% sub-problems class labels into a set of explicit sub-problems.
sizeX=size(X);
X=reshape(X,[],sizeX(end)); % make 2d
for i=1:numel(subProb)
   % set the desired set of labels
   pInd=any(repop(Y,subProb{i}{1}(:)','=='),2);
   nInd=any(repop(Y,subProb{i}{2}(:)','=='),2);   
   Xs{i}=X(:,pInd | nInd); 
   Xs{i}=reshape(Xs{i},[sizeX(1:end-1) size(Xs{i},2)]); % make n-d
   Ys{i}=Y; Ys{i}(pInd)=1; Ys{i}(nInd)=-1; Ys{i}=Ys{i}(pInd | nInd);
end
return;

%--------------------------------------------------------------------------
function []=testCase()
nProb=10; bs=randn(nProb,1);
[X,Y]=mkMultiClassTst([[-1+bs zeros(nProb,1)];1 0;.2 .5],[40*ones(1,nProb) 40 10],[ones(nProb,1)*[.3 .3];.3 .3;.2 .2],[],[1:nProb -1 -1]);
for i=1:nProb; subProb{i}={i -1}; end; subProb{nProb+1}={1:nProb -1};
[Xs Ys]=multiclass2subProb(X,Y,subProb);