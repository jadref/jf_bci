function [confMx]=pred2conf(Y,dv,dims)
% convert label predictions to confusion matrices
% Inputs:
%  Y    - [N x nClass] matrix of target labels, with class labels in *last* dimension
%  dv   - n-d matrix of predicted values with N trial in dims(1) and nClass
%         in dims(end)
%  dims - [2 x 1] the dimensions of dv along which the trials lie and then
%         the dimension of dv along which the different class precitions lie
%           [trDim spDim]
%         ([1st non-singlenton & next])
% Outputs:
%  conf - [size(dv) with size(dv,dims(1))=nClass*nClass & size(dv,dims(2))==1]
%      with dimension dims of dv replaced by a nClass*nClass elemet conf
%      vector where, in the binary case:
%      conf(1) = # true  positive    conf(3) = # false negative
%      conf(2) = # false positive    conf(4) = # true  negative 
%
%                                                      true-P true-N
% i.e. reshape(conf(:,i),nClass,nClass))=>     pred-P [  tp     fn  ] 
%                                              pred-N [  fp     tn  ]
%      for classifier i (assuming dims=[1 3])
if ( nargin < 3 || isempty(dims) ) 
   dims = find(size(dv)>1,1); if ( isempty(dims) ) dims=1; end; 
end;
if ( any(dims < 0) ) dims(dims<0) = ndims(dv)+dims(dims<0)+1; end;
if ( numel(dims)==1 ) if(size(Y,2)>1) dims(2)=dims+1;else dims(2)=0; end; end;
if ( size(Y,2)==size(dv,dims(1)) ) Y=Y'; end;
if ( size(Y,1)~=size(dv,dims(1)) || ...
     (dims(2)>0 && size(Y,2)~=size(dv,dims(2))) ) 
   error('decision values and targets must be the same number of trials');
end

nClass=size(Y,ndims(Y)); 
if ( nClass==1 ) nClass=2; Y(:,2)=-Y(:,1); end; % Binary is a special case
szdv=size(dv);
szConf=szdv;szConf(dims(dims>0))=1; szConf(dims(1))=nClass*nClass;
sztY  =szdv;sztY(dims(dims(1:end-1)>0))=1;
confMx=zeros(szConf,'single');

pred  =dv>0; % binarize+logicalise the dv's

% make some index expressions to get the right parts of dv&conf
idx={};for d=1:ndims(dv); idx{d}=1:szdv(d); end;
for di=1:numel(dims); if(dims(di)>0) idx{dims(di)}=1;end; end; % set to 1 all used dims
idxY={}; for d=1:ndims(Y); idxY{d}=1:size(Y,d); end;
for trueLab=1:nClass;   % Loop over true labels
  % extract the true labels for tihs class
  idxY{end}=trueLab;
  tY = Y(idxY{:})>0;
  % duplicate along the trial dimensions as needed
  tY = repmat(shiftdim(tY,-dims(1)+1),sztY); 
  % Only if is positive prediction and label match does it count in confMx
  % sum over trials the number of correct matches
  coltl= sum(pred & tY,dims(1));
  for di=2:numel(dims)-1; coltl= sum(coltl,dims(di)); end
  % insert result into confusion matrix at the right point
  idx{dims(1)}  = (trueLab-1)*nClass+(1:nClass);
  confMx(idx{:})= coltl;
end
%-----------------------------------------------------------------------------
function testCase()
Y =ceil(rand(100,1)*2.9); % 1-3 labels
Yi=lab2ind(Y);
dv=[ ones(1,33) zeros(1,33) zeros(1,34);...
    zeros(1,33)  ones(1,33) zeros(1,34);...
    zeros(1,33) zeros(1,33) ones(1,34)]'; % known dv's 
conf=pred2conf(Yi,dv); 
conf=reshape(conf,size(Yi,2),[]);
conf(1,1),sum(Yi(:,1)>0 & dv(:,1)>0),conf(2,2),sum(Yi(:,2)>0 & dv(:,2)>0), conf(3,3),sum(Yi(:,3)>0 & dv(:,3)>0)

% with multiple sets of classification results
pred=dv2pred(cat(3,dv,dv,dv),2);
conf=pred2conf(Yi,pred)


% argh! check for silly input size dependent bug
Y=[      1    -1    -1    -1;
    -1    -1     1    -1;
    -1    -1    -1     1;
    -1     1    -1    -1]
dv=[     1    -1    -1    -1;
     1    -1    -1    -1;
     1    -1    -1    -1;
    -1     1    -1    -1];
reshape(pred2conf(Y,dv),[4 4])

Y2=[     1    -1    -1    -1;
     1    -1    -1    -1;
    -1    -1     1    -1;
    -1    -1     1    -1;
    -1    -1    -1     1;
    -1    -1    -1     1;
    -1     1    -1    -1;
    -1     1    -1    -1]
dv2=[     1    -1    -1    -1;
     1    -1    -1    -1;
     1    -1    -1    -1;
     1    -1    -1    -1;
     1    -1    -1    -1;
     1    -1    -1    -1;
    -1     1    -1    -1;
    -1     1    -1    -1];
reshape(pred2conf(Y2,dv2),[4 4])


