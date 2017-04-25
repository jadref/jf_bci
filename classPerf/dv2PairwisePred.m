function [pred]=dv2pred(dv,spD,spMx,discretepreddv,trD)
% convert a set of decison values to a set of class predictions
%
% [pred]=dv2pred(dv,dim,spMx,discretepreddv)
%
% Inputs:
%  dv   - [n-d] input set of ** +/-1 ** decision values
%  spD  - [int] dimensions of dv which contains the binary-classifiers per-subProblem outputs
%  spMx - [size(dv,spD) x nClass] sub-problem decoding matrix, fixed for all trials
%         OR
%         [size(dv,spD) x size(dv,trD) x nClass] per-trial sub-problem decoding matrix
%  discretepreddv -- [2 x 1 bool] generate discrete or continuous predictions,decisionValues? (1)
%  trD - [int] dimensions of dv which contains the trials, for per-trial decoding, ([])
% Outputs:
%  pred - size(dv) with size(pred,dim)=nClass set of per-label decision values
if( nargin < 2 || isempty(spD) ) % default spD
   spD=find(size(dv)>1,1); if(isempty(spD))spD=1;end; 
end;
if( spD < 0 ) spD=ndims(dv)+spD+1;end;

if( nargin < 3 || isempty(spMx) ) spMx='1vR'; end;
if( nargin < 4 || isempty(discretepreddv) ) discretepreddv=[1 0]; else discretepreddv(end+1:2)=0; end;
if( nargin < 5 ) trD=[]; end;

if(isstr(spMx)) % convert encoding type name to decoding matrix
   if ( numel(spD)>2 ) error('Only spec encoding type of simply 1-d inputs'); end;
   switch(spMx)
    case '1vR'; spMx=-ones(size(dv,spD(1)),size(dv,spD(1))); spMx(1:size(spMx,1)+1:end)=1;
    case '1v1'; % N.B. we assume 1v1 in order, 1v2,1v3,...1vN,2v3,2v4,...2vN,...
     nClass=(-1+sqrt(1+8*size(dv,spD(1))))/2+1;
     spMx=[]; nSp=0;
     for ci=1:nClass-1;for cj=ci+1:nClass; nSp=nSp+1; spMx(nSp,ci)=1;spMx(nSp,cj)=-1; end;end;
    otherwise; error('not supported yet');
  end
end

% convert fixed decoding matrix to degenerate per-trial decoding matrix
if ( ~isempty(trD) && ndims(spMx)<=numel(spD)+numel(trD) ) % add extra dim for the trials dimension
   szspMx=size(spMx); spMx=reshape(spMx,[szspMx(1:end-1) ones(numel(trD),1) szspMx(end)]); % [nSp x nTr x nCls]
end

% used the decoding matrix to do the actual decoding
if ( discretepreddv(2) ) dv=sign(dv); end; % convert sp predictions to discrete values

% build the pairwise coupling matrix
nCls=size(spMx,ndims(spMx));
szdv= size(dv); nspD=setdiff(1:numel(szdv),spD);
% to hold the result
szpred = szdv; szpred(spD(1))=nCls; szpred(spD(2:end))=1; if(isa(dv,'single')) pred=zeros(szpred,'single'); else bpred=zeros(szpred); end
% pre-comp the pos/neg indicator functions
spMxp = reshape(single(spMx>0),prod(szdv(spD)),[]); spMxn=reshape(single(spMx<0),prod(szdv(spD)),[]); % pos, neg entries
idx={}; for d=1:ndims(dv); idx{d}=1:size(dv,d); end; % idx epr for getting each prediction in turn
for pi=1:prod(szdv(nspD)); % loop over all the non-subProb dims
   [idx{nspD}]=ind2sub(szdv(nspD),pi); % get this points index
   idxpred=idx; idxpred{spD(1)}=1:size(pred,spD(1)); if( numel(spD)>1 ) [idxpred{spD(2:end)}]=deal(1); end; % this points solution
   dvi   = dv(idx{:});   % extract this one's set of predictions
   dvi   = 1./(1+exp(-dvi)); % convert to probs, faster to do it here and leave dv unchanged
   Ri    = repop(dvi(:),'*',spMxn)-repop(1-dvi(:),'*',spMxp); % [ nCls x nSp ]
   Ri    = Ri';  % [ nSp x nCls ]
   if ( 0 ) % solve the constrained lagrangian equation version 
      predi = [(Ri*Ri') ones(size(Ri,1),1); ones(size(Ri,1),1)' 0]\[zeros(size(Ri,1),1);1]; % solve the system and generate output
      pred(idxpred{:}) = predi(1:end-1);%./sum(predi); 
   else % project out the constraint and solve (20% faster!)
      Rik = Ri(end,:);
      predi = -Rik/repop(Ri(1:end-1,:),'-',Rik);
      pred(idxpred{:}) = [predi'; 1-sum(predi)];
   end
end
% convert to prediction
if ( discretepreddv(1) ) 
   [ans,mi]=mmin(-pred,spD(1)); pred(:)=-1; pred(mi)=1;
end
return
%---------------------------------------------------------------------------
function testCase()
dv=randn(100,6); % [nEp x nSp]
pred=dv2PairwisePred(dv,2,'1vR');
tic,pred=dv2PairwisePred(sign(dv),2,'1v1');toc

% test a multi-dimensional decoding, i.e. multi-class sequence
dv=randn(14,30,3); % [ nEp x nSeq x nSp]
spMx=[1 -1 -1;-1 1 -1;-1 -1 1]'; % 1vR 3-class spMx [nSp x nCls]
spMx=repmat(shiftdim(spMx,-1),[size(dv,1) 1 1]); % [nEp x nSp x nCls]
pred=dv2PairwisePred(dv,[1 3],spMx);

