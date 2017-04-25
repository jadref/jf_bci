function [Xmu,dim,featD,spD] = tauERP(X,Y,dim,taus,meanp)
% compute shifted average
%
%   [Xmu,dim,featD,spD] = tauERP(X,Y,dim,taus,meanp) 
%
% Inputs:
%  X - [n-d] data
%  Y - [size(X,dim) x nSubProb] indicator for when different events happened -> design matrix
%      OR
%      [nSubProb x size(X,dim)]
%  dim - [>2 x 1] which dimensions of x/y mean what.                     ([-2 -1])
%        dim(1) = time-dimension, this is the dim along which we will computed shifted averages
%        dim(2:end) = epoch-dimensions, these dimesions will be averaged away
%  taus- [ntau x 1] the set of time-shifts to use                        (0)
%  meanp-[bool] flag if we should compute per-class average or raw-sum   (1)
% Outputs:
%  Xmu - [n-d] the computed average response
if ( nargin<3 || isempty(dim) )  dim=[-2 -1]; end;
if ( nargin<4 || isempty(taus))  taus=0; end 
if ( nargin<5 || isempty(meanp)) meanp=1; end;
dim(dim<0)=ndims(X)+dim(dim<0)+1;

szX=size(X); szX(end+1:max(dim))=1;     featD=setdiff(1:numel(szX),dim(2:end)); tauD=find(featD==dim(1));
szY=size(Y); szY(end+1:numel(dim)+1)=1; spD  =numel(dim)+1:numel(szY);  yPerm=false;
if ( szY(1:numel(dim))~=szX(dim) ) % prefer [N x nSp]
   if( szY(end-numel(dim)+1:end)==szX(dim) ) % permute Y to the right shape
      yPerm = true;
      Y=permute(Y,[numel(szY)-numel(dim)+1:numel(szY) 1:numel(szY)-numel(dim)]); % permute to [ N x nSp ]
      szY=size(Y); szY(end+1:numel(dim)+1)=1; spD  =numel(dim)+1:numel(szY);
   else
      error('X and Y must match'); 
   end
end;
if ( szY(spD)==1 && all(Y(:)==-1 | Y(:)==0 | Y(:)==1) && any(Y(:)==-1) ) % convert to 2-class
   Y=cat(spD,Y,-Y); Y(Y<0)=0; szY(spD)=2;
end
Y(isnan(Y(:)))=0; % ensure Nan's are removed..

% idx for acc/step dim in tensor product
tpidxX=1:numel(szX); tpidxX(featD)=1:numel(featD); tpidxX(dim)=-dim;

szMu=[szX(featD) szY(spD) 1]; szMu(tauD)=numel(taus);
Xmu =zeros(szMu);
xmuidx=repmat({':'},ndims(Xmu),1); xmuidx{tauD}=1;% idx to insert result

if( numel(taus)>5 ) fprintf('tauERP:'); end;
wght=zeros(size(Y));
for ti=1:numel(taus);
   tau = taus(ti);
   % compute the shifted weighting vector
   wght(:)=0;
   if ( tau>0 ) wght(tau+1:end,:) = Y(1:end-tau,:);
   else         wght(1:end+tau,:) = Y(-tau+1:end,:);
   end
   if( meanp ) % normalize the effect of the targets and variable numbers of targets
      mu=wght; for d=1:numel(dim); mu=sum(mu,d); end; mu(mu==0)=1; % compute the number of samples summed
      wght=repop(wght,'/',mu);  % normalize
   end 
   xmuidx{tauD}=ti;
   Xmu(xmuidx{:})=tprod(X,tpidxX,wght,[-dim(:)' numel(featD)+(1:numel(spD))]);
   if( numel(taus)>5 ) textprogressbar(ti,numel(taus)); end;
end
if ( nargout>2 && yPerm ) % ensure spD is correct by undoing the permute of Y
   spD=1:numel(szY)-numel(dim);
end
return;   

function testCase();
X=randn(100,10);
Y=randn(size(X)); Y(abs(Y)<.9)=0; Y=sign(Y);
Xmu=tauERP(X,Y,[],0:10); % with average
Xmu=tauERP(X,Y,[],0:10,0); % with average
clf;image3d(Xmu,1,'disptype','plot')