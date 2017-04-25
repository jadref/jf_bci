function [X,Xp]=deflate(X,p,dim,centerp,tol)
% deflate X in directions p along dimension dim
%
% [Xd,Xp]=deflate(X,p[,dim,centerp,tol])
%
%  After deflation this means that X will have no size along the directions
%  p
%
% Inputs:
%  X -- n-d data set
%  p -- [size(X,dim) x nDir] set of directions to deflate along
%  dim-- dimension(s) of X along which to deflate (1)
%        N.B. can be more than 1 dim at a time!
%  centerp -- [int] center the artifact signal before deflation (1)
%            0 : do nothing
%            1 : remove offsets (center)
%            2 : remove linear trends
%  tol -- [float] tolerance for detection of degenerate inputs (1e-4)
% Outputs:
%  Xd -- deflated X
%  Xp -- projected X, i.e. projection of X onto the orthogonalisation of p. 
%  p  -- an orthogonalised version of the input directions p
opts=struct('center',1,'detrend',1,'bands',[]);
if ( nargin < 3 || isempty(dim) ) dim=1; end;
szX=size(X); szP=size(p);
if ( nargin < 4 || isempty(centerp) ) centerp=1; end;
if ( nargin < 5 || isempty(tol) )     tol=1e-4; end;
if ( ndims(p)==numel(dim) ) szP=[szP 1]; end; % cope with the 1 def-dir case
if ( ~isequal(szX(dim),szP(1:end-1)) ) 
   error('deflate direction and X must have same size');
end

if ( centerp==1 )       p = repop(p,'-',mmean(p,1:numel(dim))); 
elseif ( centerp==2 )   p = detrend(p,numel(dim)); 
end;
% cov of the artifact signal: [nArt x nArt]
covP = tprod(p,[-(1:numel(szP)-1) 1],[],[-(1:numel(szP)-1) 2]); 
if ( numel(covP)>1 ) % whiten artifact signal - i.e. orthogonalise
  [U,S]    = eig(covP); S=diag(S); oS=S;
  si = S>=max(abs(S))*tol; S=S(si); U=U(:,si); % remove degenerate parts
  p   = tprod(p,[1:numel(szP)-1 -numel(szP)],repop(U,'./',sqrt(abs(S))'),[-numel(szP) numel(szP)],'n'); % whiten
else % normalise artifact signal
  p   = p./sqrt(abs(covP)); 
end

% N.B. using the householder matrix (H= I - p*p'/(p'*p)) is much more computationally expensive 
% in terms of memory and time, O(size(X,dim)^2)
% project X onto p
xidx=1:ndims(X); xidx(dim)=-dim; 
Xp= tprod(X,xidx,p,[-dim ndims(X)+1],'n');   
   
% deflate X
xidx=1:ndims(X); xidx(dim)=0; % make the singlention dims disappear to be replaced by p
X = X - tprod(Xp,[xidx -(ndims(X)+1)],p,[dim -(ndims(X)+1)],'n'); % deflate

return;
%----------------------------------------------------------------------------
function testCases()
% 1-d
X=randn(2,100);  p=randn(2,1);
clf;scatPlot(X,'.'); hold on; scatPlot(deflate(X,p,1,0),'r.'); plot([0 p(1,1)],[0 p(2,1)],'k'); axis equal

% degenerate deflation
clf;scatPlot(X,'.'); hold on; scatPlot(deflate(X,[p p],1,0),'r.'); plot([0 p(1,1)],[0 p(2,1)],'k'); axis equal

% 2-d deflation
clf;scatPlot(shiftdim(X),'.'); hold on; scatPlot(shiftdim(deflate(shiftdim(X,-1),shiftdim(p,-1),[1 2],0)),'r.'); plot([0 p(1,1)],[0 p(2,1)],'k'); axis equal

% 3d deflation
X = randn(3,100); p=randn(3,2);
clf;scatPlot(X,'.'); hold on; scatPlot(deflate(X,p,1,0),'r.'); plot([0 p(1,1)],[0 p(2,1)],'k'); axis equal


% time-series deflation
S = mixSig({{'sin' 5}; {'sin' 10}; {'coloredNoise' 1}},ones(1,3,100),100);
X = tprod(S,[-1 2 3],[1 0 0;0 1 0;2 .5 1],[1 -1]); % mix sig into noise
clf;mcplot([S(:,:,1);X(3,:,1);deflate(X(3,:,1),X(1:2,:,1)',2)]','labels',{'EOG1','EOG2','EEG (src)','EEG (mixed)','EEG (deflated)'})

% deflation with big offsets
S = mixSig({{'sin' 5}; {'sin' 10}; {'coloredNoise' 1}},ones(1,3,100),100);
X = tprod(S,[-1 2 3],[1 0 0;0 1 0;2 .5 1],[1 -1]); % mix sig into noise
X = repop(X,'+',[1000 3000 7000]'); % add in massive offsets
clf;mcplot([S(:,:,1);X(3,:,1);deflate(X(3,:,1),X(1:2,:,1)',2)]','labels',{'EOG1','EOG2','EEG (src)','EEG (mixed)','EEG (deflated)'})

% Compute the deflation matrix -- householder matrix
% H = -(p*p')./(p(:)'*p(:));
%H = -tprod(p,[1:numel(szP)-1 -numel(szP)],[],[numel(szP)-1+(1:numel(szP)-1) -numel(szP)])./(p(:)'*p(:));
%H(1:prod(szX(dim))+1:end)=1+H(1:prod(szX(dim))+1:end); % H = I - p*p'/(p'*p)
% apply the deflation matrix
% xidx=1:ndims(X); xidx(dim)=-xidx(dim);
% X = tprod(X,xidx,H,[-dim dim]);
