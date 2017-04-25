function [D] = covlogppDist(X, Z, dim, order, ridge, VERB)
% calculates the log probability product between covariance matrices *or*
% equivalently compute the Stein kernel between positive definite matrices
%
%Inspired by the paper "Probability Product Kernel", by Tony Jebara, Risi
%Kondor and Andrew Howard, Journal of Machine Learning Research 5 (2004),
%819-844.
%And by the paper:
%  ""
%
% Inputs:
% X   -- n-d input covariance matrix of N points
% Z   -- n-d input covariance matrix of M points, N.B. z=x if z isempty
% dim -- dimension along which the points lie (1, i.e. row vectors)
%        -1 -> trials in the last dimension (i.e. col-vectors)
%        (negative dim values index back from the last dimension)
% order -- [float] order of the Stein kernel to compute                    (.5)
%                  if order<0 then non-normalized kernel.
% ridge -- [float] diag ridge to add                                    (-1e-8)
%                  if ridge<0 then as fraction of mean squared eigenvalue
% VERB
% Outputs
% D   -- [N x M] squared distance matrix
%
if ( nargin < 2 ) Z = []; end
if ( nargin < 3 || isempty(dim) ) dim=-1; end; % dimension which contains the examples
if ( nargin < 4 || isempty(order) ) order=.5; end;
if ( nargin < 5 || isempty(ridge) ) ridge=[]; end;
if ( nargin < 6 || isempty(VERB) ) VERB=1; end;
if ( dim < 0 ) dim=ndims(X)+dim+1; end;
if ( order==0 ) order=-.5; end; % fix 0=> non-norm pp kernel


if ( dim<3 ) error('Only supported for [d1xd2xN] inputs currently'); end
%if ( size(X,1) ~= size(X,2) || size(Z,1)~=size(Z,2) )
%  error('Only for symetric matrices');
%end

% Normalize the shape of X,Z to be [ d x d x feat x N ]
szX=size(X); szX(end+1:max(dim))=1;
if ( dim>=3 )
  nf=prod(szX(3:dim-1)); if ( isempty(nf) ) nf=1; end;
  X=reshape(X,[szX(1:2) nf prod(szX(dim:end))]);
end;
isgram=false; 
if ( isempty(Z) ) 
   Z=X; isgram=true; szZ=size(Z);
else % re-shape into 4-d so consistent for all inputs
  szZ=size(Z); szZ(end+1:max(dim))=1;
  if ( dim>=3 )
	 nf=prod(szZ(3:dim-1)); if ( isempty(nf) ) nf=1; end;
	 Z=reshape(Z,[szZ(1:2) nf prod(szZ(dim:end))]);
  end;
end;

covIn=true; if( szX(1)~=szX(2) ) covIn=false; end 

diagIdx = int32(1:size(X,1)+1:size(X,1)*size(X,1));
Zcache=-ones(size(Z,4),2);
D=zeros([size(X,4) size(Z,4) size(X,3)]); % N x N x nKernel
if ( isa(X,'single') ) 
   mindet=realmin('single');
	if (isempty(ridge)) ridge=-1e-5; end;
else 
   mindet=realmin('double');
	if (isempty(ridge) ) ridge=-1e-8; end;
end;
if ( ridge<0 ) % use all the data to compute the size of ridge to use, same way as used in covEigTrans
  % compute the average covariance over all the data
   if ( covIn )
      Call = sum(X(:,:,:),3)./prod(szX(3:end));
   else
     Call = (X(:,:)*X(:,:)')./prod(szX(3:end));
   end
  if ( ~isgram ) 
   if ( covIn )
     Call = Call + sum(Z(:,:,:),3)./prod(szZ(3:end)); 
   else
     Call = Call + (Z(:,:)*Z(:,:)')./prod(szZ(3:end));
   end;
  end;
  % get it's eigen decomposition --- N.B. this tends to have a much flatter spectrum than an average cov
  sall = eig(Call);
  [ans,si]=sort(abs(sall),'ascend'); sall=sall(si); % get decreasing order
  % remove the rank defficient/non-PD entries
  si = sall>eps & ~isnan(sall) & abs(imag(sall))<eps; sall=sall(si);
  if ( abs(ridge)<1 ) % fraction of the spectrum
	 ridge = sall(max(1,round(numel(sall)*abs(ridge))));
  else % number of components back in the eigen-spectrum
	 ridge = sall(abs(ridge));
  end
end
if ( VERB>0 && numel(D)>10 ) ncomp=0; fprintf([mfilename ':']); end
for ni=1:size(D,3); % features for seperate distance matrices
  for i=1:size(X,4);
	 if (covIn) 
       Xi=double(X(:,:,ni,i));
    else
       Xi=double(X(:,:,ni,i)*X(:,:,ni,i)');
    end
    diagX=Xi(diagIdx); 
	 if ( ridge>0 ) 
       Xi(diagIdx) = Xi(diagIdx) + max(eps,ridge); 
    end;
    logdetx=logdet(Xi,'chol');
    % cache for later
	 if ( isgram ) Zcache(i,:)=logdetx; end;

	 if ( isgram ) Nz = i-1; else Nz=size(Z,4); end;
	 for j=1:Nz;
      if (covIn) 
         Zj=double(Z(:,:,ni,j));
      else
         Zj=double(Z(:,:,ni,j)*Z(:,:,ni,j)');
      end
      if ( ridge>0 )
        Zj(diagIdx) = Zj(diagIdx) + max(eps,ridge); 
      end;

		if ( Zcache(j)>=0 ) 
         logdetz=Zcache(j,1); 
		else % compute the new entry
         logdetz=logdet(Zj,'chol');
         Zcache(j,:)=logdetz;
      end
																	 % sum of the inverses
	   % log(K(X,Z)) = log(sqrt(det(X)^order*det(Z)^order)/det(X+Z)^order)
	   %             = order*(log(det(X))/2+log(det(Z))/2 - log(det((X+Z)/2)))
		%             = order*(log(det(X))/2+log(det(Z))/2 - log(det(X+Z))+log(2.^size(X,1)))
      XZ = Xi + Zj;
      logdetxz=logdet(XZ,'chol');
      if ( isnan(logdetxz) || isinf(logdetxz) ) 
         warning('log-0!');
      end
		if ( order>0 ) 
		  Kxz = order*(logdetx/2 + logdetz/2 - logdetxz + size(X,1)*log(2));
		else
		  Kxz = abs(order)*(-logdetxz -log(2)*size(X,1));
		end
		D(i,j,ni) = Kxz;
		if( VERB>1 && ...
          (isinf(detx)|| isnan(detx)|| detx==0 || isinf(detz)|| isnan(detz)|| detz==0 || isinf(detxz)) ) 
         warning('zero determinant!');
      end

		if( VERB>0 && numel(D)>10) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		if ( isgram )
		  D(j,i,ni) = Kxz;
		  if( VERB>0 && numel(D)>10) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		end;	
	 end % z 
  end % X
end% feat
if ( VERB>0 ) fprintf('\n'); end
% BODGE: convert form inner-product to a distance by negating the value....
D=-D;
return;


%-------------------------------------------------------------------------
function testCase()
X=randn(10,300,10,10);
C=tprod(X,[1 -2 3 4],[],[2 -2 3 4]); % comp cov mx


Ceig =covlogppDistEig(C(:,:,1:2),[],3,[],0); % eigenvalue decomp version.
Ccol =covlogppDist(C(:,:,1:2),[],3,[],0); % cholsky decomp version.
Ccolx=covlogppDist(X(:,:,1:2),[],3,[],0); % direct input cholsky decomp


% check affine invariance
A     =randn(size(X,1),size(X,1));
AX    =tprod(X,[-1 2 3 4],A,[-1 1]);
ACcolx=covlogppDist(AX(:,:,1:2),[],3,[],0);
ACA   =tprod(tprod(C,[-1 2 3 4],A,[-1 1]),[1 -2 3 4],A,[-2 2]);
ACcol =covlogppDist(ACA(:,:,1:2),[],3,[],0);

% test with degenerate inputs
Ct=tprod(cat(1,X,zeros([10,size(X,2),size(X,3),size(X,4)])),[1 -2 3 4],[],[2 -2 3 4]); % with extra zeros....
covlogppDistEig(Ct(:,:,1:2),[],3)
covlogppDist(Ct(:,:,1:2),[],3)
covlogppDist(Ct(:,:,1:2),[],3,[],-.1)
covlogppDist(Ct(:,:,1:2),[],3,[],1e-6) % use ridge to cope...


tic,Dx =covlogppDist(X,[]);toc  %
tic,D3 =covlogppDist(C,[]);toc  %
tic,D32=covlogppDistEig(C,[]);toc  % 
tic,D1=sqDist(X,[],[1 2]);toc % ecluidian dist
tic,D2=covlogmDist(C,[]);toc  % riemann dist
tic,D3=covlogppDist(C,[]);toc  % pp-dist
tic,D4=covklDist(C,[]);toc  % kl-dist
tic,D5=covlogppDist(C,[],[],.5);toc % stein-kernel order=.5
clf;mimage(D1./norm(D1),D2./norm(D2));%subplot(211);imagesc(D1);subplot(212);imagesc(D2);


										  % some simple scaling tests
X=zeros(10,10,20);for i=1:20; C(:,:,i)=diag(abs(randn(size(X,1),1))); end;
m=[1:10 10:5:50];
for i=1:numel(m);
  mX   = C(:,:,2)*diag([m(i);ones(size(X,1)-1,1)]);
  d(i) = det(mX);
  dp(i)= det(C(:,:,1)+mX);
  e(i) = sum(sum(C(:,:,1).*mX));
  k(i) = exp(-.5*covlogppDist(C(:,:,1),mX,3));
  k2(i) = exp(-.5*covlogppDist(C(:,:,1),mX,3,-.5));
end;
clf;plot(m',[e/10;d*10;k;k2;sqrt(m)]','linewidth',2);
					  %A: log-distance scales roughly as sqrt of the multiplier..
% actually, because is normalizes -- it actually gets smaller with inc multiplier...
