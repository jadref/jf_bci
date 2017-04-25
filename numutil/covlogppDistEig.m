function [D] = covlogppDistEig(X, Z, dim, order, eps, nrmp, VERB) 
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
% order -- [float] order of the Stein kernel to compute                  ([])
% tol -- [float] minimum eigenvalue we allow (0)
% nrmp -- [bool] flag if we use normalized probabilities in the product  (1)
% VERB
% Outputs
% D   -- [N x M] squared distance matrix
%
% TODO: [X] convert to using det(C) which is 10-15x faster than eig -- but less robust?
if ( nargin < 2 ) Z = []; end
if ( nargin < 3 || isempty(dim) ) dim=-1; end; % dimension which contains the examples
if ( nargin < 4 || isempty(order) ) order=.5; end;
if ( nargin < 5 || isempty(eps) ) eps=[]; end;
if ( nargin < 6 || isempty(nrmp) ) nrmp=true; end;
if ( nargin < 7 || isempty(VERB) ) VERB=1; end;
if ( dim < 0 ) dim=ndims(X)+dim+1; end;
if ( isempty(eps) ) if ( isa(X,'single') ) eps=1e-6; else eps=1e-10; end; end;

if ( dim<3 ) error('Only supported for [d1xd2xN] inputs currently'); end
if ( size(X,1) ~= size(X,2) || size(Z,1)~=size(Z,2) )
  error('Only for symetric matrices');
end

% Normalize the shape of X,Z to be [ d x t x feat x N ]
szX=size(X); szX(end+1:max(dim))=1;
if ( dim>=3 )
  nf=prod(szX(3:dim-1)); if ( isempty(nf) ) nf=1; end;
  X=reshape(X,[szX(1:2) nf prod(szX(dim:end))]);
end;
isgram=false; if ( isempty(Z) ) Z=X; isgram=true; szZ=size(Z);
else % re-shape into 4-d so consistent for all inputs
  szZ=size(Z); szZ(end+1:max(dim))=1;
  if ( dim>=3 )
	 nf=prod(szZ(3:dim-1)); if ( isempty(nf) ) nf=1; end;
	 Z=reshape(Z,[szZ(1:2) nf prod(szZ(dim:end))]);
  end;
end;

if ( VERB>0 ) ncomp=0; fprintf([mfilename ':']); end
Zcache={};
D=zeros([size(X,4) size(Z,4) size(X,3)]); % N x N x nKernel
for ni=1:size(D,3); % features for seperate distance matrices
  for i=1:size(X,4);
	 Xi=double(X(:,:,ni,i));
	 [Ux,sx]=eig(Xi); sx=diag(sx);
	 % N.B. just ignoring these dimensions introduces numerical issues....
	  % correct non-zero entries, by adding a ridge - cope with rank deficiency, numerical issues
	 si = sx>eps & ~isnan(sx) & abs(imag(sx))<eps;
	 if ( ~all(si) ) 
		r  = max([abs(sx(~si));eps]);
		sx = max(real(sx),r);%real(sx+2*r); %Ux=Ux(:,si);
	 end
										  % cache for later
	 if ( isgram ) Zcache{i}={Ux sx}; end;

	 if ( isgram ) Nz = i; else Nz=size(Z,4); end;
	 for j=1:Nz;
		Zj=double(Z(:,:,ni,j));
		if ( j<=numel(Zcache) ) [Uz sz]=Zcache{j}{:};
		else % compute the new entry
		  [Uz,sz]=eig(Zj); sz=diag(sz);
	  % select non-zero entries - cope with rank deficiency, numerical issues
		  si = sz>eps & ~isnan(sz) & abs(imag(sz))<eps;
		  if ( ~all(si) )
          r = max([abs(sz(~si));eps]);
			 sz= max(real(sz),r);%real(sz+2*r); % Uz=Uz(:,si);
		  end
		  % record the decomposition
        Zcache{j}={Uz sz};
		end

																	 % sum of the inverses
		if ( isempty(order) ) % pp-kernel
		  ixiz = (Ux*diag(1./sx)*Ux' + Uz*diag(1./sz)*Uz'); % (1/X+1/Z)
		  [Uxz sixiz]=eig(ixiz); sixiz=diag(sixiz);
        % select non-zero entries - cope with rank deficiency, numerical issues
		  si = sixiz>eps & ~isnan(sixiz) & abs(imag(sixiz))<eps;	 
		  sixiz=real(sixiz(si));

% N.B. prob product = det(inv((inv(X)+inv(Z)))/2)^1/2 / sqrt(det(X)^1/2*det(Z)^1/2)
%                   =     det((inv(X)+inv(Z))/2)^-1/2 / sqrt(det(X)^1/2*det(Z)^1/2)
%      log(pp)      = -1/2*sum(log(eig(inv(X)+inv(Z))/2)) - 1/4*sum(log(X)) -1/4*sum(log(Z))
		  D(i,j,ni) = -sum(log(sixiz./2))/2;
		  if ( nrmp ) % include the normalization factors
			 D(i,j,ni) = D(i,j,ni) - sum(log(sz))/2/2 - sum(log(sx))/2/2;
		  end

		else % stein kernel of given order
			% log(K(X,Z)) = log(sqrt(det(X)^order*det(Z)^order)/det(X+Z)^order)
		   %             = order*(log(det(X))/2+log(det(Z))/2 - log(det(X+Z)))
		  xz = Xi+Zj;%(Ux*diag(sx)*Ux' + Uz*diag(sz)*Uz')/2; % (Xi + Zj)/2; % (X+Z)
		  [Uxz sxz]=eig(xz); sxz=diag(sxz)/2;
	     % select non-zero entries - cope with rank deficiency, numerical issues
		  si = sxz>eps & ~isnan(sxz) & abs(imag(sxz))<eps;	 
		  if ( ~all(si) )
          r  = max([abs(sxz(~si));eps]);
			 sxz= max(real(sxz),r);%real(sz+2*r); % Uz=Uz(:,si);
		  end
		  D(i,j,ni) = order*(sum(log(sz))/2 + sum(log(sx))/2 - sum(log(sxz)));
		end

		if( VERB>0 ) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		if ( isgram )
		  D(j,i,ni) = D(i,j,ni);
		  if( VERB>0 ) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		end;	
	 end % z 
  end % X
end% feat
if ( VERB>0 ) fprintf('\n'); end
% BODGE: convert form inner-product to a distance by inverting the value....
D=-D;
return;


%-------------------------------------------------------------------------
function testCase()
X=randn(10,300,10,10);
C=tprod(X,[1 -2 3 4],[],[2 -2 3 4]); % comp cov mx
tic,D3=covlogppDist(C(:,:,1:5),[]);toc  % 
tic,D1=sqDist(C,[],[1 2]);toc % ecluidian dist
tic,D2=covlogmDist(C,[]);toc  % riemann dist
tic,D3=covlogppDist(C,[]);toc  % pp-dist
tic,D4=covklDist(C,[]);toc  % kl-dist
tic,D5=covlogppDist(C,[],[],[],.5);toc % stein-kernel order=.5
clf;mimage(D1./norm(D1),D2./norm(D2));%subplot(211);imagesc(D1);subplot(212);imagesc(D2);
