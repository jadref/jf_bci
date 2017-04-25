function D = covlogmDist(X, Z, dim, norm, VERB)
% D = covlogmDist(X, Z, dim, verb) % riemann distance for symetric pos-def matrices
% 
% Inputs:
% X   -- n-d input covariance matrix of N points
% Z   -- n-d input covariance matrix of M points, N.B. z=x if z isempty
% dim -- dimension along which the points lie (1, i.e. row vectors)
%        -1 -> trials in the last dimension (i.e. col-vectors)
%        (negative dim values index back from the last dimension)
% norm -- [float] norm to use for the vector of eigenvalues, i.e.  (2)
%           D = (\sum( log(C1^-.5C2C^-.5).^p))^(1/p) 
% Outputs
% D   -- [N x M] squared distance matrix
if ( nargin < 2 ) Z = []; end
if ( nargin < 3 || isempty(dim) ) dim=-1; end; % dimension which contains the examples
if ( nargin < 4 || isempty(norm) ) norm=2; end;
if ( nargin < 5 || isempty(VERB) ) VERB=1; end;
if ( dim < 0 ) dim=ndims(X)+dim+1; end;
if ( isa(X,'single') ) eps=1e-6; else eps=1e-10; end;

if ( dim<3 ) error('Only supported for [d1xd2xN] inputs currently'); end
%if ( size(X,1) ~= size(X,2) || size(Z,1)~=size(Z,2) )
%  error('Only for symetric matrices');
%end

% Normalize the shape of X,Z to be [ d x t x feat x N ]
szX=size(X); 
if ( dim>=3 )
  nf=prod(szX(3:dim-1)); if ( isempty(nf) ) nf=1; end;
  X=reshape(X,[szX(1:2) nf prod(szX(dim:end))]);
end;
isgram=false; if ( isempty(Z) ) Z=X; isgram=true; szZ=size(Z);
else % re-shape into 4-d so consistent for all inputs
  szZ=size(Z);
  if ( dim>=3 )
	 nf=prod(szZ(3:dim-1)); if ( isempty(nf) ) nf=1; end;
	 Z=reshape(Z,[szZ(1:2) nf prod(szZ(dim:end))]);
  end;
end;

covIn=true; if( szX(1)~=szX(2) ) covIn=false; end 

if ( VERB>0 ) ncomp=0; fprintf([mfilename ':']); end
D=zeros([size(X,4) size(Z,4) size(X,3)]); % N x N x nKernel
for ni=1:size(D,3); % features for seperate distance matrices
  for i=1:size(X,4);
	 if (covIn) 
       Xi=double(X(:,:,ni,i));
    else
       Xi=double(X(:,:,ni,i)*X(:,:,ni,i)');
    end
	 % N.B. eig is very numerically sensitive, so ensure use double....
	 [Ux,sx]=eig(Xi); sx=diag(sx);
	  % select non-zero entries - cope with rank deficiency, numerical issues
	 si = sx>eps & ~isnan(sx) & abs(imag(sx))<eps;
										  % compute the whitener
	 isqrtX = Ux(:,si)*diag(1./sqrt(real(sx(si))));

	 if ( isgram ) Nz = i; else Nz=prod(szZ(dim:end)); end;
	 for j=1:Nz;
      if (covIn) 
         Zj=double(Z(:,:,ni,j));
      else
         Zj=double(Z(:,:,ni,j)*Z(:,:,ni,j)');
      end
      % compute the whitened data
		isqrtXZisqrtX = isqrtX'*Zj*isqrtX;
		% compute the eigen spectrum of the whitened data
		sij=eig(isqrtXZisqrtX);
		ssij = sij>eps & ~isnan(sij) & abs(imag(sij))<eps;
      % compute the distance = RMS log eigen-spectrum of the whitened covariances
		if ( norm==2 ) 
		  D(i,j,ni) = sqrt(sum(log(real(sij(ssij))).^2));
		elseif ( norm==1 ) 
		  D(i,j,ni) = sum(abs(log(real(sij(ssij)))));
		else
		  D(i,j,ni) = power(sum(power(log(real(sij(ssij))),norm)),1/norm);
		end
		if( VERB>0 ) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		if ( isgram )
		  D(j,i,ni) = D(i,j,ni);
		  if( VERB>0 ) ncomp=ncomp+1; textprogressbar(ncomp,numel(D)); end;
		end;	 
	 end % for Z
  end % for X
end % for feat
if ( VERB>0 ) fprintf('\n'); end
return;

%-------------------------------------------------------------------------
function testCase()
X=randn(10,300,10,100);
X=tprod(X,[1 -2 3 4],[],[2 -2 3 4]); % comp cov mx
tic,D1=sqDist(X,[],[3 4]);toc % ecluidian dist
tic,D2=covlogmDist(X,[]);toc  % riemann dist
tic,D2=covlogmDist(X,[],[],1);toc  % riemann dist, l1 difference

clf;mimage(D1./norm(D1),D2./norm(D2));%subplot(211);imagesc(D1);subplot(212);imagesc(D2);
