function [z]=jf_robustcov(z,varargin);
% compute a single robust weighted covariance estimate along indicated dimension
%
opts=struct('subIdx',[],'verb',0,'dim',{{'ch' 'ch_2' 'window'}},'distType','covlogppDist','maxIter',10,'tol',.99);
opts=parseOpts(opts,varargin);

dim=n2d(z,opts.dim,0,0);
if ( dim(2)==0 ) dim(2)=n2d(z,[z.di(dim(1)).name '_2'],1,0); end; % BODGE!
if ( dim(2)==0 ) % need to do the cov still
   z=jf_cov(z,'dim',[dim(1) n2d(z,'time')]);
end
dim=n2d(z,opts.dim,0,0);
if ( dim(2)==0 ) dim(2)=n2d(z,[z.di(dim(1)).name '_2'],1,0); end; % BODGE!
covD=dim(1:2); wghtD=dim(3:end);

X=z.X;
szX=size(X);
z.X = zeros(szX([1:wghtD-1 wghtD+1:end]),class(X)); % squash the weighting dim

[idx,chkStrides,nchnks]=nextChunk([],size(X),dim,1);
ci=0; if ( opts.verb >= 0 && nchnks>1 ) fprintf('%s:',mfilename); end;
while ( ~isempty(idx) )
  C  = X(idx{:}); % grab this set of covariances
  K  = exp(-covlogppDist(C,[],wghtD,[],[],opts.verb-1)); % kernel-similarity
  alpha=mean(K,1)'; % kernel-similarity between examples and target
  beta =alpha; beta=beta./sum(beta); % input space weighting over examples to get this target similarity
  for iter=1:opts.maxIter;
	 Cmu = tprod(C,[1:wghtD-1 -wghtD wghtD+1:ndims(C)],beta,-wghtD); % current centered input example
	 sim = exp(-covlogppDist(C,Cmu,wghtD,[],[],opts.verb-1)); % Kernel-similarity to examples
	 % tol between current and desired similarities, converge if sufficiently small correlation
	 if ( sim'*alpha ./ sqrt(sim'*sim) / sqrt(alpha'*alpha) > 1-opts.tol ) 
		break;
	 end;
	 beta= beta.*alpha./sim; % updated input-space weighting
	 beta= beta./sum(beta);
  end
  if ( opts.verb >=0 && nchnks>1 ) ci=ci+1; textprogressbar(ci,nchnks);  end
  z.X(idx{1:wghtD-1},idx{wghtD+1:end})=Cmu;
  idx=nextChunk(idx,size(X),chkStrides);
end
if ( opts.verb>=0 && nchnks>1) fprintf('\n'); end;

% update meta-info
odi=z.di;
z.di=z.di([1:wghtD-1 wghtD+1:end]);

z =jf_addprep(z,mfilename,sprintf('over %s',odi(wghtD).name),opts,[]);
return;

%-------------------------------------------------------------
function testCase()
  oz=z;
  z=jf_windowData(z,'dim','time','width_ms',1000,'overlap',0);
  z=jf_cov(z);
  z=jf_rcov(z,'dim',{'ch' 'ch_2' 'window'});
  
