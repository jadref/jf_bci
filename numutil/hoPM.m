function [S,varargout]=hoPM(A,varargin)
% Higher Order Power Method to best rank-1 fit to n-d array
%
% [S,U1,U2,...UN]=hoPM(A[maxIter,tol,S,U1,U2,..UN])
% Find the best rank-1 approx for the n-d array A using the higher order
% power method (or rank-1 alternating least squares)
% 
% Inputs:
%  A  -- n-d array
%  maxIter -- maximum number of iterations to perform
%  tol-- tolerance on change in U, to stop iteration
%  S  -- [1x1] seed higher-order eigen-value
%  U1 -- [ size(A,1) x 1 ] starting value for A's 1st dimension
%  U2 -- [ size(A,2) x 1 ] starting value for A's 2nd dimension
%  ...
%  UN -- [ size(A,N) x 1 ] starting value for A's Nth dimension
% Options:
%  trnInd -- [bool/int size(A)] indication of which values to use for training
%            N.B. convergence is *not* guaranteed if A has small size (<~20 elm/dim)
% Outputs:
%  S  -- [1x1] higher order eigenvalue
%  U1 -- [ size(A,1) x 1 ] starting value for A's 1st dimension
%  U2 -- [ size(A,2) x 1 ] starting value for A's 2nd dimension
%  ...
%  UN -- [ size(A,N) x 1 ] starting value for A's Nth dimension
opts=struct('maxIter',1000,'tol',0,'tol0',0,'objTol',0,'objTol0',1e-4,'marate',.2,'wght',[],'trnInd',[],'rewghtStep',5,...
            'verb',0,'priorSoln',[],'alg',[],'initType','mean','seed',[],'seedNoise',0,...
            'orthoPen',1,'orthoLambda',[]);
[opts,varargin]=parseOpts(opts,varargin);

% Extract the provided seed values
if ( nargin > 3 & ~isempty(varargin) )
  if ( isnumeric(varargin{1}) )
   S = varargin{1};
   U(1:numel(varargin)-1)=varargin(2:end);
   if (iscell(U) && numel(U)==1 && iscell(U{1})) U=U{1}; end;
  else
    S=[]; U={};
    warning('Unrecog options ignored');
  end
else
   S=[]; U={};
end
if ( ~isempty(opts.seed) ) 
   S = opts.seed{1}; U=opts.seed(2:end);
end

sizeA=size(A); nd=ndims(A);
priorSoln=opts.priorSoln; if ( numel(priorSoln)==1 && isempty(priorSoln{1}) ) priorSoln=[]; end;
% get non-negavity constraint
alg=opts.alg; 
if ( ~iscell(alg) ) alg={alg}; end; 
for i=1:numel(alg); if(isempty(alg{i}))alg{i}='ls'; end; end;

oA=A; % save the orginal inputs
if ( ~isempty(priorSoln) ) % pre-compute the norm of the prior
  Ap = parafac(priorSoln);
  A  = A-Ap;
end
  
% do the value imputation stuff for the ignored points
tstInd=[]; 
if ( ~isempty(opts.wght) )
  tstInd=opts.wght;
  if ( islogical(tstInd) )     tstInd=(tstInd==0); % points with 0-weight are test points
  elseif ( isnumeric(tstInd) && all(tstInd(:)==1 | tstInd(:)==0 | tstInd(:)==-1) )
    tstInd=tstInd>0;     % tst points have +1 label
  elseif ( isnumeric(tstInd) && all(tstInd(:)>=1) && all(tstInd(:))<=numel(A) ) 
    tstInd=false(size(A)); tstInd(int32(opts.wght))=true; % weight is list of test points
  else error('wght is of unsupported type'); 
  end
  % replace tst elments with 0
  A(tstInd)=0;
end

oA2= oA(:)'*oA(:);
A2 = A(:)'*A(:); % pre-compute for fast sse computation
% orthogonalisation penalties stuff
orthoPen=opts.orthoPen; 
if ( isempty(orthoPen) || isempty(priorSoln) ) 
  orthoPen=0; 
else 
  orthoPen=A2./numel(A)*orthoPen; 
end;
orthoLambda=[]; 
if(~isempty(opts.orthoLambda) &&  ~isempty(priorSoln)) 
  orthoLambda=priorSoln{1};%zeros(size(priorSoln{1}));%
  lambdaStep=1;
end;

% Fill in the (rest of) the seed values
if ( isempty(S) ) S=1; end;
if ( numel(U)<nd ) 
  switch lower(opts.initType)
   case 'mean'; 
    mu=A;
    for d=1:nd; 
      U{d}=sum(reshape(mu,size(A,d),[]),2); % mean along this dim
      U{d}=U{d}./norm(U{d}); 
      mu  =sum(mu,d);
    end;
    
   case {'1stel','1stelm'};
   [idx{1:nd}]=deal(1); % build index expression to extract the bit we want from A
   for d=numel(U)+1:nd;      
      idx{d}=1:sizeA(d);
      U{d}=shiftdim(A(idx{:}));
      U{d}=U{d}./norm(U{d}); % seed guess
      idx{d}=1;
    end
    otherwise; error('Unrec initType : %s',opts.initType);
  end
end

if ( ~isempty(opts.seedNoise) && opts.seedNoise>0 ) % level of noise to add to seed solution
  for d=1:numel(U);
    nrm=sqrt(sum(U{d}.^2,1));
    % generate random noise of ave length=1, then scale by ave U len * seedNoise size
    U{d} = U{d}+randn(size(U{d}))./sqrt(size(U{d},1))*mean(nrm)*opts.seedNoise;
    U{d} = repop(U{d},'./',sqrt(sum(U{d}.^2))); % re-normalise the length
  end
end


% compute performance of the seed solution
AU=A;  % A^{1:N} \prod_{j neq d} U_j
if ( ~isempty(priorSoln) ) PU=ones(numel(priorSoln{1}),1); end; % proj on prior
if ( ~isempty(tstInd) )    WU2=single(~tstInd); end;
for d=1:nd; % Cached [1:d-1] in AU so only mult the rest
  AU=tprod(AU,[1:d-1 -d d+1:nd],U{d},[-d]);
  % accumulate projection onto the prior components
  if ( ~isempty(priorSoln) ) PU = PU.*(U{d}'*priorSoln{d+1})'; end 
  % accumulate projection onto the weighting
  if ( ~isempty(tstInd) )    WU2= tprod(WU2,[1:d-1 -d d+1:nd],U{d}.^2,[-d]); end;
end
% compute an updated "optimal" scaling
S = AU; 
if ( ~isempty(tstInd) )  S = S./WU2; end;
if ( S<0 ) S=abs(S); U{1}=-U{1}; end; % ensure positive
if ( isempty(tstInd) )
    sse = A2 - 2*AU*S + S.^2;
else
    sse = A2 - 2*AU*S + S.^2*WU2;
end
ssep=  sse./A2;
J=sse;
if ( orthoPen>0 ) J=J+orthoPen*PU'*PU; end; % convergence on the modified objective
if ( opts.verb > 0 ) 
  fprintf('%2d)\t|S|=%8g\tsse=%8g\t%%sse=%8g\tJ=%8g\tdJ=%8g\tdeltaU=%8g\n',0,S,sse,ssep,J,0,0);
end   

deltaU=inf;
osse=sse; oJ=J; odeltaU=deltaU; madJ=J*2; Jopt=J; Sopt=S; Uopt=U;
for iter=1:opts.maxIter;
  Js(iter)=J; osse=sse; oJ=J; odeltaU=deltaU; oS=S; oU=U;
  
   AU = A; % Temp store of the accumulated product up to this dim
   if ( ~isempty(priorSoln) ) PU=ones(numel(priorSoln{1}),1); end; % proj on prior
   if ( ~isempty(tstInd) )    WU2=single(~tstInd); end; % proj on the weighting matrix
   deltaU=0;
   for d=1:nd;
      oUd=U{d};
      
      tAU=AU;  % A^{1:N} \prod_{j neq d} U_j
      if ( ~isempty(priorSoln) ) tPU=PU; end; % prod_{j neq d} Prior_j^r U_j
      if ( ~isempty(tstInd) )    tWU2=WU2; end; % prod_j neq d} W_j (U_j.^2)
      for d2=d+1:nd; % Cached [1:d-1] in AU so only mult the rest
         tAU=tprod(tAU,[1:d2-1 -d2 d2+1:nd],U{d2},[-d2]);
         % accumulate projection onto the prior components
         if ( ~isempty(priorSoln) ) tPU = tPU.*(U{d2}'*priorSoln{d2+1})'; end 
         % accumulate projection onto the weighting
         if ( ~isempty(tstInd) )    tWU2= tprod(tWU2,[1:d2-1 -d2 d2+1:nd],U{d2}.^2,[-d2]); end;
      end
      
      % version if the A hasn't already been deflated
      % U{d}=shiftdim(tAU) - U{d}*repop(priorSoln{1},'*',tPU); % U{d} = A^{1:N} \prod_{j neq d} U_j
      U{d}=shiftdim(tAU); % U{d} = A^{1:N} \prod_{j neq d} U_j      
      if ( ~isempty(tstInd) ) % compensate for the missing values
        U{d} = U{d}./shiftdim(tWU2);
      end
      S   =norm(U{d});
      U{d}=U{d}/S;  % Normalise the final vector
      if ( orthoPen > 0 ) % deflate anything in previous directions
          % include the effect of the orthogonalisation penalty
          if( size(priorSoln{d+1},1)<size(priorSoln{d+1},2) ) % efficient when d>>rank
            U{d} = (eye(size(U{d},1)) + orthoPen/S*priorSoln{d+1}*repop(tPU.^2,'.*',priorSoln{d+1}'))\U{d};
          else
            % Computationally efficient inverse using searle identity, when rank<<d
            %  N.B. Unfortunately it's numerically unstable!... so we use the more expensive pinv to solve it
            % weighting over components -- i.e. projection onto the prior basis
            proj = pinv(eye(size(priorSoln{d+1},2))+orthoPen/S*(priorSoln{d+1}'*priorSoln{d+1}).*(tPU*tPU'))*repop(orthoPen/S*tPU.^2,'.*',(priorSoln{d+1}'*U{d}));
            U{d} = U{d} - priorSoln{d+1}*proj;
          end
          U{d} = U{d}./sqrt(U{d}'*U{d}); % normalise again          
        end % if orthoPen
      S   =norm(U{d});
      U{d}=U{d}/S;  % Normalise the final vector

      if ( strcmp('nnls',lower(alg{min(numel(alg),d)})) ) % impose positivity constraint
        U{d}(U{d}<0)=abs(U{d}(U{d}<0));
      end

      if ( opts.verb>1 ) % report the intermediate J value
        % include this dim in the accumulation info
        tAU=tprod(tAU,[1:d-1 -d d+1:nd],U{d},[-d]);        
        sse = A2 - 2*tAU*S + S.^2;
        ssep= sse./A2;
        J=sse; 
        if ( orthoPen>0 ) J=J+orthoPen*tPU'*tPU; end; % convergence on the modified objective
        dJ=oJ-J;
        fprintf('%2d.%2d)\t|S|=%8g\tsse=%8g\t%%sse=%8g\tJ=%8g\tdJ=%8g\tdeltaU=%8g\n',iter,d,S,sse,ssep,J,dJ,deltaU);
      end
      
      deltaU = deltaU+sum(abs(oUd(:)-U{d}(:)));
      AU = tprod(AU,[1:d-1 -d d+1:nd],U{d},[-d]); % update the accumulated info
      if ( ~isempty(priorSoln) ) PU = PU.*(U{d}'*priorSoln{d+1})'; end % acc proj on prior 
      if ( ~isempty(tstInd) )    WU2= tprod(WU2,[1:d-1 -d d+1:nd],U{d}.^2,[-d]); end; % acc proj on weight
   end   
   
   S  = AU;
   if ( ~isempty(tstInd) ) S=S./(WU2+1e-6); end % compensate for missing values
   
   % compute the error for this solution (fast way)
   if ( isempty(tstInd) )
     sse = A2 - 2*AU*S + S.^2;
   else
     sse = A2 - 2*AU*S + S.^2*WU2;
   end
   ssep= sse./A2;     
   J=sse; 
   if ( orthoPen>0 ) J=J+orthoPen*PU'*PU; end; % convergence on the modified objective

   if ( J<Jopt ) Jopt=J; Sopt=S; Uopt=U; end % track the best solution so far
   dJ = (oJ-J);
   if ( iter==1 )   deltaU0=max(deltaU,opts.tol0);  dJ0=max(abs(dJ),opts.objTol0); madJ=max(oJ,J)*2;
   elseif (iter<3 ) deltaU0=max(deltaU,opts.tol0);  dJ0=max(abs(dJ),opts.objTol0); madJ=max(madJ,J*2);
   end;
   madJ =madJ*(1-opts.marate)+dJ(1)*(opts.marate); % move-ave obj est
   if ( opts.verb > 0 ) 
      fprintf('%2d)\t|S|=%8g\tsse=%8g\t%%sse=%8.4g\tJ=%8g\tdJ=%8g\tdeltaU=%8g',iter,S,sse,ssep,J,dJ,deltaU);
   end   
   if ( deltaU < opts.tol    || deltaU < deltaU0*opts.tol0 || ...
        (iter>20 && ( madJ < opts.objTol || madJ   < dJ0*opts.objTol0 ) ) || ... % long term, nowt, or increase=bad
        abs(dJ)< opts.objTol || abs(dJ)< dJ0*opts.objTol0 ) % long term, nowt, or increase=bad
      break; 
   end;

   % update the lagrange multipliers
   if ( ~isempty(orthoLambda) && ((opts.orthoLambda>0 && mod(iter,opts.orthoLambda)==0 && deltaU<1) || (opts.orthoLambda<0 && deltaU<-opts.orthoLambda)) )
     %lambdaStep  = lambdaStep*.99;
     orthoLambda = orthoLambda + orthoPen*PU*lambdaStep;
     if ( opts.verb>0 ) fprintf('%d) lambda update -> [%s]',iter,sprintf('%g,',orthoLambda)); end;
   end
   if ( opts.verb>0 ) fprintf('\n'); end;
end
if ( opts.verb>0 ) fprintf('\n'); end;
S=Sopt; U=Uopt;
% ensure solution is positive
if ( S<0 ) U{1}=-U{1}; S=-S; end;

if ( opts.verb >= 0 ) 
  if ( isempty(tstInd) )
    % compute performance of the final solution
    AU=A;  % A^{1:N} \prod_{j neq d} U_j
    if ( ~isempty(priorSoln) ) PU=ones(numel(priorSoln{1}),1); end; % proj on prior
    for d=1:nd;
      AU=tprod(AU,[1:d-1 -d d+1:nd],U{d},[-d]);
      % accumulate projection onto the prior components
      if ( ~isempty(priorSoln) ) PU = PU.*(U{d}'*priorSoln{d+1})'; end 
      % accumulate projection onto the weighting
      if ( ~isempty(tstInd) )    WU2= tprod(WU,[1:d-1 -d d+1:nd],U{d}.^2,[-d]); end;
    end
    if ( isempty(tstInd) )
      sse = A2 - 2*AU*S + S.^2;
    else
      sse = A2 - 2*AU*S + S.^2*WU2;
    end
    ssep=  sse./A2;
    J=sse;
    if ( orthoPen>0 ) J=J+orthoPen*PU'*PU; end; % convergence on the modified objective
    fprintf('%2d)\t|S|=%8g\tsse=%8g\t%%sse=%8.4g\tJ=%8g\n',iter,S,sse,ssep,J);
  else
    Sall=S; Uall=U;
    if ( ~isempty(priorSoln) ) 
      Sall=[priorSoln{1}; Sall]; for d=1:nd; Uall{d}=[priorSoln{d+1} Uall{d}]; end;
    end
    Ae  = parafac(Sall,Uall{:}); % estimated solution
    Err = oA-Ae;
    sse = [sum(Err(~tstInd).^2); sum(Err(tstInd).^2)]; % train/test perf
    clear Ae Err;
    ssep= sse./[sum(oA(~tstInd).^2);sum(oA(tstInd).^2)];
    J=sse(1);
    if ( orthoPen>0 ) J=J+orthoPen*PU'*PU; end; % convergence on the modified objective
    fprintf('%2d)\t|S|=%8g\tsse=%8g/%8g\t%%sse=%8.4g/%8.4g\tJ=%8g\n',iter,S,sse,ssep,J);    
  end
  if ( 0 && ~isempty(orthoLambda) ) fprintf('%d) lambda update -> [%s]\n',iter,sprintf('%g,',orthoLambda)); end;
end

obj=[J;sse];
varargout={U{:} obj};
return;

function [A]=parafac(S,varargin);
% Compute the full tensor specified by the input parallel-factors decomposition
%
% [A]=parafac(S,U_1,U_2,...);
%
U=varargin;
if ( nargin==1 && iscell(S) ) U=S(2:end); S=S{1}; end;
if ( numel(U)==1 && iscell(U{1}) ) U=U{1}; end;
if ( numel(S)==1 && size(U{1},2)>1 && S==1 ) S=ones(size(U{1},2),1); end;
nd=numel(U); A=shiftdim(S(:),-nd);  % [1 x 1 x ... x 1 x M]
for d=1:nd; A=tprod(A,[1:d-1 0 d+1:nd nd+1],U{d},[d nd+1],'n'); end
A=sum(A,nd+1); % Sum over the sub-factor tensors to get the final result
return;



%------------------------------------------------------------------------------
function testCase()
A = randn(11,11,11); A = cumsum(cumsum(A,1),2); % add some structure
[S,U{1:ndims(A)}]=hoPM(A,'verb',1,'rank',5);

% try incremental version
nd=ndims(A); rank=5; S=[]; U={};
[S,U{1:nd}]  =hoPM(A,'verb',0);
rS=S; rU=U;
Sr=[];Ur={};
for r=2:rank;
  [Sr,Ur{1:nd}]  =hoPM(A,'priorSoln',{S U{:}},'verb',1,'tol',1e-2);
  S=[S;Sr]; for d=1:nd; U{d}=[U{d} Ur{d}]; end;
  %rA  = A-parafac(rS,rU{:});
  %[Sr2,Ur2{1:nd}]  =hoPM(rA,'verb',1);  
  %rS=[rS;Sr2]; for d=1:nd; rU{d}=[rU{d} Ur2{d}]; end;
end
[sse,ssep]=parafacSSE(A,S,U{:})
[sse,ssep]=parafacSSE(A,rS,rU{:})

% test if computing the deflation in parts works
Sp=S(1:end-1); St=S(end); Up={}; Ut={}; for d=1:nd; Up{d}=U{d}(:,1:end-1); Ut{d}=U{d}(:,end); end;
Ar  = A-parafac(Sp,Up{:});
AU=A; ArU=Ar; PU=Sp;
for d=[2:nd];  
  AU = tprod(AU,[1:d-1 -d d+1:nd],Ut{d},-d);
  PU = PU.*(Up{d}'*Ut{d});
  ArU= tprod(ArU,[1:d-1 -d d+1:nd],Ut{d},-d);
end
[AU-PU*Up{1}-ArU]
AU-sum(PU), ArU

% try parafac_als directly
[Sp,Up{1:ndims(A)}]=parafac_als(A,'rank',rank,'verb',1);

% try with non-negativity constraint
[S,U{1:nd}]  =hoPM(A,'verb',0,'alg',{[] [] 'nnls'});

% test increasing penalty version
tic,seed={}; orthoPens=1e4*10.^(0:3); for pen=orthoPens; [seed{1:4}]  =hoPM(A,'priorSoln',priorSoln,'seed',seed,'verb',1,'objTol0',1e-5,'orthoPen',pen); end;toc


% test with excluded points
trnInd=randn(size(A))>0;
nd=ndims(A); rank=5; S=[]; U={};
[S,U{1:nd}]  =hoPM(A,'verb',0,'wght',trnInd);
Sr=[];Ur={};
for r=2:rank;
  [Sr,Ur{1:nd}]  =hoPM(A,'priorSoln',{S U{:}},'verb',1,'wght',trnInd);
  S=[S;Sr]; for d=1:nd; U{d}=[U{d} Ur{d}]; end;
end
% compute the train/test performance
Aerr=A-parafac(S,U{:});
trnsse=sum(Aerr(trnInd).^2); trnssep=trnsse./sum(A(trnInd).^2);
tstsse=sum(Aerr(~trnInd).^2);  tstssep=tstsse./sum(A(~trnInd).^2);
% compute degeneracy info
UU=1; for d=1:numel(U); UU=UU.*(U{d}'*U{d}); end;  degen=UU-eye(size(UU));   
fprintf('|S|=%5.2f\tdgn=%5.3f\tsse=%6g/%6g\t%%sse=%6g/%6g\n',sum(abs(S)),sum(abs(degen(:)))./size(degen,1),trnsse,tstsse,trnssep,tstssep);


% test missing value completion method...
szA=[10 21];
xtrue=randn(szA(1),1); ytrue=randn(szA(2),1);
A=xtrue*ytrue';
trnInd=randn(size(A))>0;
ytst=randn(size(ytrue)); xtst=randn(size(xtrue));
%xtst=mean(A,2);  ytst=mean(A,1)'; 
xtst=xtst./norm(xtst);  ytst=ytst./norm(ytst); stst=norm(A0);

% minimise - direct optimisation
W   =trnInd;
x   =xtst; y=ytst; s=stst;
A0  =A; A0(W==0)=0;
J    =A0(:)'*A0(:) - 2*x'*A0*y*s + s.^2*(x.^2)'*W*(y.^2);
fprintf('%2d) |S|=%g\tJ=%g\n',0,s,J);
for i=1:10;
  x = (A0*y)./(W*(y.^2));
  x = x./norm(x);
  y = ((x'*A0)./((x.^2)'*W))';
  s    = norm(y);
  y = y./norm(y);
  % Aest = x*y';
  % Err  = A-Aest;
  % J    = sum(Err(W).^2); % direct
  s    = (x'*A0*y)./((x.^2)'*W*(y.^2));
  J    = A0(:)'*A0(:) - 2*x'*A0*y*s + s.^2*(x.^2)'*W*(y.^2);
  fprintf('%2d) |S|=%g\tJ=%g\n',i,s,J);
end
xdir=x;ydir=y;

% tensor version
szA=[20 21 6];
xtrue=randn(szA(1),1); ytrue=randn(szA(2),1); ztrue=randn(szA(3),1);
A=parafac(1,xtrue,ytrue,ztrue);
trnInd=randn(size(A))>-0; 
fprintf('%%trn=%g\t0s [d1,d2,d2]=[%d,%d,%d]\n',sum(trnInd(:))./numel(trnInd),...
        sum(sum(sum(trnInd,1)==0)),sum(sum(sum(trnInd,2)==0)),sum(sum(sum(trnInd,3)==0)));
ytst=randn(size(ytrue)); xtst=randn(size(xtrue)); ztst=randn(size(ztrue));
%xtst=mean(A,2);  ytst=mean(A,1)'; 
xtst=xtst./norm(xtst);  ytst=ytst./norm(ytst); ztst=ztst./norm(ztst); stst=norm(A0);

[S,U{1:nd}]  =hoPM(A,'verb',1,'wght',trnInd,'seed',{stst xtst ytst ztst},'rewghtStep',0);



% minimise - imputation
W   =trnInd;
x   =xtst; y=ytst; s=stst;
A0 = A; A0(W==0)=0; Ai=A0;
J    =A0(:)'*A0(:) - 2*x'*A0*y*s + s.^2*(x.^2)'*W*(y.^2);
fprintf('%2d) |S|=%g\tJ=%g\n',0,s,J);
for i=1:30;
  x = Ai*y;
  x = x./norm(x);
  y = (x'*Ai)';
  s    = norm(y);
  y = y./norm(y);
  if ( mod(i,1)==0 ) % missing value replacement & true perf est
    Aest = s*x*y';
    Err  = A-Aest;
    J    = [sum(Err(W).^2) sum(Err(W==0).^2)]; % direct    
    Ai(W==0)=Aest(W==0); 
  else % normal perf est
    J = A0(:)'*A0(:) - 2*x'*A0*y*s + s.^2*(x.^2)'*W*(y.^2);
  end 
  fprintf('%2d.2) |S|=%g\tJ=%g/%g\n',i,s,J);
end
ximp=x;yimp=y;


