function [alphab,C]=evidKLR(K,Y,varargin)
% [alpha,C]=evidKLR(K,Y,varargin)
% Evidence Optimised Regularised Kernel Logistic Regression using a
% pre-conditioned conjugate gradient solver so extends to large input
% kernels.
% 
% Optional arguments:
%  alphab      - initial guess at the kernel parameters       [N+1x1]
%  maxEvidIter - max number of evidence CG steps to do        (100)
%  Evidtol     - absolute error tolerance                     (.1)
%  CTol        - accuracy to find C to                        (1e-4)
%  {step,minStep,maxStep} - parameters for the line search. 
%  approxUpdate - use the approximate lambda update equation 
%                 (no-hessian or inverse required)

opts=struct('C',1,'alphab',[],'maxEvidIter',100,'evidTol',1e-1,'approxUpdate',0,'lineSearch','secant','CTol',1e-4,'step',1,'verb2',1,'minStep',.05,'maxStep',2);
recopts=[];
if(~isempty(varargin) && isstruct(varargin{1})) % struct->cell
   varargin=[fieldnames(varargin{1})'; struct2cell(varargin{1})'];
end
mOpts=false(numel(varargin),1);
for i=1:2:numel(varargin); % process the options 
   if( isfield(opts,varargin{i}) ) % leave unrecognised entries
      opts.(varargin{i})=varargin{i+1};
      mOpts(i:i+1)=true;
   end;
end
varargin(mOpts)=[]; % remove processed arguments

[dim,N]=size(K); 

alphab=opts.alphab;
C=opts.C(1); if ( C==0 ) C=1; end;
logC=log(C); 

J2=inf; dJ2dlogC=0; oJ2=J2; odJ2dlogC=dJ2dlogC; ologC=logC-1; % init bracket=1
for it=1:opts.maxEvidIter;
   
   % use KLR to find the optimal solution with this C
   [alphab,J]=klr_cg(K,Y,C,'alphab',alphab,varargin{:});
   
   % update C
   % first compute the KLR parameters
   alphaK = alphab(1:end-1)'*K;
   g      = 1./(1+exp(-Y'.*(alphaK+alphaK(end)))); g=max(g,eps); % stop log 0
   Ed     = -sum(log(g));  % P(D|w,b,fp)
   Ealpha = alphaK*alphab(1:end-1);    % P(w,b|R);
   neglogpost = Ed + logC*Ealpha; % fx=neg log posterior
   
   Yerr= Y'.*(1-g);
   wght= g.*(1-g);
   % N.B. for numerical issues we transform the hessian determinant as:
   % |H| = |K( C(1) + W*K )| = |K W^.5( C(1) + W^.5 K W^.5 ) W^-.5| 
   %     = |K| |C(1)+W^.5 K W^.5|
   % Thus,
   % d ln(|H|)/d(C(1))= d ln(|C(1)+W.^.5 K W.^.5|)/d(C(1)) ...
   %                  = tr( [C(1)+W.^.5 K W.^.5]^-1 )
   % hence we can work with just the matrix,
   %  A = W*K+C(1)*I
   % which is guaranteed to have eigenvalues greater than 1
   % N.B. Compute H from A using:  H   = C(1)*K*W.^.5*B*W.^-.5;
   sqrtwght=sqrt(wght);
   if ( exist('repop')>1 ) % use repop if available \approx 5x faster!
      B=speye(N,N)*C + repop(repop(sqrtwght,K,'.*'),sqrtwght','.*');
   else % fall back on matlab! Use sparse matrices for memory savings.
      Dsqrtwght=spdiags(sqrtwght,0,N,N);
      B=speye(N,N)*C + Dsqrtwght*K*Dsqrtwght;
   end
   
   if ( opts.approxUpdate ) % N.B. this isn't any faster!      
      [trinvA,ans,ans,mineig,maxeig]=trinvBound(B);
      logdetA=trlnBound(B,mineig,maxeig);        
   
   else  
      R=chol(B);
      logdetA=2*sum(log(diag(R))); %trace(log(-KH));
      
      % use the mex if its available
      if ( exist('invtriu')>1 ) invR=invtriu(R); else invR=R\eye(size(K)); end
      
      trinvA=invR(:)'*invR(:); % sum(invR(:).*invR(:));
   
   end

   % Compute the evidence and its gradient
   J2           = Ed + C*Ealpha - dim/2*log(C) + logdetA/2;
   dJ2dlogC=      C*Ealpha     - dim/2         + C*trinvA/2;
   if ( opts.verb2>0 )
     fprintf('Evid %2d)  C=%5.3g   nll=%7g  -logevid=%7g   d-logevid=%7g\n',...
              it,C,J,J2,dJ2dlogC);
     if ( opts.verb2 > 1 ) 
        hold on; plot(logC,dJ2dlogC,'*'); 
        text(logC,dJ2dlogC,num2str(it)); drawnow;
     end
   end
   
   % update the C
   if ( it>1 && ~isempty(opts.lineSearch) ) % use line search when have 2 pts
      nlogC=secant(logC,J2,dJ2dlogC,ologC,oJ2,odJ2dlogC,opts.step);      
      if( oodJ2dlogC*odJ2dlogC < 0 & odJ2dlogC*dJ2dlogC>0 & ...
          (nlogC<min(ologC,logC) | nlogC>max(ologC,logC)) ) % bracketing check
         nlogC = secant(logC,J2,dJ2dlogC,oologC,ooJ2,oodJ2dlogC,opts.step);
         ologC=oologC;ooJ2=oJ2;oodJ2dlogC=odJ2dlogC; % swap pts so keep bracket
      end
         
   else % at the first step use the MacKay approx for the evidence update
      nlogC=log(dim./(2*Ealpha+trinvA));      
      
   end
   % ensure step isn't to big/small
   if(logC<ologC)l=logC;u=ologC;else l=ologC;u=logC; end; w=u-l;
   if ( nlogC>=l & nlogC<=u )   % should ensure sufficient step size
      nlogC=max(min(nlogC,u-opts.minStep*w),l+opts.minStep*w); 
   elseif ( nlogC<l ) % extrap down
      nlogC=min(max(nlogC,l-w*opts.maxStep),l-w*opts.minStep);
   else % extrap up
      nlogC=max(min(nlogC,u+w*opts.maxStep),u+w*opts.minStep);
   end
   
   % record the old values
   ooJ2=oJ2;oJ2=J2;oodJ2dlogC=odJ2dlogC;odJ2dlogC=dJ2dlogC;
   oologC=ologC;ologC=logC;   
   % update to the new value
   logC =max(nlogC,log(eps));
   C    =exp(logC);
   
   % convergence tests
   if ( abs(logC-ologC) < opts.CTol ) % delta logC
      fprintf('C tol convergence\n');   break; 
   end; 
   if ( abs(dJ2dlogC) < opts.evidTol )  % evidence gradient mag
      fprintf('evidTol convergence\n');      break; 
   end; 
   
end
return


function [xt]=secant(x0,f0,df0,x1,f1,df1,step)
% secant search for the zero's
invH = (x1-x0)/(sign(df1-df0)*max(abs(df1-df0),eps));
xt   = x1 - step*invH*df1;
return;

%----------------------------------------------------------------------------
function testcase()
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2],[],[-1 1 1]);
K=X*X'+eye(size(X,1));

fIdxs=gennFold(Y,10); trnIdx=any(fIdxs(:,1:9),2); tstIdx=any(fIdxs(:,10),2); % get train/test folds
[alpha,lambda]=evidKLR(K(trnIdx,trnIdx),Y(trnIdx)); % call the optimizer

dv=K(tstIdx,trnIdx)*alpha;
dv2conf(dv,Y(tstIdx))

% overfitted linear seperable problem?
[X,Y]=mkMultiClassTst([-1 0; 1 0; -.4 1.25; .4 -1.25],[200 200 1 1],[.3 .3; .3 .3; .0 .0; .0 .0],[],[-1 1 1 -1]);

z=vgrid_prep(bci('eeg/vgrid/felix/lettergrid/AT_S1'));
z=reserve(z,.25);
X=z.x; y=z.y;
trnIdx=false(ntrials(z),1);trnIdx(getouterfold(z))=true; tstIdx=~trnIdx;
[N dim samp]=size(X);

wb=([X(:,trnIdx);ones(1,sum(trnIdx))]*alpha); % extract weights from lin
dv=wb(1:end-1)'*X(:,tstIdx)+wb(end);
sdv=1./(1+exp(dv));     % compute the probabilities
dv2conf(dv,Y(tstIdx))   % performance test