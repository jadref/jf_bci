function [w,b,alpha,svs]=adatron(X,Y,wght,pen,alpha,b,termParms,verb)
% function [w,b,alpha,svs]=adatron(X,Y,wght,pen,alpha,b,termParms,verb)
% An implementation of the adatron algorithm, to solve the problem:
% L = w'w + C*\sum_i\eta_i -\sum_i d_i alpha_i(y_i*(x'_iw+b)-1+\eta_i), 
%                                                                 s.t.\eta_i>=0
% Using a Gaus-siedel style gradient descent on the paired primal and dual 
% variables, w and alpha, as
% dL/dalpha = y_i(x_i w+b)-1+eta_i) = max(0,1-y_i (x'_i w + b))
% where, from the dual formulation: w=\sum_i alpha_i y_i x_i
% Hence, when we update alpha we must also update w.
%
% N.B. We use an adaptive learning rate, nu/K(x,x) to guarantee and maximise
%      convergence speed where the nu must be in the range (2,0), def (1)
% 
% N.B. to compute the bias we add an extra bias dimension to feature space
% (which is equivalent to adding 1 to the kernel matrix) and then setting
% b=\sum_i alpha_i y_i 
%
% N.B. The alternative of optimising b directly in the lagrangian \sum_i
% alpha_i d_i y_i =0 constraint which we update using the secant method as
% proposed in "Simple learning algorithms for training SVMs", by Colin
% Cambell, Nello Cristianini, U.Bristol Tech Report.  However this was found
% to be *very* unstable!
% 
% (Note imposing this constraint directly on \alpha gives SMO!)
%
% SMO style heuristics are used to to speed up convergence.  where we
% iterate over SV's (alpha~=0) until convergence, before iterating over all
% points to get new SV's the SV's only etc.
%
% N.B. 2-norm loss is implented by adding C to the diagonal of K & pen=0
%Inputs:
% X=[Nxd] Y=[Nx1] wght=importance weight for each point [Nx1]
% w=[dx1] b=[1x1] -- initial solution, and it's dual weights alpha, [Nx1]
% pen - penalty C for the loss, or if 2 el vector, pel for [Y==-1,Y==1]
% termParms-termination parameters [maxIt tol SVsit SVstol]([100,1e-4,80,1e-2])
% verb -- verbosity level
% 
% $Id: adatron.m,v 1.9 2007-03-06 18:56:08 jdrf Exp $
t0=clock;
MARGINTOL=1e-5; 
if ( nargin < 4 ) error ('Insufficient Arguments'); end;
if ( nargin < 5 ) alpha=[]; end
if ( nargin < 6 ) b=[]; end
if ( nargin < 7 | isempty(termParms) ) termParms=[100,5e-4,80,1e-3]; 
else
  if ( length(termParms) < 2 ) termParms(2)=1e-3;end;
  if ( length(termParms) < 3 ) termParms(3)=10;end;
  if ( length(termParms) < 4 ) termParms(4)=termParms(2);end;
end
if ( nargin < 8 | isempty(verb) ) verb=0; end;
[N,dim]=size(X);
labels=unique(Y);
if ( length(labels) == 2 && any(labels'~=[-1 1]) )%convert to +1/-1 format
  fprintf('Converting Y to +1/-1 format\n');
  Y=single(Y);Y(Y==labels(1))=-1; Y(Y==labels(2))=1;
end

% pre-include importance so wght=wght*pen;
if ( ~isempty(wght) ) 
   if ( max(size(pen))==1 ) wght=wght*pen; 
   else wght=wght.*(Y==-1).*pen(1)+wght.*(Y==1).*pen(2); % per-lab weight
   end
else 
   if ( max(size(pen))==1 ) wght=pen(ones(size(Y))); 
   else wght=pen(1).*(Y==-1)+pen(2).*(Y==1);             % per-lab
   end;
end
% N.B. alpha and w,b have to be updated together
if ( isempty(b) ) b=0; end;
if ( isempty(alpha) ) 
  alpha=zeros(N,1); w=zeros(dim,1); b=0;
else % compute the corresponding w
  w=X'*(alpha.*Y); b=alpha'*Y;
end;
dalpha=zeros(size(alpha)); svs=[]; midPt=floor(N/2);

X2=sum(X.*X,2)+1; % pre-compute K(x,x)+1, needed for the optimal step size

% BODGE: ignore the input bias, as it messes up convergence!
db=0.1; alphaY=alpha'*Y; % info for the bias update

% Loop over the data until we've converged
for outerIt=1:termParms(1);
  
  % Stage 1: first loop once over the entire data set.
  % N.B. this *must* be in a loop so we use previously updated alphas 
  %   in the computation of later alphas!
  % Alternate the order of processing points -- massively improves convergence
  switch mod(outerIt,4); 
   case 0; prm=1:N; % -> ->
   case 1; prm=[midPt-1:-1:1 N:-1:midPt ];%prm=N:-1:1;  % <- <-
   case 2; prm=[midPt-1:-1:1 midPt:N ]; % <- -> 
   case 3; prm=[1:midPt-1 N:-1:midPt];  % -> <-
  end; %prm=randperm(N);
  for i=1:N; idxs=prm(i);

    grad=1-Y(idxs).*(X(idxs,:)*w+b);                   % gradient

    % compute constrained update: constrain pen * wght_i >= alpha_i >= 0 
    dalpha(idxs)=min(max(grad./X2(idxs),-alpha(idxs)),wght(idxs)-alpha(idxs)); 

    alpha(idxs) = alpha(idxs)+dalpha(idxs) ;           % update alpha
    w = w + X(idxs,:)'*(dalpha(idxs).*Y(idxs));        % update w
    b = b + dalpha(idxs)'*Y(idxs);                     % update b
    % alphaY=alphaY+dalpha(idxs)'*Y(idxs);              % update alphaY

  end

  % Stage 1.2: use the secant method to find the bias/lagrange mult
%   oalphaY=alphaY;alphaY=alpha'*Y;dalphaY=alphaY-oalphaY; % del constraint
%   if ( abs(dalphaY)>0 ) % limit step size, to ensure convergence.
%     db=-alphaY*db/sign(dalphaY)/max(abs(dalphaY),1);  % secant step
%   else
%     db=-alphaY*1e-3;
%   end
%   b=b+db;                      % new value  
  
  % Print out debug info!
  %figure(100);plot(alpha,'b');hold on; plot(dalpha,'r');
  if ( verb>0 )
    f=X*w+b; r=Y-f;% prediction and residual    
    % Regularlised loss
    err=Y.*r; pts=(err>-MARGINTOL);R(outerIt)=w'*w + sum(wght(pts).*err(pts));
    % Performance
    [nErr,binCls(outerIt,:),eeoc(outerIt,:)]= dv2conf(f,Y,[],[],verb);
    fprintf('%d) nSVs=%d  |w|^2=%g  #,s Err=%d,%g  L=%g PP=%g tol=%g\n',...
            outerIt,sum(abs(alpha)>0),w'*w,sum(err>1),sum(err(pts)),...
            R(outerIt),binCls(outerIt,1),norm(dalpha)/N);
    if ( verb > 1 ) 
      figure(1);hold off; plot(1-binCls(:,1)); hold on; plot(R/R(1),'r');
      figure(2);hold off; plot([Y(Y==1);Y(Y==-1)],'r.'); 
      hold on; plot([f(Y==1);f(Y==-1)],'b');
      %figure(3);hold off; plot(R);
      figure(floor((outerIt-1)/9)+3);subplot(3,3,mod(outerIt-1,9)+1); 
      if ( dim > 1 )      
        hold off; labScatPlot(X',Y,alpha); hold on;
        drawLine(w,b,min(X),max(X)); hold on;
        drawLine(w,b+1,min(X),max(X),'r-'); hold on;
        drawLine(w,b-1,min(X),max(X),'r-'); hold on;
      else
        hold off; plot(X,Y(:,1),'g.');
        hold on; drawLine([alpha -1],b,min([X Y]),max([X Y]));
      end 
      pause(0.05);
    end
  end  

  if ( norm(dalpha)/N < termParms(2) ) break; end; % tolerance test
  
  
  % Stage 2: now loop over only SVs until convergence.
  for innerIt=1:termParms(3);

    svs=find(alpha>0); nSVs=length(svs); % only re-process the current svs set
    
    % Randomise the order of processing points-- massively improves convergence
    prm=randperm(nSVs);

    for i=1:nSVs; idxs=svs(prm(i));

      grad=1-Y(idxs).*(X(idxs,:)*w+b);                % gradient

      % compute constrained update: constrain pen * wght_i >= alpha_i >= 0 
      dalpha(idxs)=min(max(grad./X2(idxs),-alpha(idxs)),wght(idxs)-alpha(idxs)); 

      alpha(idxs) = alpha(idxs)+dalpha(idxs) ;           % update alpha
      w = w + X(idxs,:)'*(dalpha(idxs).*Y(idxs));        % update w
      b = b + dalpha(idxs)'*Y(idxs);                     % update b
      %aphaY=alphaY+dalpha(idxs)'*Y(idxs);                % update alphaY
    
    end

    % Stage 1.2: use the secant method to find the bias lagrange mult
%     oalphaY=alphaY;alphaY=alpha'*Y;dalphaY=alphaY-oalphaY; % del constraint
%     if ( abs(dalphaY)>0 ) % limit step size, to ensure convergence.
%       db=-alphaY*db/sign(dalphaY)/max(abs(dalphaY),1);% secant step
%     else
%       db=-alphaY*1e-3;
%     end
%     b=b+db;                           % new value

    % tolerance test
    if ( norm(dalpha(svs))/nSVs < termParms(4) ) break; end; 
  end

end
svs=find(alpha>0);
if ( verb ) fprintf('Tot Time %g\n',etime(clock,t0)); end;

%TESTCASE:
%
% [X,Y]=mkMultiClassTst([-1 0;1 0; .2 .5],[400 400 100],[.3 .3; .3 .3; .2 .2]);
% Y=2*(Y==1)-1;
% [w b alpha svs]=adatron(X,Y,[],1,[],[],[],1);
% labScatPlot(X,Y);hold on;drawLine(w,b,min(X),max(X));
% drawLine(w,b-1,min(X),max(X),'r-');drawLine(w,b+1,min(X),max(X),'r-');
%
%conf =    500     0
%             2   398

% Bin:		0.99778 0.99778 / 0.99778
% EEOC:		0.99775 0.99775 / 0.99775
% AUC: 		0.99748 0.99799 / 0.99774
% 4) nSVs=29  |w|^2=12.0654  sErr=11.5468  L=23.6122 PP=0.997778 tol=0.000346515%
% TESTCASE2: with real data!
% X=importdata('~/temp/diabetes.sdata.txt');Y=2*(X(:,end)>0)-1;X=X(:,1:end-1);
% [w b alpha svs]=adatron(X,Y,[],1,[],[],[],1);
% conf =    205    42
%            63   132

% Bin:		0.76244 0.76244 / 0.76244
% EEOC:		0.74885 0.74885 / 0.74885
% AUC: 		0.82375 0.82483 / 0.82429
% 5) nSVs=361  |w|^2=73.0927  sErr=279.694  L=352.787 PP=0.762443 tol=0.000461964
