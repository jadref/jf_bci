function [alpha,b,svs]=kerAdatron(K,Y,wght,pen,alpha,termParms,verb,X)
% function [alpha,b,svs]=adatron(K,Y,wght,pen,alpha,tol)
% An implementation of the kernel-adatron algorithm, to solve the problem:
% min_w w'w + C*\sum_i\eta_i -\sum_i d_i alpha_i y_i*(\phi'(x_i)w+b), 
%                                                                 s.t.\eta_i>=0
% where d_i is a sample importance weight.
% Using a Gaus-siedel style gradient descent on the Lagrangian dual problem:
%  min_alpha \sum_i alpha_i - (alpha.Y.d)'K*(alpha.Y.d)  
%                                       s.t. C>=alpha>=0,\sum_i alpha_i y_i=0
% which has gradient d(alpha_d) = 1 - \sum_i alpha_i Y_i d_i K(x_i,x_d)
% where to simplify matters we ignore the \sum_i alpha_i d_i y_i =0 constraint
% (which is equivalent to assuming b=0) 
% (Note if we imposed this constraint we end up with SMO!)
%
% N.B. to compute the bias we add an extra bias dimension to feature space
% (which is equivalent to adding 1 to the kernel matrix) and then setting
% b=\sum_i alpha_i y_i 
%
% N.B. The alternative of optimising b directly in the lagrangian \sum_i
% alpha_i d_i y_i =0 constraint which we update using the secant method as
% proposed in "Simple learning algorithms for training SVMs", by Colin
% Cambell, Neloo Cristianini, U.Bristol Tech Report.  However this was found
% to be *very* unstable!
% 
% (Note imposing this constraint directly on \alpha gives SMO!)
%
% For efficiency reasons we use SMO style heuristics to speed up convergence.
% where we iterate over SV's (alpha~=0) until convergence, before iterating 
% over all points to get new SV's the SV's only etc.
%
% N.B. 2-norm loss is implented by adding C to the diagonal of K.
%
%Inputs:
% K=[NxN] Y=[Nx1] w= importance weight for each point [Nx1]
% pen - penalty C for the loss
% nu  - the learning rate should be 0 < nu < 2                           (1)
% termParms-termination parameters [maxIt tol SVsit SVstol]([100,1e-3,60,1e-2])
%
% $Id: kerAdatron.m,v 1.5 2007-03-06 18:56:08 jdrf Exp $
MARGINTOL=1e-5;
STEPDECAY=1;
C2=1;
if ( nargin < 4 ) error ('Insufficient Arguments'); end;
if ( nargin < 5 ) alpha=[]; end;
if ( nargin < 6 | isempty(termParms) ) termParms=[100,1e-3,80,1e-2]; 
else
  if ( length(termParms) < 2 ) termParms(2)=1e-3;end;
  if ( length(termParms) < 3 ) termParms(3)=10;end;
  if ( length(termParms) < 4 ) termParms(4)=termParms(2);end;
end
if ( nargin < 7 | isempty(verb) ) verb=0; end;
if ( ~isstruct(K) ) [N,dim]=size(K); else N=K.size(1);dim=K.size(2); end;
midPt=floor(N/2);
labels=unique(Y);
if ( length(labels) == 2 && any(labels'~=[-1 1]) )%convert to +1/-1 format
  fprintf('Converting Y to +1/-1 format\n');
  Y=single(Y);Y(Y==labels(1))=-1; Y(Y==labels(2))=1;
end

% pre-include importance so pen=wght*pen, if wanted!
if ( ~isempty(wght) ) wght=wght*pen; else wght=pen(ones(size(Y))); end;
if ( isempty(alpha) ) 
  alpha=zeros(N,1); b=0; 
else 
  b=alpha'*Y; 
end

dalpha=zeros(size(alpha)); db=0.1; alphaY=alpha'*Y; lr=1;% info for the bias update
for outerIt=1:termParms(1);

  % Stage 1: first loop once over the entire data set.
  % N.B. this *must* be in a loop so we use previously updated alphas 
  %   in the computation of later alphas! or nu << 1
%   for i=1:N/10+1; % do sets of 10 at a time....
%     idxs=(i-1)*10+1:min(i*10,N);

  % Alternate the order of processing points -- massively improves convergence
  switch mod(outerIt,4); 
   case 0; prm=1:N; % -> ->
   case 1; prm=[midPt-1:-1:1 N:-1:midPt ];%prm=N:-1:1;  % <- <-
   case 2; prm=[midPt-1:-1:1 midPt:N ]; % <- -> 
   case 3; prm=[1:midPt-1 N:-1:midPt];  % -> <-
  end; %prm=randperm(N);

  for i=1:N; idxs=prm(i);

    grad=1-Y(idxs).*(K(idxs,1:dim)*(Y.*alpha)+b/C2); % gradient

    % constrain pen*d_i >= alpha_i >= 0
    dalpha(idxs)=lr*min(max(grad./(K(idxs,idxs)+1),-alpha(idxs)),...
                        wght(idxs)-alpha(idxs)); 
    alpha(idxs) =alpha(idxs)+dalpha(idxs) ;            % update
    b = b + dalpha(idxs)'*Y(idxs)/C2;                     % update b    
    %aphaY=alphaY+dalpha(idxs)'*Y(idxs);                % update alphaY
  end

  % Stage 1.2: use the secant method to find the bias/lagrange mult
%   oalphaY=alphaY;alphaY=alpha'*Y;dalphaY=alphaY-oalphaY; % del constraint
%   if ( abs(dalphaY)>0 ) % limit step size, to ensure convergence.
%     db=-alphaY*db/sign(dalphaY)/max(abs(dalphaY),.5);  % secant step
%   else
%     db=-alphaY*1e-3;
%   end
%   b=b+db;                      % new value  

  % Print out debug info.
  %figure(100);plot(alpha,'b');hold on; plot(dalpha,'r');
  if ( verb>0 )
    f=K*(Y.*alpha)+b; % prediction
    % hyperplane w (assuming linear kernel)
    % Regularlised loss
    err=1-Y.*f;
    R(outerIt)=(alpha.*Y)'*K*(alpha.*Y)+pen*sum(err(err>-MARGINTOL));
    % Performance
    [nErr,binCls(outerIt,:),eeoc(outerIt,:)]= dv2conf(f,Y,[],[],verb);
    fprintf('%d) nSVs=%d  |w|^2=%g  #,s Err=%d,%g  L=%g PP=%g tol=%g\n',...
            outerIt,sum(abs(alpha)>0),(alpha.*Y)'*K*(alpha.*Y),...
            sum(err>1),sum(err(err>0)),R(outerIt),...
            binCls(outerIt,1),norm(dalpha)/N);
    if ( verb > 1 ) 
      if ( exist('X') ) w=X'*(Y.*alpha); else w=alpha; end; 
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

    svs=find(alpha>0); nSVs=length(svs);

    % Randomise the order of processing points-- massively improves convergence
    prm=randperm(nSVs);
    
    for i=1:length(svs); idxs=svs(prm(i));

       grad=1-Y(idxs).*(K(idxs,1:dim)*(Y.*alpha)+b/C2); % gradient

      % constrain pen*wght_i >= alpha_i >= 0
      dalpha(idxs)=lr*min(max(grad./(K(idxs,idxs)+1),-alpha(idxs)),...
                          wght(idxs)-alpha(idxs)); 
      alpha(idxs) = alpha(idxs)+dalpha(idxs) ;           % update alpha
      b = b + dalpha(idxs)'*Y(idxs)/C2;                     % update b    
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
    if ( norm(dalpha(svs))/length(svs) < termParms(4) ) break; end; 
  end
  oalpha=alpha;
  lr=lr*STEPDECAY;
end
svs=find(alpha>0); nSVs=length(svs);
err=1-Y.*(K*(Y.*alpha)+b);
fprintf('%d,%d) nSVs=%d  |w|^2=%g  #,sErr=%d,%g  L=%g\n',...
        outerIt,innerIt,sum(abs(alpha)>0),(alpha.*Y)'*K*(alpha.*Y),...
        sum(err>1),sum(err(err>0)),...
        (alpha.*Y)'*K*(alpha.*Y)+pen*sum(err(err>0)));

% TESTCASE: 
% [X,Y]=mkMultiClassTst([-1 0;1 0; .2 .5],[400 400 100],[.3 .3; .3 .3; .2 .2]);
% Y=2*(Y==1)-1;
% [alpha,b,svs]=kerAdatron(X*X',Y,[],1,[],[],1,X); w=X'*(Y.*alpha);
% labScatPlot(X,Y);hold on;drawLine(w,b,min(X),max(X));
% drawLine(w,b-1,min(X),max(X),'r-');drawLine(w,b+1,min(X),max(X),'r-');
% drawDecisFn(@(x) (x*X'*(Y.*alpha)+b),min(X),max(X));
%
% TESTCASE: 
% K=kernel(X,X,'rbf',1);
% [alpha,b,svs]=kerAdatron(K,Y,[],1,[],[100 1e-5],1,X); 
% labScatPlot(X,Y,alpha);hold on;drawDecisFn(@(x) (kernel(x,X,'rbf',1)*(Y.*alpha)+b),min(X),max(X));
%
%
% TESTCASE 3:
% for i=1:12; [X,Y]=loadBench(i,1); fprintf('\nBenchmark Problem %d\n',i);[x,b,svs]=kerAdatron(kernel(X,[],'rbf',1),Y,[],1,[],[100 1e-5]); [tst.X,tst.Y]=loadBench(i,1,1);dv2conf(kernel(X,tst.X,'rbf',1)'*(Y.*x)+b,tst.Y,[],[],1); end
%
% 12) nSVs=202  |w|^2=43.8541  sErr=85.2665  L=129.121
% conf =    2540    201
%            333   1826
% Bin:		0.89102 0.89102 / 0.89102
% EEOC:		0.87817 0.87817 / 0.87817
% AUC: 		0.95411 0.95421 / 0.95416
% Benchmark Problem 2
% 5) nSVs=616  |w|^2=37.1938  sErr=405.033  L=442.227
% conf =    131    47
%            90   132
% Bin:		0.65750 0.65750 / 0.65750
% EEOC:		0.64736 0.64511 / 0.64623
% AUC: 		0.67211 0.67094 / 0.67153
% Benchmark Problem 3
% 7) nSVs=819  |w|^2=324.74  sErr=39.1522  L=363.892
% conf =    416    14
%            22   558
% Bin:		0.96436 0.96436 / 0.96436
% EEOC:		0.96532 0.96532 / 0.96532
% AUC: 		0.99160 0.99102 / 0.99131
% Benchmark Problem 4
% 11) nSVs=71  |w|^2=24.4396  sErr=6.70117  L=31.1408
% conf =    48    1
%            1   25
% Bin:		0.97333 0.97333 / 0.97333
% EEOC:		0.96036 0.96036 / 0.96036
% AUC: 		0.95683 0.97488 / 0.96586
% Benchmark Problem 5
% 3) nSVs=400  |w|^2=373.12  sErr=12.9963  L=386.117
% conf =       0   3504
%              0   3496
% Bin:		0.49943 0.49943 / 0.49943
% EEOC:		0.92129 0.91786 / 0.91957
% AUC: 		0.94757 0.94738 / 0.94748
% Benchmark Problem 6
% 6) nSVs=196  |w|^2=73.7564  sErr=41.4934  L=115.25
% conf =    52    2
%           22    1
% Bin:		0.68831 0.68831 / 0.68831
% EEOC:		0.56965 0.56965 / 0.56965
% AUC: 		0.58132 0.56280 / 0.57206
% Benchmark Problem 7
% 4) nSVs=700  |w|^2=317.539  sErr=116.015  L=433.553
% conf =    217     0
%            83     0
% Bin:		0.72333 0.72333 / 0.72333
% EEOC:		0.69963 0.69963 / 0.69963
% AUC: 		0.69719 0.69291 / 0.69505
% Benchmark Problem 8
% 3) nSVs=400  |w|^2=375.316  sErr=10.5983  L=385.915
% conf =     105   3438
%              0   3457
% Bin:		0.50886 0.50886 / 0.50886
% EEOC:		0.98057 0.98057 / 0.98057
% AUC: 		0.99786 0.99758 / 0.99772
% Benchmark Problem 9
% 3) nSVs=400  |w|^2=197.641  sErr=66.6405  L=264.282
% conf =    3085      0
%           1515      0
% Bin:		0.67065 0.67065 / 0.67065
% EEOC:		0.85070 0.85070 / 0.85070
% AUC: 		0.88548 0.88581 / 0.88564
% Benchmark Problem 10
% 5) nSVs=423  |w|^2=256.261  sErr=57.2706  L=313.531
% conf =    178    24
%            69    29
% Bin:		0.69000 0.69000 / 0.69000
% EEOC:		0.72363 0.72363 / 0.72363
% AUC: 		0.76813 0.76318 / 0.76566
% Benchmark Problem 11
% 5) nSVs=170  |w|^2=139.716  sErr=10.2894  L=150.005
% conf =    56    0
%           38    6
% Bin:		0.62000 0.62000 / 0.62000
% EEOC:		0.72971 0.72971 / 0.72971
% AUC: 		0.77557 0.78044 / 0.77800
% Benchmark Problem 12
% 5) nSVs=147  |w|^2=5.57958  sErr=58.0042  L=63.5838
% conf =    1228    155
%            315    353
% Bin:		0.77084 0.77084 / 0.77084
% EEOC:		0.68555 0.68260 / 0.68408
% AUC: 		0.74043 0.73679 / 0.73861

