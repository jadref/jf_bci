function [w,b,wght,sWght,muX,covXX,riter]=rrwls(X,Y,wght,pen,lambda,varargin);
% varargin=(w,b,pen,lambda,reweightFn,rwghtThresh,termParms,cutSize,verb)
% function [w,b,wght,sWght,muX,covXX,riter]=rrwls(X,Y,wght,pen,lambda,...)
% 
% Solve the regularised linear regression problem: 
%   min_{w,b} w'*lambda*w + pen*L(Y,X*w+b)
% using the iterative least squares technique for a variety of loss functions L
% technique.
% Inputs:
%  X     -- inputs          [Nxd]
%  Y     -- desired outputs [NxL]
%  wght  -- initial weight  [Nx1]
%  w     -- a starting solution [dx1]
%  b     -- a starting bias.    [1x1]
%  pen   --  weight for the loss term, [1x1] or [Nx1] 
%            if Nx1 then includes point specific importance = C*wght_i  (1)
%  lambda -- regularisation weight for LS solution [1x1] or [dxd]..     (1)
%            if [dxd] must be positive semi-definite
%  reweightFn,rwghtParmas -- reweighting function and it's parameters,
%      used to implement different loss functions. 
%         'none',  []                        Loss = r^2
%         'huber', [Thresh],          Loss = r < Thresh ? r^2 : pen*r
%                       N.B. Grad ~= Thresh gives discontinuous loss function.
%         'etainsL1', [Thresh], Loss = r < Thresh ? 0   : pen*r
%                       N.B. Has discontinouity about Thresh.
%         'etainsL2', [Thresh], Loss = r < Thresh ? 0   : pen*(abs(r)-Thresh)^2
%         'hinge', [] only for +/-1 classification problems then,
%                                     Loss = Y*r < 0    ? 0   : pen*r 
%         'logistic'  only for 0/1 classification problems (currently)
%  [maxreweightIter,tol,tol2]        ([40,5e-3])
%                maximum number of reweighting iterations 
%                termination threshold, min percentage change in [w b] 
%                termination threshold, loss must get at least this much 
%                   better per iteration. (should be proportional to Grad) (0)
%  cutSize  -- (5e3)
%  verb     -- verbosity level. (0)
%
% Outputs:
%  alpha -- weights for the learned linear hyperplane
%  b     -- bias
%  w     -- final weighting
%  [sWght,muX,covXX,muY,covYY,covYX] -- statistics for the weighted data
%                                       as for dataStats
%
% TODO: Extend logistic to the multi-class case
%
% $Id: rrwls.m,v 1.11 2007-08-23 09:09:27 jdrf Exp $
NU=.5;        % adaptive step size step size..
ZEROTOL=1e-3; % maximum error contribution to allow point to be given 0 weight
MARGINTOL=1e-3; % residual for point to be considered *on* the margin
if ( nargin < 5 ) help rrwls; return; end;
%w,b,pen,lambda,reweightFn,rwghtThresh,termParms,cutSize,verb
opt=struct('w',[],'b',[],'reweightFn','hinge','rwghtThresh',[],'maxIter',...
           40,'tol',5e-3,'tol2',-1,'probType','class','cutSize',1e5,'verb',0,'plot',0);
[opt,varargin]=parseOpts(opt,varargin);
if(~isempty(varargin))error('Unrecognised Option(s)'); end;

% def: Thresh = Grad = pen
if( isempty(opt.rwghtThresh) )  opt.rwghtThresh=pen; end
if( isempty(wght) ) wght=ones(size(X,1),1); end
% hinge/logistic are classification only special case's 
if ( any(strcmp(opt.reweightFn,{'hinge','logistic'})) ) 
   opt.probType='class';
  if ( size(Y,2) > 1 )
    error('Hinge/logistic weighting only works for 2 class problems!');
  else    
    labels=unique(Y);
    if ( length(labels) ~= 2 ) 
      error('Hinge/logistic weighting only works for 2 class problems!');
    end    
    if ( strcmp(opt.reweightFn,'hinge') && any(labels'~=[-1 1]) )
      %convert to +1/-1 format
      fprintf('Converting Y to +1/-1 format\n');
      Y=single(Y);Y(Y==labels(1))=-1; Y(Y==labels(2))=1; labels=[-1;1];
    elseif ( strcmp(opt.reweightFn,'logistic') && any(labels'~=[0 1]) )
      fprintf('Converting Y to 0/1 format\n');
      Y=single(Y);Y(Y==labels(1))=0; Y(Y==labels(2))=1;labels=[0;1];
    end
  end
  %if ( all(labels'==[0 1]) ) dt=.5; else dt=0; end;
elseif ( ~isempty(strfind(opt.reweightFn,'eta')) )
   opt.probType='regress';
end


% turn the penalties into a vector to deal with weighted cases.
if ( isscalar(pen) ) pen=pen(ones(size(Y))); end;

% Set the initial target values, 0/1 for logistic, -1/+1 for SVM
if ( strcmp(opt.reweightFn,'logistic') ) 
  T=(Y>0)-.5; % N.B. T=X*w+b + W^-1(Y-u(X)) = Y-.5 for w=b=0
else T=Y; end; 

% Loop until convergence.
[N,dim]=size(X); ow=inf(dim+1,1);minR=inf; binCls=[]; labels=[];
for riter=1:opt.maxIter

  % Compute a seperating hyperplane using the least-squares technique. 
  [sWght,muX,covXX,muT,covTT,covTX]=...
      multiCutDataStats(X,T,wght,0,opt.cutSize,opt.verb);
%     if(rcond(covXX) < 1e-7)
%       fprintf('%d) Warning: covXX badly conditioned, rcond: %g\n',...
%               riter,rcond(covXX));
%       covXX=covXX+diag(eps(ones(size(covXX,1),1)).^(2/3)); 
%     end;

  if ( isempty(lambda) | lambda==0 ) 

    % Find un-regularised least squares solution
    if(rcond(covXX) < 1e-7) % unregularised least squares solution
      fprintf('%d) Warning: covXX badly conditioned, rcond: %g\n',...
              riter,rcond(covXX));
      covXX=covXX+diag(eps(ones(size(covXX,1),1)).^(2/3)); 
    end;
    w=covXX\covTX';
    b=muT-muX*w;
  
  else % regularised solution
    % N.B. covXX=X'*diag(w)*X/sWght, covYX=Y'*diag(w)*X so need to divide
    % regulariser buy this also to compensate.
%     invMx=[covXX+eye(dim,dim)*lambda/sWght muX'; muX 1];
%     if ( rcond(invMx) < 1e-7)
% %       fprintf('%d) Warning: SOLN mx badly conditioned, rcond=%g\n',...
% %               riter,rcond(invMx));
%       invMx=invMx+diag(eps(ones(size(invMx,1),1)).^(2/3));
%     end    
%     w=(invMx)\([covYX'; muY]);b=w(end); w=w(1:end-1);
    % Treat w and b as independent..
    %   w=(covXX+eye(dim,dim)*lambda/sWght)\covYX';b=muY-muX*w;
    %N.B. with FVal seln this performs *much* better.

    % Scalar regularised
    if ( isscalar(lambda) ) 
%       w2=conjGrad([covXX+eye(dim,dim)*lambda/sWght muX'; muX 1],...
%                       [covYX'; muY],w2,1e-10,0);
      w=([covXX+eye(dim,dim)*lambda/sWght muX'; muX 1])\([covTX'; muT]);
      %fprintf('t_cg=%g t_ls=%g\n',cgrt,lsrt);
    
      
    else % Matrix regularised solution

      % Avoid singularity problems
      if ( rcond(covXX+lambda/sWght) < 1e-18 )
        fprintf('---%d covXX+lambda/sWght badly conditioned, correcting!---\n',riter);
%       if ( rcond(lambda) <1e-6) 
%         fprintf('----lambda badly conditioned, correcting!----\n');
        for sc=.1.^[8 6 4 2 1 0]; % estimate the correction factor we need
          if ( rcond(covXX+(lambda+eye(dim,dim)*sc)/sWght) > 1e-12 ) 
%           if ( rcond(lambda+eye(dim,dim)*sc) > 1e-6 )
            break; 
          end;
        end
        lambda=lambda+eye(dim,dim)*sc;
      end;
      %       lambda=lambda+eye(dim,dim)/10000; 

      % Compute the solution
      w=([covXX+lambda/sWght muX'; muX 1])\([covTX'; muT]);
    
    end

    b=w(end); w=w(1:end-1); % extract the w,b parts from the solution
  end

  % Check for and correction minimal overshoot:
  % Adaptive step sizer, simply halve the step size each time if we've
  % over-shot the minima.
  % N.B. assumes this is a convex function with no local minima
  for i=1:6; % compute a good step size, down to 3% orig step

    switch (opt.reweightFn) ; % compute the true loss
     
     case 'huber'; % iteratively reweight LS solution with Huber fn.
      f=X*w+b;  r=T-f;% prediction and residual
      pts=(abs(r) > opt.rwghtThresh); 
      R(riter)=w'*lambda*w +sum(pen(pts).*r(pts))+sum(pen(~pts).*(r(~pts).^2));
     
     case 'etainsL1';  % reweight with eta insensitive
      f=X*w+b;  r=T-f;% prediction and residual
      pts=(abs(r) > opt.rwghtThresh);
      % zero out cost
      ppts=find(r>opt.rwghtThresh);  r(ppts)=r(ppts)-opt.rwghtThresh; 
      npts=find(r<-opt.rwghtThresh); r(npts)=r(npts)+opt.rwghtThresh;      
      R(riter)=w'*lambda*w + sum(pen(pts).*r(pts));

     case 'etainsL2';  % reweight with eta insensitive
      f=X*w+b;  r=T-f;% prediction and residual
      pts=(abs(r) > opt.rwghtThresh);
      % zero out cost
      ppts=find(r>opt.rwghtThresh);  r(ppts)=r(ppts)-opt.rwghtThresh; 
      npts=find(r<-opt.rwghtThresh); r(npts)=r(npts)+opt.rwghtThresh;      
      R(riter)=w'*lambda*w + sum(pen(pts).*r(pts).*r(pts));
      
     case 'hinge';  % reweight with hinge loss, like svm   
      f=X*w+b;  r=T-f;% prediction and residual
      err=Y.*r; 
      pts=(err > -MARGINTOL);
      R(riter)=w'*lambda*w + sum(pen(pts).*abs(err(pts)));

     case 'logistic';
      f=1./(1+exp(-X*w-b)); % Pr(Y=1|x,w,b) 
      r=T-f;% prediction and residual
      pts=Y>0;
      R(riter)=w'*lambda*w -sum(pen(pts).*f(pts))- sum(pen(~pts).*(1-f(~pts)));
     
     case 'none';
      f=X*w+b;  r=T-f;% prediction and residual      
     
     otherwise; warning(['Unknown reweightFn :' reweightFn]);
    end

    if ( opt.verb>1 )
      if ( i==1 & riter>1 ) fprintf('%d) Adapt R(i-1)=%g  ',riter,R(end-1));
      else                  fprintf(',%g ',R(end));
      end
    end
    % Check that new value improves the loss function.
    if ( riter ==1 | R(end-1) > R(end) ) 
      break;  % found a good function reducing value.
    else % approx broke down and overshot minima, make smaller step.
      w=NU*w+(1-NU)*ow(1:end-1); b=NU*b+(1-NU)*ow(end);
    end
  
  end

  % use the old value and abort because we've converged!
  if ( minR < R(end) + opt.tol2 ) 
    % prev was better, so use it
    if ( minR < R(end) ) w=minW(1:end-1); b=minW(end); end;
    break; % bail out now.
  end

  % update the lower bound loss and associated hyperplane
  if ( R(end) <= minR ) minR=R(end); minW=[w;b]; end;
  
  % compute some other stats
  SSE=sWght*(covTT-covTX*w);
  SAE=sum(abs(r));
  if ( opt.verb ) 
     if ( strcmp(opt.probType,'class') )
        [nErr,binCls(riter,:),eeoc(riter,:)]=dv2conf(f,Y,[],'nClass',opt.verb);
        fprintf('%d) SSE %g  SAE %g PP %g \n',riter,SSE,SAE,sum(diag(nErr))./N);
     else
        fprintf('%d) SSE %g  SAE %g\n',riter,SSE,SAE);
     end
 end

  % Update the weights
  switch (opt.reweightFn) ;
   
   case 'huber'; % iteratively reweight LS solution with Huber fn.
    wght(:)=1;
    wght(pts)=pen(pts)./abs(r(pts)); % loss prop to residual
   
   case 'etainsL1';  % reweight with eta insensitive
    wght(:)=0;  % points near hyperplane ignored!
    wght(pts)=pen(pts)./abs(r(pts));
   
   case 'etainsL2';  % reweight with eta insensitive
    wght(:)=0;  % points near hyperplane ignored!
    wght(pts)=pen(pts); % everything else is quadratic so OK
    
   case 'hinge';  % reweight with hinge loss, like svm   
    wght(:)=0;
    wght(pts)=pen(pts)./abs(r(pts)); % loss is prop to residual
    % points on/near margin have W=C, needed to stabilise convergence
    wght(abs(r)<MARGINTOL)=pen(abs(r)<MARGINTOL)/MARGINTOL;   
   
   case 'logistic';
    wght=f.*(1-f); wght(wght==0)=eps; % to stop division by 0
    T=X*w+b+(Y-f)./wght;
   
   case 'none'; 
   otherwise; warning(['Unknown reweightFn :' reweightFn]);
  end
  
  if ( opt.verb ) 
    fprintf('%d) |wght|_1 = %d  |w|^2 = %7.3f  w*r^2 = %7.3f  L=%7.3f\n',...
            riter,sum(abs(wght)>0),w'*w,wght'*(r.^2),R(riter));
  end

  if ( opt.plot ) plotSoln(X,Y,wght,w,b,f,labels,riter,binCls,R,ow); end  

  % early termination test, no change in hyperplane direction for 2 iter!
  dw(riter)=1-ow'*[w;b]./(norm(ow)*norm([w;b]));
  if ( opt.verb>0 ) fprintf('%d) d W : %g\n',riter,dw(riter)); end
  if( mean(dw(max(riter-1,1):riter)) < opt.tol ) break; end;
  ow=[w;b];
  
end


function []=plotSoln(X,Y,wght,w,b,f,labels,riter,binCls,R,ow)
if ( nargin > 8 && ~isempty(binCls) && ~isempty(R) )
  figure(1);hold off; plot(1-binCls(:,1)); hold on; plot(R/R(1),'r');
end
figure(2);hold off; [A,B]=sort(Y); 
if ( ~isempty(f) ) hold on; plot(f(B),'b'); end; plot(Y(B),'r.'); 
if ( nargin < 8 || isempty(riter) ) riter=1; end;
figure(floor((riter-1)/9)+3);subplot(3,3,mod(riter-1,9)+1); 
if ( size(X,2) > 1 )
  hold off; labScatPlot(X',Y,wght); 
  hold on;
  if ( nargin > 10 && ~isempty(ow) ) 
    drawLine(ow(1:end-1),ow(end),min(X),max(X),'y'); 
  end
  drawLine(w,b,min(X),max(X));
  drawLine(w,b+1,min(X),max(X),'r-');
  drawLine(w,b-1,min(X),max(X),'r-');
else
  hold off; plot(X,Y(:,1),'g.');
  pts=find(wght>0); hold on; plot(X(pts),Y(pts,1),'b.');
  hold on; drawLine([w -1],b,min([X Y]),max([X Y]));
end 
pause(0.05);


%TESTCASE 1) hinge vs. logistic (unregularised)
% [X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 20],[.3 .3; .3 .3; .2 .2]);
% Y=2*(Y==1)-1;
% [alpha,b,w]=rrwls(X,Y==1,[],1,0,'reweightFn','hinge','maxIter',40,'tol',1e-6,'verb',2,'plot',1);
% alpha=  [-25.8507; -0.5067]
% b=      [-6.6015]
% [alpha,b,w]=rrwls(X,Y==1,[],1,0,'reweightFn','logistic','maxIter',40,'tol',1e-6,'verb',2,'plot',1);
% alpha = [-145.0268; -3.2759]
% b=      [-36.7221]
%
% TESTCASE 2) hinge vs. logistic (regularised)
% [alpha,b,w]=rrwls(X,Y==1,[],1,1,'reweightFn','hinge','maxIter',40,'tol',1e-6,'verb',2,'plot',1);
% alpha=  [-1.7112; -0.3280]
% b=      [-0.2482]
% [alpha,b,w]=rrwls(X,Y==1,[],1,1,'reweightFn','logistic','maxIter',40,'tol',1e-6,'verb',2,'plot',1);
% alpha = [-5.2517   -0.8201]
% b=      [-.3479]
% 
% TESTCASE 3) real data
% X=importdata('~/temp/diabetes.sdata.txt');Y=2*(X(:,end)>0)-1;X=X(:,1:end-1);
% [alpha,b,w]=rrwls(X,Y,[],1,1,'reweightFn','hinge','maxIter',40,'tol',1e-6,'verb',2,'plot',0);
% err=Y.*(Y-X*w-b);L=w'*w+sum(err(err>0))
% conf =    206    41
%            65   130
% Bin:		0.76018 0.76018 / 0.76018
% EEOC:		0.73765 0.73765 / 0.73765
% AUC: 		0.82807 0.82915 / 0.82861
% 17) SSE 397.401  SAE 303.07 PP 0.760181 
% 17) |wght|_1 = 415  |w|^2 =  52.296  w*r^2 = 300.007  L=352.259
%
%
% [alpha,b,w]=rrwls(X,Y,[],1,1,'reweightFn','logistic','maxIter',40,'tol',1e-6,'verb',2,'plot',0);
% conf =    212    35
%            73   122

% Bin:		0.75566 0.75566 / 0.75566
% EEOC:		0.74426 0.74426 / 0.74426
% AUC: 		0.82228 0.82336 / 0.82282
% 26) SSE 406.108  SAE 911.339 PP 0.755656 
% 26) |wght|_1 = 442  |w|^2 =  15.391  w*r^2 = 479.297  L=-226.553
%
%
% TESTCASE: non-linear kernel
% K=kernel(X,X,'rbf',1);
% [alpha,b,w]=rrwls(K,Y,[],1,K,'reweightFn','hinge','maxIter',40,'tol',1e-6,'verb',2,'plot',0);
% clf;labScatPlot(X',Y,abs(alpha));hold on; drawDecisFn(@(x) (kernel(x,X,'rbf',1)*alpha+b),min(X),max(X));
%
% TESTCASE: linear kernel, etainsensitive regression
% X=[1:1000]'; Y=sum(X,2)+(randn(size(X,1),1))*max(X(:))/5;  
% outliers=floor(rand(100,1)*numel(Y)/2)+numel(Y)/2;Y(outliers)=Y(outliers)+mean(Y);
% w=X\Y; b=mean(Y);
% [alpha,b,w]=rrwls(X,Y,[],1,1,'reweightFn','etainsL2','rwghtThresh',200,'maxIter',40,'tol',1e-6,'probType','regress','verb',2,'plot',0);
