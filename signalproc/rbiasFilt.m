function [x,s]=rbiasFilt(x,s,alpha,pr,scl)
% robust-bias removing filter (high-pass) based on running median estimation
%
%   [x,s,mu,std]=rbiasFilt(x,s,alpha,pr,scl)
%
% Inputs:
%   x - [nd x 1] the data to filter
%   s - [struct] internal state of the filter
%   alpha - [1 x 1] OR [nd x 1] exponiential decay factor for the median estimation for 
%                   all [1x1] or each [ndx1] input feature
%   pr  - [1 x 1] target precential to track, i.e. pr% of predictions should be < 0  (.5)
%   scl - [float] rescaling parameter for the median update.  N.B. max change in median/step = scl
% Outputs:
%   x - [nd x 1] filtered data,  x(t) = x(t) - fx(t)
%   s - [struct] updated filter state
if ( nargin<4 || isempty(pr) )  pr=.5; end;
if ( nargin<5 || isempty(scl) ) scl=1; end;
if ( nargin<6 || isempty(verb) ) verb=0; end;
if ( isempty(s) ) s=struct('sx',0,'sx2',0,'N',0,'x',0,'warmup',1); end;
if ( any(alpha>1) ) alpha=exp(log(.5)./alpha); end; % convert to decay factor
s.N = alpha.*s.N + (1-alpha).*1; % weight accumulated so far for each alpha, for warmup
if ( any(s.N(:,1) < .5) && s.warmup ) % still in warmup, use robust-mean estimator
   cx    = min(scl,max(x,-scl)); % bounded step size for x to robustify the mean estimation
   s.sx = alpha(:,1).*s.sx + (1-alpha(:,1)).*cx;
   s.sx2= alpha(:,1).*s.sx2+ (1-alpha(:,1)).*cx.^2; % est of 2nd cumulant, so can get to an pr est...
   if ( isempty(pr) || isequal(pr,.5) )
      b   = s.sx./s.N(:,1);
   else
      mu  =s.sx./s.N(:,1);
      std =sqrt(abs((s.sx2-s.sx.^2./s.N(:,1))./s.N(:,1)));
      b   =mu+std*sqrt(2)*-erfcinv(2*pr); % est percental based on the gaussian approx to the full dist
   end
   if ( all(s.N(:,1)>.5) ) % switch out of warmup mode
      s.warmup=0;
      s.sx    =b;
   end
else  % switch to median estimator
   step = x-s.sx;
   %step = min(scl,max(step,-scl)); % bounded step size
   step = sign(step);
   if ( ~isempty(pr) && ~isequal(pr,.5) ) % step up/down in proportion to target percental
      step(step(:)>0) = step(step(:)>0)*(pr./pr);
      step(step(:)<0) = step(step(:)<0)*((1-pr)./pr);
   end
   s.sx= s.sx + (1-alpha(:,1))*step; % udpate the estimator
   b   = s.sx;
end
if ( verb>0 ) fprintf('x=[%s]\ts=[%s]',sprintf('%g ',x),sprintf('%g ',b)); end;
x=x-b; % bias adapt 
if ( verb>0 ) fprintf(' => x_new=[%s]\n',sprintf('%g ',x)); end;
return;
function testCase()
x=cumsum(randn(2,1000),2);ox=x;

% simple test
s=[];fs=[];for i=1:size(x,2); [fx(:,i),si]=rbiasFilt(x(:,i),s,20,[],20); fs(:,i)=si.sx; s=si; end;
s=[];fs=[];for i=1:size(x,2); [fx(:,i),si]=biasFilt(x(:,i),s,20); fs(:,i)=si.sx./si.N; s=si; end;

x=randn(2,1000)+randn(2,1)*5;
s=[];fs=[]; fx=[];for i=1:size(x,2); [fx(:,i),si]=rbiasFilt(x(:,i),s,50,.9,.5); fs(:,i)=si.sx; s=si; textprogressbar(i,size(x,2)); end; % est .90 percental
sum(fx>0,2)./size(fx,2), % compute the actually percental

% add in outliers and see how it does
x=ox+ (rand(size(x))>.8)*100;

clf;for si=1:size(x,1); subplot(size(x,1),1,si); plot([x(si,:);fx(si,:);fs(si,:)]');legend('x','filt(x)','bias');end;set(gca,'ylim',[-5 5]);

