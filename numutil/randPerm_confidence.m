function [pv,perf]=randPerm_confidence(Y,dv,delta,nSamp,func,verb)
% Compute confidence bound for binomial variates, e.g. classification performance
%
% [pv,perf]=randperm_confidence(Y,dv,delta,nSamp,func,verb)
% Y     - [Nx1] set of example labels
% dv    - [Nx1] set of predicted values for each example
% delta - [1x1] estimate the statistic value for this confidence
% nSamp - number random permutations to do (1000)
% func  - function to use for computing the objective value;
%         one of:
%          'bal' - balanced binary loss
%          'cr'  - classification rate
%          'auc' - AUC score
%         OR
%          function handle to function to use as func(Y,dv)
% verb  - [1x1] verbosity level (0)
% Outputs:
%  pv   - [1x1] probability of randomly getting a value *greater* than p with a
%              random permutation
%  perf - [1 x nSamp] function values for each of the random permutations
if ( nargin < 3 || isempty(delta) ) delta=.05; end;
if ( nargin < 4 || isempty(nSamp) ) nSamp=1000; end;
if ( nargin < 5 || isempty(func) )  func='cr'; end;
if ( nargin < 6 || isempty(verb) )  verb=1; end;

incIdx=find(all(Y~=0,2));
Ytrn=Y(incIdx,:); dv=dv(incIdx,:);
if ( verb>0 ) fprintf('RandPerm:'); end;
for pi=1:nSamp;
  perm=randperm(size(Ytrn,1));
  if ( ischar(func) )
    switch (func);
     case 'balcr';
      perf(:,pi)=conf2loss(dv2conf(Y(perm,:),dv),[],'bal');
     case 'cr';
      perf(:,pi)=conf2loss(dv2conf(Y(perm,:),dv),[],'cr');    
     case 'auc';
      if ( pi==1 ) % save the sorted decision values to speed up later permutations
        [perf(:,pi) sidx]=dv2auc(Y(perm,:),dv,[]);
      else
        [perf(:,pi)]=dv2auc(Y(perm,:),dv,[],sidx);
      end
     otherwise;
      perf(:,pi)=feval(func,Y(perm,:),dv);
    end;
  elseif ( isa(func,'handle') )
    perf(:,pi)=feval(func,Y(perm,:),dv);
  end
  if ( verb> 0 ) textprogressbar(pi,nSamp); end;
end
if ( verb>0 ) fprintf('\n'); end;
if ( isempty(delta) || (numel(delta)~=1 && numel(delta)==size(perf,1)) ) 
  % compute the p-value for the true/given value
  if ( isempty(delta) ) p=perf(:,1); else p=delta; if(p<.5)p=1-p;end; end;
  for i=1:size(perf,1); pv(i)=sum(perf(i,:)>p(min(i,end)))./size(perf,2); end;
else % compute the given confidence interval
  for i=1:size(perf,1);
	 sperf=sort(perf(i,:),2,'descend'); % sort in increasing order
	 pv(:,i)=sperf(:,max(1,floor(delta*size(perf,2)))); % pv is threshold
  end
end;

return;
% -----------------
function testCase()
Y =sign(randn(200,1));
dv=Y+randn(size(Y))*2;
p=conf2loss(dv2conf(Y,dv));

[pv,perf]=randPerm_confidence(Y,dv,.05,[],'cr');
pv,.5+binomial_confidence(size(Y,1))

[pv,perf]=randPerm_confidence(Y,dv,.05,[],'auc');
pv,.5+auc_confidence(size(Y,1))

Ns=[10 20 30 40 50 60 70 80 90];% 100 200 300 400 500 1000];
for ni=1:numel(Ns);
  Y =sign(randn(Ns(ni),1));
  dv=Y+randn(size(Y))*3;

  [cr_rp(ni),perf]=randPerm_confidence(Y,dv,.05,[],'cr');
  cr_np(ni)=.5+binomial_confidence(size(Y,1));

  [auc_rp(ni),perf]=randPerm_confidence(Y,dv,.05,[],'auc');
  auc_np(ni)=.5+auc_confidence(size(Y,1));
end
clf;subplot(121);plot([cr_rp; cr_np]'); subplot(122); plot([auc_rp; auc_np]');
