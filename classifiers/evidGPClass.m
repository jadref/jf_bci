function [z]=evidGPClass(z,trnIdx,tstIdx,varargin)
opts=struct('covFn','covLINone','likFn','logistic','maxEval',60,'alpha0',1,'approx','laplace');
opts=parseOpts(varargin,opts);
% the format of the output structure
loss=struct('func','class_loss','train',1,'train_se',1,...
            'test',1,'test_se',1);
% get the training set
if ( nargin < 2 | isempty(trnIdx) ) 
 trnIdx=false(ntrials(z),1);trnIdx(getouterfold(z))=true; tstIdx=~trnIdx;
end;
X=z.x; Y=z.y;


% train
% extract solution
switch lower(opts.approx);
  case 'laplace';
    nlogcost=minimize(log(opts.alpha0), 'binaryLaplaceGP',-opts.maxEval,...
                      opts.covFn,opts.likFn,X(trnIdx,:), Y(trnIdx));
    [dvp ans,ans,gpevid,best_a]=...
        binaryLaplaceGP(nlogcost, opts.covFn, opts.likFn,...
                        X(trnIdx,:),Y(trnIdx), X);
  case 'ep';
    nlogcost=minimize(log(opts.alpha0), 'binaryEPGP',-opts.maxEval,...
                      opts.covFn,X(trnIdx,:), Y(trnIdx));
    [dvp ans,ans,gpevid,best_a]=...
        binaryEPGP(nlogcost, opts.covFn,X(trnIdx,:),Y(trnIdx), X);
  otherwise;
    error('Unknown GP approximation type');
end
w=X(trnIdx,:)'*best_a;b=sum(best_a); alpha=exp(nlogcost);

% add prep info?
z=addprepinfo(z,mfilename(),...
              sprintf('Optimise %s %s GPc',opts.approx,opts.likFn),opts,...
              struct('w',w,'b',b,'alpha',alpha),'WTF?',[]);

% evaluate performance -- using the extracted hyper-plane info..
[conf,trnbin]=dv2conf(dvp(trnIdx)-.5,Y(trnIdx)); %N.B. dv2conf gives class rate
[conf,tstbin]=dv2conf(dvp(tstIdx)-.5,Y(tstIdx));

loss.train=1-trnbin(1);  %N.B. error rates!
loss.test =1-tstbin(1);  %N.B. error rates!

% evaluate performance -- using the extracted hyper-plane info.. for comparsion
dv=X*w+b; 
[conf,trnbin]=dv2conf(dv(trnIdx),Y(trnIdx)); % N.B. dv2conf reports class rate
[conf,tstbin]=dv2conf(dv(tstIdx),Y(tstIdx));

% add the results info to z
summary=sprintf('%s optimised train:%s, test:%s',mfilename,...
                lossstr('class loss',loss.train),...
                lossstr('class loss',loss.test));
z = setresults(z, mfilename, summary, [], loss);



return;

%-----------------------------------------------------------------------------
% testcases

evidGPClass(zc,[],[],'maxEval',100); % default logistic and laplace

evidGPClass(zc,[],[],'maxEval',100,'likFn','cumGauss') % cumGauss

evidGPClass(zc,[],[],'approx','EP','maxEval',100)      % EP
