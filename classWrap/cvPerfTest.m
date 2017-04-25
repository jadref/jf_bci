function [perf,conf,mcClsfier]=cvmcSVMTest(nFold,X,Y,L,SVMType,SVMEng,params,bp,verb)
% [mcClsfier]=cvMCSVMTest(nFold,X,Y,L,SVMType,SVMEng,params,bp)
%
% Cross validated multi-class SVM train.  As for mcSVMTrn except using
% nFold-cross validation to estimate the system performance.
%
% This file contains the code to test a given set of image feature
% descriptions using the one-against all evalutation strategy.
% We assume the input has the form
% X -- matrix of d dimesional feature vectors, used for training (Nxd)
% Y -- vector of *INTEGER* feature labels, used for training (Nx1)
% L -- the number of distinct label types.
% SVMType -- id of the kernel to use, 1=linear,2=2 poly, 3=3 poly, 4=rbf
% SVMEng -- svm training routine to use, 'libSVM', 'irwls', 'matSVM', 'svmLght'
%           'linRSLC' -- use regularised linear least squares classifier,
%                        param(C)=regularisation degree.
% params -- parameter array for the svm consists of: 
%           [gamma,Const Coef,C,CacheSize,epsilon]
% bp     -- use biased penalties? (0)
% verb   -- verbosity level.      (0)
% Output:
%  perf -- the estimated classifier performance
%  conf -- the estimated classifier confusion matrix
%  mcClsfier -- structure containing the multi-class classifier learned
%               with all its parameters:
%      .SVMtype -- id of the kernel used for training.
%      .SVMEng  -- the engine used for training and hence ocClsfier structure.
%      .params  -- parameter set used for training
%    .ocClsfier{1:L} -- set of L one class classifiers learned from the
%                       data.  Internal structure of this slot is:
%                       .AlphaY, .svs, .bias - for 'libSVM' and 'irwls' engines,
%                       the trained network output structure for the rest.

if ( nargin < 4 ) help cvMCSVMTest; return; end;
if ( nargin < 5 ) SVMType=[]; end;
if ( nargin < 6 ) SVMEng=[]; end;
if ( nargin < 7 ) params=[]; end;
if ( nargin < 8 ) bp=[]; end;
if ( nargin < 9 ) verb=0; end;

% first compute a set of fold index lists.
foldIdxs=gennFold(Y,nFold);

% Now loop over training with nFold-1 folds and testing with the other
% one to estimate the classifiers performance.
conf=zeros(L,L);
for fold=1:nFold;
  fprintf('Fold %d\n',fold);
  % compute indexs of train
  trnIdxs=logical(sum(foldIdxs([1:fold-1 fold+1:nFold],:))); 

  % Train the classifier on this fold
  mcClsfier=mcSVMTrn(X(trnIdxs,:),Y(trnIdxs,:),SVMType,SVMEng,params,bp);

  % Test it one the validation set.
  tstconf=mcSVMTst(X(~trnIdxs,:),Y(~trnIdxs),mcClsfier);
  if ( verb > 0 )
    tstconf
    tstpconf=tstconf./ (repmat(sum(tstconf,2),[1,size(tstconf,1)]))
    sum(diag(tstpconf))/size(tstpconf,1), sum(diag(tstconf))/ sum(sum(tstconf))
  end

  % record it performance
  conf=conf+tstconf;
end;
% compute the summary performance value.
perf=sum(diag(conf))/sum(sum(conf));
if ( verb > 0 )
  pconf=conf./ (repmat(sum(conf,2),[1,size(conf,1)])) 
  pperf=sum(diag(pconf))/size(pconf,1)
  perf
end
