function [perf,conf,dvs,binCls,eeoc]=mcSVMPerfComp(trnFeat,trnXcls,tstFeat,tstXcls,kerType,svmEng,svmParm,svmPen,bp,verb)
%function [perf,conf,dvs]=mcSVMPerfComp(trnFeat,trnXcls,tstFeat,tstXcls,kerType,svmEng,svmParm,svmPen,bp,verb)
% N.B. svmPen is a *row* vector of penalties (and optionally
%      coefficients) to try.
% defaults:
% kerType='lin';svmEng='libSVM';svmPen=0.5;svmParm=[1,0,svmPen,100e6,0.001];
% N.B. from Jocahins SVMlight a good starting C is mean(x'*x)^-1 
%      =mean(sum_d x_d*x_d)^-1 thus C should decrease inversely with
%       increasing dimensions. 
%
% $Id: mcSVMPerfComp.m,v 1.4 2006-11-16 18:23:47 jdrf Exp $
if ( nargin < 4 ) error('Insufficient arguments'); end;
if ( nargin < 5 | isempty(kerType) ) kerType='lin'; end;
if ( nargin < 6 | isempty(svmEng) ) svmEng='libSVM'; end;
if ( nargin < 7 | isempty(svmParm) ) svmParm=[0,0,100e6,0.001];end
if ( nargin < 8 | isempty(svmPen) ) svmPen=0.5; end;
if ( nargin < 9 | isempty(verb) ) verb=0; end
perf=[];conf=[];dvs={};binCls=[];eeoc=[];
for i=1:size(svmPen,2);  %loop over the list of penalties to try
  if ( size(svmPen,1) == 1 ) pen=svmPen(i); coeff=svmParm(1);
  else pen=svmPen(1,i); coeff=svmPen(2,i); end;
  fprintf('---------------------------------------------\n');
  fprintf('\nLearning with penalty: %g  (coeff=%g)\n\n',pen,coeff);
  fprintf('---------------------------------------------\n');
  svmParm(1)=coeff; svmParm(2)=pen; %[coeff, penalty, cache, termTol, SVM]
  disp(['Learning Classifier']);
  mcClsfier=mcSVMTrn(trnFeat,trnXcls,kerType,svmEng,svmParm,1);
%   disp('Training set performance');
%   trnconf=mcSVMTst(trnFeat,trnXcls,mcClsfier)
%   trnpconf=trnconf./ (repmat(sum(trnconf,2),[1,size(trnconf,1)])),sum(diag(trnpconf))/size(trnpconf,1), sum(diag(trnconf))/sum(sum(trnconf))
  disp('Test set performance');
  [conf(:,:,i),dv,binCls(i,:),eeoc(i,:)]=mcSVMTst(tstFeat,tstXcls,mcClsfier);
  dvs={dvs{:} dv};
  conf(:,:,i)
%  binCls(i,:)./size(tstXcls,1)
  pp=sum(diag(conf(:,:,i)))/sum(sum(conf(:,:,i)))
%  eeoc(i,:)
  perf=[perf;pp];
end
if(verb>0) svmPen,perf, end;
