function [confMx,dv,binCls,eeoc,auc]=mcSVMTst(X,Y,mcClsfier,pon)
% [confMx,dv,binPerf,eeoc,auc]=mcSVMTst(X,Y,mcClsfier,pon)
%
% This file contains the code to test a given set of image feature
% descriptions using the one-against all evalutation strategy.
% Inputs: 
%   X -- matrix of d dimesional feature vectors (Nxd)
%   Y -- vector of *INTEGER* feature labels (Nx1)
%   pon  -- plot the svm's output?
% Outputs:
%   mcClsfier -- structure containing the classifier to test, format as
%              returned by mcSVMTrn.
% Outputs:
% confMx -- Returns the confusion matrix (where each row is a test class
%           and each column a predicted classification) and the learned
%           classifier's info. 
% dv      -- each classifiers output for each input.
% binPerf -- binary performance of the classifier. 
cols='rgbcymk';

if ( nargin < 3 ) help mcSVMTst; error('Insufficient arguments'); return; end;
if ( nargin < 5 ) pon=0; end;
dim=size(X,2);N=max(size(Y)); L=numel(mcClsfier.ocClsfier);
if( ~any(N==size(X)) ) error('X and Y have different number elements');end;
% ensure Y is the right way round...
%if ( size(Y,1) < size(Y,2) ) Y=Y'; end;
% convert to indicator matrix if necessary, and get the labels list.
if ( size(Y,2) < L  ) Y=lab2ind(Y,mcClsfier.labels);  end;
%if ( size(Y,2) == 2 ) Y=Y(:,1);end; % deal with binary

switch mcClsfier.SVMEng
 case 'libSVM';
  disp('osu_lib SVM');
 case 'irwls';
  disp('irwls SVM');
 case 'matSVM';
  disp('matlab SVM');
 case 'svmLght';
  disp('SVM light');
  defOpts=svmlopt('Verbosity',0,'ExecPath','./svm/svml');
 case 'id';  % identity classifier, simply copies given input col to output.
  disp('Simple MAX classifier');
 case 'linRLSC'; % recursive least squares classifier.
  disp('RLSC');
 case 'preCompSVM'; % pre-computed kernel svm.
  disp('Pre-computed kernel SVM');
 case 'kerPLS';
  disp('kerPLS');
 otherwise
  error('Unknow SVMEnging');return
end

switch mcClsfier.KerType
 case {1,'lin'};  
  irwlsker='lin';svmlker=0;svmLParm=[0];          
  libSVMParms=[0 1 1]; 
 case {2,'2poly'};
  irwlsker='poly_h';svmlker=1;svmLParm=[2 mcClsfier.params(1)];
  libSVMParms=[1 2 1]; 
 case {3,'3poly'};
  irwlsker='poly_h';svmlker=1;svmLParm=[3 mcClsfier.params(1)];
  libSVMParms=[1 3 1];
 case {4,'rbf'};  
  irwlsker='rbf';   svmlker=2;svmLParm=[mcClsfier.params(1)]; 
  libSVMParms=[2 0 mcClsfier.params(1)];
 case {5,'precomp'};
  irwlsker='precomp';svmlker=5;svmLParm=[];  libSVMParms=[4 1 1];
 otherwise 
  if ( isstr(KerType) ) error(['Unknown KerType' KerType ]); 
  else error(['Unknown KerType' num2str(KerType) ]); end
end

%L=length(mcClsfier.labels);
confMx = zeros(L,L);
dv=zeros(N,L);
  
% loop over the testing examples for each class
if ( strcmp(mcClsfier.SVMEng,'kerPLS') )  
  dv=X'*mcClsfier.alpha; confMx=dv2conf(dv,Y);
  return;
% elseif ( strcmp(mcClsfier.SVMEng,'preCompSVM') ) 
%   % pre-comp kernel SVM test in single pass.
%   dv=svm_predict(X,Y,mcClsfier.ocClsfier,'-v');
%   [confMx,binCls]=dv2conf(dv,Y);
%   %delete(mcClsfier.ocClsfier);  % remove the modelfile
%   return;
end  
dv=zeros(N,L);
% loop over the classifiers finding the one with the highest quality
for clsfier=1:L;
  pos=find(Y(:,clsfier));neg=find(~Y(:,clsfier));
% pos=find(Y==mcClsfier.labels(clsfier));  % get the test pts in this class
% neg=find(Y~=mcClsfier.labels(clsfier));

  labs=ones(N,1);labs(neg)=-1;
  fprintf('Testing classifier %d\n',clsfier);
  % test this classifier on this input set
  %if ( c == clsfier ) lab=1; else lab=-1; end %get the right label
  switch mcClsfier.SVMEng
   case 'libSVM'; % use libSVM
    [classRate,clsDv]=SVMTest(X',labs', ...
                              mcClsfier.ocClsfier{clsfier}.alphaY,...
                              mcClsfier.ocClsfier{clsfier}.svs,...
                              mcClsfier.ocClsfier{clsfier}.bias,...
                              [libSVMParms mcClsfier.params]);
    % Alternative BODGE bodge to fix funny libSVM bug.
    %      if ( Ytrn(1) ~= clsfier ) clsDv=-clsDv; end; % 
    clsDv=clsDv'; % just to be consistent...    
   case 'irwls'; % use irlws: f(x)=\sum_i \alpha_i*K(x_i,x) + c
    clsDv=kernel(X,mcClsfier.ocClsfier{clsfier}.svs,irwlsker,mcClsfier.params(1))* ...
          mcClsfier.ocClsfier{clsfier}.alphaY(abs(mcClsfier.ocClsfier{clsfier}.alphaY)>1e-4) + ...
          mcClsfier.ocClsfier{clsfier}.bias;
   case 'matSVM'; % use matSVM, ocClsfier{c}=net
    [classRate, clsDv] = svmfwd(mcClsfier.ocClsfier{clsfier}, X);
   case 'svmLght'; % use svm-light, ocClsfier{c}=net.
    clsDv = svmlfwd(mcClsfier.ocClsfier{clsfier}, X);
   case 'id'; % use identity classifier.
              % output is simply copy of appropriate input column
    clsDv = X(:,clsfier); 
   case 'linRLSC' ; % use linear regularised least squares classifier.
    clsDv=mcClsfier.ocClsfier{clsfier}.alpha.*...
          X*mcClsfier.ocClsfier{clsfier}.svs' + ... 
          mcClsfier.ocClsfier{clsfier}.bias;
    case 'preCompSVM' ;
     % pre-comp kernel SVM test in single pass.
     clsDv=svm_predict(X,labs,mcClsfier.ocClsfier{clsfier}.modelfile,'-v');
     if ( mcClsfier.ocClsfier{clsfier}.firstLab < 0 ) clsDv=-clsDv; end;
   otherwise; 
    error('Unkonwn SVM engine');return;
  end
  % record this classifiers output, to use later for multi-class
  dv(:,clsfier)=clsDv;
  if ( pon )   hold on; plot(pos,clsDv,[cols(clsfier) '.']); end;
end % for c=1:L
if( pon ) dv,  end;
[confMx,binCls,eeoc,auc]=dv2conf(dv,Y,mcClsfier.labels,[],1);
% fprintf('\nconfMx = ');for i=1:L;fprintf('%1.5f ',confMx(i,:));fprintf('\n');end;
% fprintf('\nBin:\t\t');fprintf('%1.5f ',binCls./sum(sum(confMx)));fprintf('/ %1.5f\n',mean(binCls./sum(sum(confMx))));
% fprintf('EEOC:\t\t');fprintf('%1.5f ',eeoc);fprintf('/ %1.5f\n',mean(eeoc));
