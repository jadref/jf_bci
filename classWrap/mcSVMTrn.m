function [mcClsfier]=mcSVMTrn(X,Y,KerType,SVMEng,params,bp,verb)
% [mcClsfier]=mcSVMTrn(X,Y,KerType,SVMEng,params,bp,verb)
%
% This file contains the code to test a given set of image feature
% descriptions using the one-against all evalutation strategy.
% We assume the input has the form
% X -- matrix of d dimesional feature vectors, used for training (Nxd)
% Y -- vector of *INTEGER* feature labels, used for training (Nx1)
% KerType -- id of the kernel to use, 
%              'lin','2poly','3poly', 'rbf', 'precomp'
% SVMEng -- svm training routine to use, 'libSVM', 'irwls', 'matSVM', 'svmLght'
%           'preCompSVM'
%           'linRSLC' -- use regularised linear least squares classifier,
%                        param(C)=regularisation degree.
%           'kerPLS'  -- user kernel PLS regression calssifier.
%                        param(C)= # PLS dimensions.
% params -- parameter array for the svm consists of: 
%           [Const Coef,Penalty C,CacheSize,epsilon]
% bp     -- use biased penalties? (0)
% Output:
%  mcClsfier -- structure containing the multi-class classifier learned
%               with all its parameters:
%      .SVMtype -- id of the kernel used for training.
%      .SVMEng  -- the engine used for training and hence ocClsfier structure.
%      .params  -- parameter set used for training
%    .ocClsfier{1:L} -- set of L one class classifiers learned from the
%                       data.  Internal structure of this slot is:
%                       .AlphaY, .svs, .bias - for 'libSVM' and 'irwls' engines,
%                       the trained network output structure for the rest.
%
%

% set the learning system to use
curdir=fileparts(which(mfilename));
defParms=[1,1,100e6,0.01];
if ( nargin < 2 ) help mcSVMTrn; return; end
if ( nargin < 3 | isempty(KerType) ) KerType=1; end % default to linear
if ( nargin < 4 | isempty(SVMEng) ) SVMEng='libSVM'; end
if ( nargin < 5 | isempty(params) ) params=defParms; else
if (length(params) < 4) params=[params defParms(length(params)+1:end)]; end
end;
if ( nargin < 6 | isempty(bp) ) bp=1; end;
if ( nargin < 7 ) verb=1; end;

% ensure Y is the right way round...
%if ( size(Y,1) < size(Y,2) ) Y=Y'; end;
% convert to indicator matrix if necessary, and get the labels list.
if ( sum(size(Y)>1) == 1 ) [Y,labels]=lab2ind(Y); L=size(Y,2); end;
if ( size(Y,2) == 2 ) % deal with binary
   Y=Y(:,2); L=1; 
   labels=[1;-1]; %else labels=[labels(2) labels(1)]; end;
else
   L=size(Y,2);labels=1:L;   
end; 


% Parameters=[KerType,degree,gamma,Const Coef,C,CacheSize,epsilon];
[N,dim]=size(X);par=[];ker=[];
switch KerType
 case {1,'lin'};          % linear
  disp('Linear Kernel'); 
  ker='linear'; svmlker=0; svmLParm=[0];           libSVMParms=[0 1 1]; 
 %if(nargin < 5) params=[0,1,1,1,1,100e6,0.01];else params=[0,1,1,params]; end
 case {2,'2poly'};          % polynomial degree 2
  disp('2nd Order Polynomial kernel'); 
  ker='poly_h'; svmlker=1; svmLParm=[2 params(1)]; libSVMParms=[1 2 1]; 
%  if(nargin < 5) params=[0,2,1,1,1,100e6,0.01];else params=[0,2,1,params]; end
 case {3,'3poly'};          % polynomial degree 3
  disp('3nd Order Polynomial kernel');
  ker='poly_h'; svmlker=1; svmLParm=[3 params(1)]; libSVMParms=[1 3 1];
%  if(nargin < 5) params=[1,3,1,1,1,100e6,0.01];else params=[0,3,1,params]; end
 case {4,'rbf'}'          % Gaussian, N.B. gamma==0 is set in SVMTrain
  disp('Gaussian kernel'); 
  ker='rbf';    svmlker=2; svmLParm=[params(1)]; libSVMParms=[2 0 params(1)];
 case {5,'precomp'};        % X is pre-computed kernel matrix.
  disp('Pre-computed kernel');
  ker='precomp';svmlker=5; svmLParm=[];  libSVMParms=[4 1 1];
%  if(nargin < 5) params=[4,1,0,0,1,100e6,0.01];else params=[4,1,0,params]; end
 otherwise % warn of error but assume linear...
  if ( isstr(KerType) ) error(['Unknown KerType :' KerType ]); 
  else error(['Unknown KerType' num2str(KerType) ]); end
end
switch SVMEng
 case 'libSVM';
  disp('osu_lib SVM');
 case 'irwls';
  disp('irwls SVM');
 case 'matSVM';
  disp('matlab SVM');
 case 'svmLght';
  disp('SVM light');
  defOpts=svmlopt('Verbosity',0,'ExecPath',[curdir '/svm/svml']);
 case 'linRLSC'; % recursive least squares classifier.
  disp('RLSC');
 case 'preCompSVM'; % pre-computed kernel svm.
  disp('Pre-computed kernel SVM');
 case 'kerPLS'; 
  disp('kerPLS');
 otherwise
  error('Unknow SVMEnging');return
end

confMx = zeros(L,L);
ocClsfier=cell(L);
if ( strcmp(SVMEng,'linRLSC') ) % train in a single pass..
  if( isfinite(params(1)) ) lambda=params(1); else lambda=[]; end;
  [alpha,svs,bias]=linRLSCTrain(X,Y,lambda,'class');
  for c=1:L;
    ocClsfier{c}.svs=svs(c,:);
    ocClsfier{c}.bias=bias(c,:);    
    ocClsfier{c}.alpha=alpha(c,:);
  end
% elseif( strcmp(SVMEng,'preCompSVM') ) % pre-comp kernel SVM single pass train
%   % Train the classifier on this fold
%   ocClsfier=svm_train(X,Y,params);
elseif ( strcmp(SVMEng,'kerPLS') )
  if( isfinite(params(1)) ) plsDim=params(1); else plsDim=20; end;
  mcClsfier.alpha=kerPLSTrain(X,Y,plsDim,'class',1);
else
  for c=1:L
    disp(['Training for class ' num2str(c)]);
    
    % run the learning algorithm on the sets
    % **********************************************
    pos=find(Y(:,c));neg=find(~Y(:,c));
    %pos=find(Y==labels(c)); neg=find(Y~=labels(c));         
    trnLabs=ones(N,1);trnLabs(neg)=-1;
    % record the first labels sign to debug libSVM bugs...
    ocClsfier{c}.firstLab=trnLabs(1);
    %   % generate the appropriate labels for the X training set.
    %   trnIdx=ones(1,N,'logical');
    %   trnLabs=double(Y==c) - double(Y~=c);
    
    switch SVMEng
     case 'libSVM'; % use libSVM    
      % generate a training set of all positives, then all negatives.
      % bodge to fix funny libSVM bug.
      % LIBSVM needs bodged training labels to stop silly bugs!
      trnIdx=[pos; neg];
      trnLabs=[ones(1,length(pos)) -ones(1,length(neg))]';
      if ( bp == 0 ) % use penalties scaled by the # data elements?
        penScl=[1 1];
      else 
        penScl=[length(pos)/length(trnIdx), length(neg)/length(trnIdx)];
      end
      [alphaY,svs,bias,libSVMParms,nSV]=SVMTrain(X(trnIdx,:)',trnLabs',...
                                                 [libSVMParms params],penScl);
      % record this classifiers info.
      ocClsfier{c}.alphaY=alphaY;
      ocClsfier{c}.svs=svs;
      ocClsfier{c}.bias=bias;     
     case 'irwls'; % user irwls
      [nSV,alphaY,bias]=irwls(X, trnLabs, KerType,params(2),params(1),params(3));
      % N.B. to test use: kernel(ker,TestSet,svs,4)*alpha(find(alpha))+bias
      % svsIdx=find(abs(alphaY)>1e-4);      
      ocClsfier{c}.alphaY=alphaY;
      ocClsfier{c}.svIdx=find(abs(alphaY)>1e-4);
      ocClsfier{c}.svs=X(ocClsfier{c}.svIdx,:); % extract the SVs
      ocClsfier{c}.bias=bias;
      fprintf('Total nSV = %d %d / %d \n',size(ocClsfier{c}.svs,1),nSV,length(trnLabs));
      
     case 'matSVM'; % use matsvm
      net = svm(dim,ker,params(1),params(2),0,'',250); % build svm
      net = svmtrain(net, X, trnLabs,[],verb);         % train it    
      ocClsfier{c}=net;      % record this classifiers info.
      fprintf('Total nSV = %d / %d \n', size(net.sv,1),length(trnLabs));

     case 'svmLght'; % use svm-light, ocClsfier{c}=net.
%       if ( strcmp(ker,'linear')) kernelParam=[params(2)]; 
%       else kernelParam=[params(2) params(4)];
%       end
      net = svml('',defOpts,'Kernel',svmlker,'KernelParam',svmLParm,'C',params(2),'EpsTermin',params(4),'CacheSize',params(3)); % build svm    
      net = svmltrain( net, X, trnLabs ) ;
      ocClsfier{c}=net;      % record this classifiers info.
     case 'preCompSVM'; % use pre-computed SVM training...
      ocClsfier{c}.modelfile=svm_train(X,trnLabs,[libSVMParms params]);
     otherwise;
      error('Unkonwn SVM engine');return;
    end
  end  
end

% generate the return structure.
mcClsfier.labels=labels;
mcClsfier.KerType=KerType;
mcClsfier.SVMEng=SVMEng;
mcClsfier.params=params;
mcClsfier.ocClsfier=ocClsfier;

