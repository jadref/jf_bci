function [confMx, classifier, params]=mcSVMTest(Xtrn,Ytrn,Xtst,Ytst,L,SVMType,SVMEng,params)
% [confMatx, classifier, params]=mcSVMTest(Xtrn,Ytrn,Xtst,Ytst,L,SVMType,SVMEng,params)
%
% This file contains the code to test a given set of image feature
% descriptions using the one-against all evalutation strategy.
% We assume the input has the form
% Xtrn -- matrix of d dimesional feature vectors, used for training (Nxd)
% Ytrn -- vector of *INTEGER* feature labels, used for training (Nx1)
% Xtst -- matrix of d dimesional feature vectors (Nxd)
% Ytst -- vector of *INTEGER* feature labels (Nx1)
% L -- the number of distinct label types.
% SVMType -- id of the kernel to use, 1=linear,2=2 poly, 3=3 poly, 4=rbf
% SVMEng -- svm training routine to use, 'libSVM', 'irwls', 'matSVM', 'svmLght'
% params -- parameter array for the svm consists of: 
%           [gamma,Const Coef,C,CacheSize,epsilon]
%
% Returns the confusion matrix (where each row is a test class and each
% column a predicted classification) and the learned classifier's info.

[N,dim]=size(Xtrn);
% set the learning system to use
if ( nargin < 5 ) help mcSVMTest; return; end
if ( nargin < 6 ) SVMType=0; end % default to linear
if ( nargin < 7 ) SVMEng='libSVM'; end

% Parameters=[SVMType,degree,gamma,Const Coef,C,CacheSize,epsilon];
switch SVMType
 case 1          % linear
  disp('Linear Kernel'); ker='linear'; svmlker=0;
  if( nargin < 8 ) params=[0,1,1,0,1,100e6,0.01]; else params=[0,1,params]; end
 case 2          % polynomial degree 2
  disp('2nd Order Polynomial kernel'); ker='poly_h'; svmlker=1;
  if( nargin < 8 ) params=[0,2,1,0,1,100e6,0.01]; else params=[0,2,params]; end
 case 3          % polynomial degree 3
  disp('3nd Order Polynomial kernel'); ker='poly_h'; svmlker=1;
  if( nargin < 8 ) params=[1,3,1,0,1,100e6,0.01]; else params=[0,3,params]; end
 case 4          % Gaussian, N.B. gamma==0 is set in SVMTrain
  disp('Gaussian kernel'); ker='rbf'; svmlker=3;
  if( nargin < 8 ) params=[2,1,1/dim,0,1,100e6,0.01]; 
  else params=[2,1,params]; end
 otherwise 
  error('Unknown SVMType');return
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
  defOpts=svmlopt('Verbosity',0,'ExecPath','./svm/svml');
 otherwise
  error('Unknow SVMEnging');return
end

% ensure Y is the right way round...
if ( size(Ytrn,1) < size(Ytrn,2) ) Ytrn=Ytrn'; end;

confMx = zeros(L,L);
classifiers=cell(L);
for c=1:L
  disp(['Training for class ' num2str(c)]);
  
  % run the learning algorithm on the sets
  % **********************************************

  % generate a training set of all positives, then all negatives.
  % bodge to fix funny libSVM bug.
  pos=find(Ytrn==c); neg=find(Ytrn~=c);         
  trnIdx=[pos; neg];
  trnLabs=[ones(1,length(pos)) -ones(1,length(neg))]';
  
%   % generate the appropriate labels for the Xtrn training set.
%   trnIdx=ones(1,N,'logical');
%   trnLabs=double(Ytrn==c) - double(Ytrn~=c);
  
  % get the gamma or degree depending on kernel type
  if ( SVMType == 3 ) par=params(3); else par=params(2); end
  switch SVMEng
    
   case 'libSVM'; % use libSVM
    [alphaY,svs,bias,params,nSV]=SVMTrain(Xtrn(trnIdx,:)',trnLabs',...
                                          params,[1 1]);
    classifier{c}= {alphaY, svs, bias}; % record this classifiers info.
    
   case 'irwls'; % user irwls
    [nSV,alphaY,bias]=irwls(Xtrn(trnIdx), trnLabs, ker,params(5),par, ...
                            params(7));
    % N.B. to test use: kernel(ker,TestSet,svs,4)*alpha(find(alpha))+bias
    svsIdx=find(abs(alphaY)>1e-4);
    svs = X(trnIdx(svsIdx),:);  % extract the SVs      
    classifier{c}= {alphaY(svsIdx), svs, bias}; % record classifiers info
    fprintf('Total nSV = %d %d / %d \n', size(svs,1),nSV,length(trnLabs));
    
   case 'matSVM'; % use matsvm
    net = svm( dim, ker, par, params(5),0,'',250); % build svm
    net = svmtrain(net, Xtrn(trnIdx), trnLabs);         % train it    
    classifier{c}=net;      % record this classifiers info.
    fprintf('Total nSV = %d / %d \n', size(net.sv,1),length(trnLabs));

   case 'svmLght'; % use svm-light, classifier{c}=net.
    net = svml('',defOpts,'Kernel',svmlker,'KernelParam',params(2),'C',params(5),'EpsTermin',params(7)); % build svm    
    net = svmltrain( net, Xtrn(trnIdx), trnLabs ) ;
    classifier{c}=net;      % record this classifiers info.
   otherwise; 
    error('Unkonwn SVM engine');return;
  end
  
end

keyboard

% now we've got all C classifiers test them on the testing set and build up
% the confusion matrix.  Row = actual class Col = predicted class
  
% loop over the testing examples for each class
for c=1:L
  pos=find(Ytst==c);  % get the test pts in this class
  disp(['Testing for class ' num2str(c)]);
  % loop over the classifiers finding the one with the highest quality
  decisVal=zeros(length(pos),L);
  for clsfier=1:L
    % test this classifier on this input set
    if ( c == clsfier ) lab=1; else lab=-1; end %get the right label
    switch SVMEng
      
     case 'libSVM'; % use libSVM, classifier{c}={alphaY,svs,bias}
      [classRate,dv]= SVMTest(Xtst(pos,:)',lab(ones(1,length(pos))), ...
                              classifier{clsfier}{1},...
                              classifier{clsfier}{2},...
                              classifier{clsfier}{3},params);
      %dv
%       % Alternative BODGE bodge to fix funny libSVM bug.
%       if ( Ytrn(1) ~= clsfier ) dv=-dv; end; % 
      dv=dv'; % just to be consistent...
      
     case 'irwls'; % use irlws: f(x)=\sum_i \alpha_i*K(x_i,x) + c
      dv = kernel(ker,Xtst(pos,:),classifier{clsfier}{2},par)* ...
           classifier{clsfier}{1} + classifier{clsfier}{3};
      classRate = sum( lab*dv > 0 ) ;
      
     case 'matSVM'; % use matSVM, classifier{c}=net
      [classRate, dv] = svmfwd(classifier{clsfier}, Xtst(pos,:));
      classRate = sum( lab*dv > 0 ) ;
      
     case 'svmLght'; % use svm-light, classifier{c}=net.
      dv = svmlfwd(classifier{clsfier}, Xtst(pos,:));
     otherwise; 
      error('Unkonwn SVM engine');return;
    end
    % record this classifiers output, to use later for multi-class
    decisVal(:,clsfier)=dv;
  end
  % The biggest output for each class is the predicted classification.
  % So this row of the confusion matrix is simply the # of times this
  % classifier gave the highest decision value.
  [Val,I]=max(decisVal');
  for j=1:L;    
    confMx(c,j)=sum(I==j) / length(pos);
  end
end % for c=1:L

