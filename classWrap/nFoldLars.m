function [foldRes,tstIdxs]=nFoldLars(X,Y,tstIdxs,varargin)
% Run an n-fold set of experiments on the LARS algorithm.
% Inputs:
% X,Y - features and targets [N,dim],[N,L]
% tstIdxs - [N x nFold] logical indicator array of the elements in 
%            the training folds
% Outputs:
%  foldRes - structure holding the results for each of the folds.  N.B. for
%  performance results this has one column for each distinct set of
%  paramters computed by LARS.
%
% %Id%
if ( nargin < 2 ) error('Insufficient arguments'); end;
if ( nargin < 3 | isempty(tstIdxs) ) tstIdxs=gennFold(Y,10); end;
for i=1:size(tstIdxs,2);
   figure(1);
   [AIdx,beta,olsSoln,C,SSE,singDim]=...
       lars(X(~tstIdxs(:,i),:),Y(~tstIdxs(:,i)),varargin{:});
   figure(2);clf;
   subplot(3,1,1:2);
   labScatPlot(X(~tstIdxs(:,i),:)',Y(~tstIdxs(:,i)));hold on;
   for j=1:size(beta,2);drawLine(beta(:,j),0,[-1 -1],[1 1]);end;
   pause(0.01);
   foldRes(i).AIdx=AIdx; foldRes(i).beta=beta; foldRes(i).singDim=singDim;
   % evaluate the testing performance, accross all the output parameter 
   % settings.
   f=X(tstIdxs(:,i),AIdx)*foldRes(i).beta;            % comp predictions
   subplot(313);labScatPlot(X(tstIdxs(:,i)',:),Y(tstIdxs(:,i)));hold on
   for j=1:size(f,2);
      drawLine(foldRes(i).beta(:,j),0,[-1 -1],[1 1]);pause(0.1);
      [conf,binCls,eeoc,auc]=dv2conf(f(:,j),Y(tstIdxs(:,i)));
      foldRes(i).conf(:,:,j)=conf;  foldRes(i).binCls(:,j)=binCls(:);
      foldRes(i).eeoc(:,j)=eeoc(:); foldRes(i).auc(:,j)=auc(:);   
   end
   pause
end
% TESTCASE
% X=importdata('~/temp/diabetes.sdata.txt');size(X)
% tstIdxs=gennfold(10,ones(size(X,1),1)); % N.B. requ'd as Y isn't class lab
% [foldRes,tstFolds]=nFoldLars(X(:,1:end-1),X(:,end),tstIdxs,'verb',2,'plot',1)
% 
% [X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 20],[.3 .3; .3 .3; .2 .2]);Y=2*(Y==1)-1; % convert to +/- 1
% tstIdxs=gennfold(10,Y,'perm',randperm(size(X,1)));
% [foldRes,tstFolds]=nFoldLars(X,Y,tstIdxs,'verb',2,'plot',1);
