function [selCols,rel,opt]=rfe(X,Y,w,numDim,varargin)
% Perform 'Recursive Feature Elimination' feature selection algorithm
%
% [selCols,rel]=rfe(X,Y,w,numDim,varargin)
% Run linear "Recursive Feature Elimination"
%Optional arguments:
% maxRm  - at most this frac rem dims per iter
% fracRm - at most this num dim per iter
% verb   - verbosity level
if ( nargin < 4 ) error('Insufficient arguments'); end;
opt=struct('maxRm',1,'fracRm',.5,'svmArgs',{{'lin','libSVM'}},'verb',0);
[opt,varargin]=parseOpts(opt,varargin);
if(~isempty(varargin))error('Unrecognised Option(s)'); end;

[W,mu,covXX]=multiCutDataStats(X,Y,w);

selCols=1:size(X,2);
while (numel(selCols)>numDim)
   [mcClsfr]=mcSVMTrn(X(:,selCols),Y,opt.svmArgs{:});
   rel=mcClsfr.ocClsfier{1}.svs*mcClsfr.ocClsfier{1}.alphaY';

   [ans,srelIdx]=sort(abs(rel),'ascend');        % sort by wght
   % # pts to remove this time
   nRm=min([numel(selCols)-numDim,opt.maxRm,floor(numel(selCols)*opt.fracRm)]);
   selCols=selCols(srelIdx(nRm+1:end));          % remove the selected # dims
   if ( opt.verb ) fprintf('.'); 
      if ( opt.verb > 1 ) fprintf('Removing %d dims\n',nRm); end;
   end;
   
end
orel=rel;rel=zeros(1,size(X,2));rel(selCols)=orel(srelIdx(nRm+1:end));
return;

%------------------------------------------------------------------------
function testCase()
[X,Y]=mkMultiClassTst([-1 0; 1 0],[400 400],[.3 .3; .3 .3]);
[selcols,rel]=rfe(X,Y,[],1);
