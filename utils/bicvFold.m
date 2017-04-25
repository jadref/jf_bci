function [fIdxs,dimfIdxs]=bicvFold(szX,nFold,varargin)
% generate folding where we split along each dim of X in turn
%
% [fIdxs,dimfIdxs]=bicvFold(szX,nFold,varargin)
% 
% Inputs:
%  szX   - size of the data matrix/tensor to be folded
%  nFold - [numel(szX),1] number of non-overlapping splits to make along each of the dimensios of X
% Options:
%  perm     - (bool) randomally permute the indices in each dimension
%  oneBlock - [bool] use only 1 of the blocks for training
%  repeats  - [int] number of repeats of the folding to generate
%  maxIter  - [int] max iter to search for new non-similar block folding
%  verb     - [int] verbosity level
opts=struct('perm',0,'oneBlock',0,'repeats',1,'maxIter',300,'verb',0);
opts=parseOpts(opts,varargin);

if ( prod(size(szX))~=numel(szX) ) szX=size(szX); end;
nFold(end+1:numel(szX))=nFold(end);
nFold=min(nFold,szX); % at most szX values along each dim
nFolds=max(nFold);
fSize=szX./nFold;
% we make max nFold foldings (initially)
% Generate nFold(di) indices into each dim, i.e. split each di nFold(di) times
dimfIdxs=cell(numel(szX),nFolds);
for di=1:numel(szX);
  if ( opts.perm ) perm=randperm(szX(di)); else perm=[]; end;
  for bi=1:nFold(di); % make the blocks
    idx=floor(fSize(di)*(bi-1))+1:floor(fSize(di)*bi);
    if ( ~isempty(perm) ) idx=perm(idx); end; % permute if wanted
    dimfIdxs{di,bi}=idx;
  end
  % add copies of these blocks to make up any extras needed
  %for fi=bi+1:nFolds; dimfIdxs{di,fi}=dimfIdxs{di,mod(fi-1,nFold(di))+1}; end;
end
fIdxs=-ones([szX nFolds*opts.repeats]);
if ( opts.oneBlock )
  if ( opts.repeats > nFolds ) 
    warning('too many repeats requested: reduced to %d',nFolds); opts.repeats=nFolds; 
  end;
  fi=0;
  shift=zeros(numel(szX),1);
  for repi=1:opts.repeats;
    for rowi=1:nFolds;
      fi=fi+1;
      idx={}; for di=1:size(dimfIdxs,1); idx{di}=dimfIdxs{di,mod(shift(di)+rowi-1,nFold(di))+1}; end
      fIdxs(idx{:},fi)=1;
    end
    % permute to make new fold
    for di=1:numel(szX);
      shift(di)=shift(di)+1;
      if ( shift(di)<nFold(di) ) break; else shift(di)=0; end % stop if 1 shifted  
    end
  end
else
  fIdxs=reshape(fIdxs,[],size(fIdxs,ndims(fIdxs))); % makes dup identification easier
  % add permuted copies of these blocks to make up any extras that are needed
  fi=0; repi=0;
  while (fi<opts.repeats*nFolds && repi<opts.maxIter) 
    repi=repi+1;
    if ( repi==1 )
      bIdx={};for di=1:numel(szX); bIdx{di}=mod(0:nFolds-1,nFold(di))+1;shift(di)=0;end; % no=perm the 1st repeat
    else
      for di=1:numel(szX);
        if( mod(nFolds,nFold(di))~=0 ) bIdx{di}=mod(repi-1+(0:nFolds-1),nFold(di))+1; end;
        bIdx{di}=bIdx{di}(randperm(nFolds)); shift(di)=0; 
      end;      
    end
    for ffi=1:nFolds;
      fIdx=-ones(szX);
      if ( opts.verb>0 ) fprintf('%d)\t',fi+1); end;
      for rowi=1:nFolds;
        idx={};
        if ( opts.verb>0 ) fprintf('('); end;
        for di=1:size(dimfIdxs,1); 
          ei=bIdx{di}(mod(rowi-1,nFolds)+1);
          if ( opts.verb>0 ) fprintf('%d,',ei); end;
          idx{di}=dimfIdxs{di,ei}; 
        end
        if ( opts.verb>0 ) fprintf(')'); end
        fIdx(idx{:})=1;
      end;
      % check for uniqueness - ignore this generation if not unique
      dup = fIdxs(:,1:fi)'*fIdx(:)==prod(szX);
      if ( any(dup) )
        if ( opts.verb>0 ) fprintf('dup (%d)',find(dup)); end;
      else
        fi=fi+1;
        fIdxs(:,fi)=fIdx(:);        
        if ( fi==nFolds*opts.repeats ) break; end;
      end
      if ( opts.verb>0 ) fprintf('\n'); end;
      % permute to make new fold
      for di=1:numel(szX);
        shift(di)=shift(di)+1;
        bIdx{di}=[bIdx{di}(2:end) bIdx{di}(1)];
        if ( shift(di)<nFold(di) ) break; else shift(di)=0; end % stop if 1 shifted
      end
    end
  end
  if ( opts.verb>0 ) fprintf('\n'); end;
  fIdxs=int8(fIdxs(:,1:fi));
  fIdxs=reshape(fIdxs,[szX fi]); % makes dup identification easier
end
return;
%----------------------------------------------
function testCase()
fIdxs=bicvFold([10 10],2)
fIdxs=bicvFold([10 10],2,'perm',1)
fIdxs=bicvFold([10 2],2)
fIdxs=bicvFold([10 2],4)
fIdxs=bicvFold([2 10],4)
fIdxs=bicvFold([3 3],3,'repeats',3); % max 6, 66% train
fIdxs=bicvFold([3 4],[3 4],'repeats',3); % max 12, 66% train
fIdxs=bicvfold([4 4],4,'repeats',300); % max 24, 75% train
clf;mimage(fIdxs);

fIdxs=bicvFold([4 4 2],2,'repeats',100); % max 4, 75% train
fIdxs=bicvFold([4 4 2],3,'repeats',100); % max 36, 75% train
fIdxs=bicvFold([3 3 2],2,'repeats',100);
fIdxs=bicvFold([3 3 2],3,'repeats',100); % max 36, 88% train
fIdxs=bicvFold([3 3 3],3,'repeats',100); % max 36, 88% train
fIdxs=bicvFold([4 4 2],4,'repeats',100); 
fIdxs=bicvFold([4 4 4],4,'repeats',100);
fIdxs=bicvFold([19 1000 2],[4 3],'repeats',100);

clf;mimage(tprod(single(fIdxs>0),[1 2 -3 3],(1:size(fIdxs,3))',-3)); % color says which hypercolx
