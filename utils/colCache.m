function [cache,cacheCIdxs,newCIdxs]=colCache(cache,colFn,selCIdxs,maxCols)
% Get the selected columns from a cache.
% generate the kernel for these rows, using the cache if necessary..
% Do replacement using LRU policy.
%
% $Id: colCache.m,v 1.4 2006-11-16 18:23:48 jdrf Exp $
MAXCOLS=1e4;
if ( nargin < 4 | isempty(maxCols) ) maxCols=MAXCOLS; end
if ( isempty(cache) )  % initialise the cache.
  cache.K=single([]); 
  cache.colIdxs=[];
  % info for LRU replacement
  cache.maxCols=maxCols; maxCols, cache.t=1; cache.colTs=zeros(maxCols,1); 
  cacheCIdxs=1:length(selCIdxs);
  newselCIdx=1:length(selCIdxs); % force to be row vector
  newCIdxs=1:length(newselCIdx);
else % compute the features for the selected rows not in the cache.
  cache.t=cache.t+1;
  newselCIdx=[];newCIdxs=[];cacheCIdxs=[];
  for j=1:length(selCIdxs);
    cacheCol=find(cache.colIdxs==selCIdxs(j));
    if ( cacheCol ) % found in the cache
      cacheCIdxs(j)=cacheCol;
      cache.colTs(cacheCol)=cache.t;  % mark as used to stop removing later
    else % not in the cache so mark it to be added      
      newselCIdx(end+1) =j;      % j'th entry is new col to be added
      cacheCIdxs(j)  =0;           % mark as unused, fill in later
    end
  end
end
if ( ~isempty(newselCIdx) ) % add new rows to the cache
  ncurCols=length(cache.colIdxs); nnewCols=length(newselCIdx);
%   if ( ncurCols+nnewCols < cache.maxCols )     
%     newCIdxs=ncurCols+1:ncurCols+nnewCols; % add new ones to the end
%   else % remove old
    % find LRU elements to replace (or unused if there are any!)
    [ts,lru]=sort(cache.colTs,'ascend');
    if ( ts(1)==cache.t | ts(nnewCols)==cache.t ) 
      % problem as we've not got any cols we're not using!
      error('Using more cols than the cache size!');
    end
    newCIdxs=lru(1:nnewCols); % replace LRU cols
%   end
  
    %for 
      i=1:length(newCIdxs); % add col at a time to save ram
      cache.colIdxs(newCIdxs)=selCIdxs(newselCIdx);  
      cache.K(:,newCIdxs(i))=single(colFn(cache.colIdxs(newCIdxs(i))));
      cache.colTs(newCIdxs(i))=cache.t;      % update time-stamps 
      cacheCIdxs(newselCIdx(i))=newCIdxs(i); % record col in cache
                                             %end
end
if ( cache.t==0 & ~isinf(cache.maxCols) ) % init cache to prevent copies
  nK=cache.K;cache.K=zeros(size(nK,1),maxCols,'single');cache.K(:,newCIdxs)=nK;
end
cache.colTs(cacheCIdxs)=cache.t;  % update time-stamps for accessed cols
fprintf('Cache hit rate %d elm %d/%d=%g\n',length(cache.colIdxs),...
        length(newCIdxs),length(selCIdxs),...
        1-length(newCIdxs)/max(1,length(selCIdxs)));