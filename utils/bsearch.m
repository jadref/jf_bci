function [idx]=bsearch(tab,vals,tol,minmax)
% returns the row index *closest* to val in the sorted table tab
%
%  [idx]=bsearch(tab,vals,tol)
%
% Inputs:
%  tab  -- [N x d] or [d x N] *sorted* table to search
%  vals -- [nval x d] set of values to find
%  tol  -- [float] accuracy range equality - (eps*1e6)
%  minmax- [] min/max value along each dim   ([])
% Outputs:
%  idx  -- [nval x 1] set of *nearest* row indices (N.B. check if exact match wanted)
if ( nargin<3 || isempty(tol) ) tol=eps(tab(1))*1e6; end;
if ( nargin<4 ) minmax=[]; end;
[nrows,ncols]=size(tab);
maxStep=ceil(log2(nrows));
if ( ~isempty(minmax) ) % bound test by range of table
   for di=1:size(vals,2); 
      vals(vals(:,di)<minmax(1,di)-tol,di)=minmax(1,di); 
      vals(vals(:,di)>minmax(2,di)+tol,di)=minmax(2,di);
   end; 
end
rng=[ones(1,size(vals,1)); nrows*ones(1,size(vals,1))]; idx=zeros(1,size(vals,1));
lb=true(size(vals,1),1); ub=true(size(vals,1),1);
if ( tab(1,1) > tab(end,1) ) % descending order

   for si=1:maxStep; % loop over decisions
      idx = floor(mean(rng,1));
      lb(:)=false; ub(:)=false; % both lb and ub
      for coli=1:ncols; % loop over cols to identify if idx is lower/upper bound 
         eq=~lb&~ub; % only update unsure values
         if ( ~any(eq) ) break; end;
         lb(eq) = vals(eq,coli) < tab(idx(eq),coli)-tol;
         ub(eq) = vals(eq,coli) > tab(idx(eq),coli)+tol;
      end;
      rng(1,lb) = idx(lb); % update lower-bound
      rng(2,ub) = idx(ub); % update upper-bound
      eq=~lb&~ub; rng(1,eq) = idx(eq); % equality become lower bound
   end

else % ascending order

   for si=1:maxStep; % loop over decisions
      idx = floor((rng(1,:)+rng(2,:))./2);
      lb(:)=false; ub(:)=false; % neither lb or ub
      for coli=1:ncols; % loop over cols to identify if idx is lower/upper bound 
         eq=~lb&~ub; % only update unsure values
         if ( ~any(eq) ) break; end;
         lb(eq) = vals(eq,coli) > tab(idx(eq),coli)+tol;
         ub(eq) = vals(eq,coli) < tab(idx(eq),coli)-tol;
      end;
      rng(1,lb) = idx(lb); % update lower-bound
      rng(2,ub) = idx(ub); % update upper-bound
      eq=~lb&~ub; rng(1,eq) = idx(eq); % equality become lower bound
   end

end
% use the closest bound for return
bd=[sum(abs(vals-tab(rng(1,:),:)),2) sum(abs(vals-tab(rng(2,:),:)),2)]; % L1 dist
[bd,bi]=min(bd,[],2);
idx=rng(1,:); idx(bi==2)=rng(2,bi==2);

return;
%------------
function testCase();
X=randn(100,3); [X]=sortrows(X); % generate sorted list of stuff
bsearch(X,X(23,:))
bsearch(X,X([23 50 12 12 20],:))

X=X(end:-1:1,:); % reverse sort order
