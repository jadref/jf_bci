function [X,mu,op]=center(X,dim,type,zeronan);
% center along a given dimension, i.e. make all entries zero mean
%
% Inputs:
%  X
%  dim    -- the dimension(s) over which to compute the statistics
%  type   -- 'str' type of normalization to do;               ('rel')
%              one-of: 'abs'=zero-mean, 'rel'=unit-power
%  zeronan -- [bool] set elements which are nan to value 0    (0)
if ( nargin<2 || isempty(dim) ) dim=ndims(X); end;
if ( nargin<3 || isempty(type) ) type='abs'; end;
if ( nargin<4 || isempty(zeronan) ) zeronan=false; end;
mu=X;
if ( zeronan ) mu(isnan(mu(:)))=0; else mu(isnan(mu))=mean(mu(~isnan(mu(:)))); end;
N=1;
for di=1:numel(dim); N=N*size(mu,dim(di)); mu=sum(mu,dim(di));  end; 
mu=mu./N;
if ( strcmp(type,'abs') )   op='-';
elseif (strcmp(type,'rel')) op='/';
else error('unknown center type'); 
end
X   = repop(X,op,mu);
return;
