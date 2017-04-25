function [X,pow,op]=normalize(X,dim,type,minStd,zeronan,wght);
% normalise along a given dimension, i.e. unit/zero ave power by dividing by the RMS amp
%
%   [X,pow,op]=normalize(X,dim,type,minStd,zeronan,wght);
%
% Inputs:
%  X
%  dim    -- the dimension(s) over which we compute the statistics
%  type   -- 'str' type of normalization to do;               ('rel')
%              one-of: 'abs'=zero-mean, 'rel'=unit-power
%  minStd -- min allowed std                                  (1e-5)
%  zeronan -- [bool] set elements which are nan to value 0    (0)
%  wght    -- [n-d] weighting over points for the compuation  ([])
if ( nargin<2 || isempty(dim) ) dim=ndims(X); end;
if ( nargin<3 || isempty(type) ) type='rel'; end;
if ( nargin<3 || isempty(minStd) ) minStd=0; end;
if ( nargin<4 || isempty(zeronan) ) zeronan=false; end;
if ( nargin<5 || isempty(wght) ) wght=[]; end;

if ( zeronan ) X(isnan(X))=0; end;
if ( ~isempty(wght) ) X=repop(X,'*',wght); end;

sz  = size(X); sz(end+1:max(dim))=1; 
idx = 1:numel(sz); idx(dim)=-idx(dim);
if ( isreal(X) ) pow= tprod(X,idx,[],idx);
else             pow=(tprod(real(X),idx,[],idx)+tprod(imag(X),idx,[],idx));
end
pow= sqrt(abs(pow))./sqrt(prod(sz(dim))); 
pow(pow==0 | isnan(pow))=1; pow=max(pow,minStd); % fix division by 0
if ( strcmp(type,'abs') )   op='-';
elseif (strcmp(type,'rel')) op='/';
else error('unknown center type: %s',type); 
end
X   = repop(X,op,pow);
return;
