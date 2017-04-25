function x=interpnans(x,dim,varargin);
% FIXGAPS Linearly interpolates gaps in a time series
% YOUT=FIXGAPS(YIN,dim,...) linearly interpolates over NaN
% in the input time series (may be complex), but ignores
% trailing and leading NaN.
%
if ( nargin<2 || isempty(dim) ) dim=1; end;
extrap=false;
if ( ~isempty(varargin) && ischar(varargin{1}) && strcmpi(varargin{1},'extrap') ) extrap=true; varargin(1)=[]; end;
szX=size(x);
% make 3d with time in 2nd dimension
x=reshape(x,[max([1 prod(szX(1:dim-1))]),szX(dim),max([1 prod(szX(dim+1:end))])]);
fprintf('interpnans:');
for li=1:size(x,1);
  for ti=1:size(x,3);			 
	 x(li,:,ti)=fixgapsvec(x(li,:,ti),extrap,varargin{:});
  end
  textprogressbar(li,size(x,1));
end
fprintf('\n');
x=reshape(x,szX);
return;

function y=fixgapsvec(x,extrap,varargin);
y=x;
bd=isnan(x);  % indicator for the bad points
if ( ~any(bd) ) return; end;
gd=find(~bd); % indices of the good points
if ( ~extrap ) bd([1:(min(gd)-1) (max(gd)+1):end])=0; end; % don't extrapolate outside good points
y(bd)=interp1(gd,x(gd),find(bd),varargin{:});
return;
