function [V,D]=drawellipse(cent,Mi,npts,varargin);
% function drawellipse(cent,Mi,npts,varargin);
%
%  cent - center location
%  Mi   - [2x2] covariance matrix
%          OR
%         [l1, l2, theta]
if(nargin<2 || isempty(Mi)  ) Mi=eye(2); end;
if(nargin>2 && ischar(npts) ) varargin={npts varargin{:}}; npts=[]; end;
if(nargin<3 || isempty(npts)) npts=50; else  npts=npts/2;end;
if(isempty(cent) ) cent=[0;0]; end;
hold on;
theta=0;
if ( numel(Mi)==3 ) theta=Mi(3); Mi=Mi(1:2); end;
if ( size(Mi,1)==2 && size(Mi,2)==2 )
   [V D]=eig(Mi); D=sqrt(abs(diag(D)));
else
   V=[1 0;0 1];   D=Mi;
end

alpha=atan2(V(4),V(3));
t = linspace(0,2*pi,npts*2);
y=D(2)*sin(t);
x=D(1)*cos(t);

xbar=x*cos(alpha+theta) + y*sin(alpha+theta);
ybar=y*cos(alpha+theta) - x*sin(alpha+theta);
plot(ybar+cent(1),xbar+cent(2),varargin{:});
hold off;
end
