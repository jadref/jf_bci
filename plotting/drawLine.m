function drawLine(alpha,b,minX,maxX,varargin)
% Draw a clipped line on the axes
%
% drawLine(alpha,b,minX,maxX,varargin)

% 0=m'*x +b => x(i) = (m(1:i-1 i+1:end)*x(1:i-1 i+1:end) + b) / -m(i)
if ( nargin < 2 || isempty(b) ) b=0; end;
if ( nargin < 3 || isempty(minX) ) minX=-ones(size(alpha)); end;
if ( nargin < 4 || isempty(maxX) ) maxX=ones(size(alpha)); end;
if ( nargin < 5 ) varargin{1}='k-'; end;
if ( numel(alpha)==1 ) % deal with 1-d inputs as special case
   plot([minX(1) maxX(1)],([b b])/-alpha(1) ,varargin{:});
   return; 
end;
alpha(abs(alpha)<eps)=eps;
m = alpha(2)/alpha(1); % gradient
bt=[(alpha(2)*minX(2)+b)/-alpha(1) minX(2)];
tp=[(alpha(2)*maxX(2)+b)/-alpha(1) maxX(2)];
lf=[minX(1) (alpha(1)*minX(1)+b)/-alpha(2)];
rg=[maxX(1) (alpha(1)*maxX(1)+b)/-alpha(2)];
% [tp; rg; bt; lf]
if     ( tp(1) <= maxX(1) ) 
   tr=tp;
   if (  rg(2) < maxX(2) && maxX(2)-rg(2) < maxX(1)-tp(1) ) tr=rg; end
elseif ( rg(2) <  maxX(2) ) tr=rg; 
else % line outside the bb..
  if ( maxX(1)-tp(1) > maxX(2)-rg(2) ) tr=tp; else tr=rg; end; 
end;
if     ( bt(1) >= minX(1) ) 
   bl=bt; 
   if ( lf(2) > minX(2) && lf(2)-minX(2) < bt(1)-minX(1) ) bl=lf; end;
elseif ( lf(2) >  minX(2) ) bl=lf; 
else % line outside the bb..
  if ( bt(1)-minX(1) > lf(2)-minX(2) ) bl=bt; else bl=lf; end; 
end;
plot([bl(1) tr(1)],[bl(2) tr(2)],varargin{:});
end