function [A,src,dest]=toyFwdMx(src,dest,epsilon_0)
% compute a toy Fwd Mx based upon distances
%
% [A,src,dest]=mksfToy(src,dest,[,options])
%
% Inputs:
%  src -- [ 2 x d1 ] or [ 3 x d1 ] positions of the sources  (2)
%         OR
%         [ 1 x 1 ] number of equally spaced electrodes -- on circle of radius .5
%  dest-- [ 2 x d2 ] or [ 3 x d2 ] positions of the electrodes (10)
%         OR
%         [ 1 x 1 ] this number of equally spaced electrodes
%            (10 semi-circle on circumference circle radius 1 + 2 at +/1 .25,0)
%  epsilon_0 -- [1x1] value of permittivity for signal attenuation, (.1)
% Outputs:
%  A         -- [ d x d ] forward source mixing matrix
if ( nargin < 1 || isempty(src) ) src=2; end;
if ( nargin < 2 || isempty(dest) ) dest=10; end;
if ( nargin < 3 || isempty(epsilon_0) ) epsilon_0=.1; end;
if ( numel(src)==1 ) % generate requested number of source locations
   % build the default source locations
  if ( size(dest,1)==2 ) % 2-d problem
    src = .5*[cos(linspace(0,pi,src))' sin(linspace(0,pi,src))']';
  else % 3-d problem, radius .5
    src = .5*pointsOnSphere(src,'equal',1); % relatively uniform spacing, top hemi-sphere only
  end
end
if ( numel(dest)==1 ) % generate requested number of electrodes
   % build the default electrod locations
  if ( size(src,1)==2 ) % 2-d problem
    dest = [cos(linspace(0,pi,dest))' sin(linspace(0,pi,dest))']';
  else % 3-d problem
    dest = pointsOnSphere(dest,'equal',1); % relatively uniform spacing, top hemi-sphere only
  end
end

% Compute the mixing matrix
% Compute the electrode pairwise distance matrix
s2 = sum(src.^2,1)';
d2 = sum(dest.^2,1)';
D2 = repop(repop(-2*src'*dest,'+',s2),'+',d2');
% Attenuate with epsilon_0/r.^2
A  = D2/epsilon_0; A(A<eps)=1; A = 1./A;
return;
%-------------------------------------------------------------------------
function testCase()
[A,pos2d]=toyFwdMx(10);