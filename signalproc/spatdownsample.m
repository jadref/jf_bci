function [R,xy,xyz,iseeg]=spatdownsample(src,dest,varargin);
% spatiallly downsample to a new size
% [R,xy,xyz,iseeg]=sphericalInterpolate(src,dest,varargin);
%
% Options:
%  smthness -- [1x1] smoothness parameter for the downsample (.5)
%  dim      -- the dimension to downsample
%  idx      -- indices of this dim to include in the downsample (1:size(X,dim))
%              'eegonly' : use [di(dim).extra.iseeg] to identify electrodes to use.
%  method   -- [str] one of:                                ('sphericalSplineInterpolate')
%               sphericalInterpolate - gaussian about each electrode
%               sphericalSplineInterpolate - what it says
%               nearest              - average of nearest order source electrode
%               slap, csd            - surface-laplacian based on sphericalSplineInterpolate     
%               hjorth               - nearest electrode - average of nearest order electrodes
%  pos3d    -- 3d positions of the destination electrodes
opts=struct('idx',[],'method','sphericalSplineInterpolate',...
            'capDir',[],'smthness',[],'order',[],'pos3d',[],'subIdx',[]);
opts=parseOpts(opts,varargin);
idx=opts.idx; if ( isempty(idx) ) idx=1:size(src,2); end;
if ( islogical(idx) ) idx=find(idx); end;
if ( nargin<2 ) dest=[]; end;

electrodePos=src;
keepIdx=setdiff(1:size(src,2),idx);

xy=[];
if ( isempty(dest) ) % use the electrodePos
  Cname=1:size(src,2);
  xyz=src;
elseif ( ischar(dest) )
   [Cname latlong xy xyz]=readCapInf(dest,opts.capDir);
elseif ( isnumeric(dest) ) 
   Cname=1:size(dest,2);
   xyz=dest;
   if ( size(xyz,1)==2 ) 
      xy=xyz;      
      latlong= [sqrt(sum(xy.^2,1)); atan2(xy(2,:),xy(1,:))];
      xyz= [sin(latlong(1,:)).*cos(latlong(2,:)); sin(latlong(1,:)).*sin(latlong(2,:)); cos(latlong(1,:))]; %3d
   end
end
% compute 2-d coords too
if ( isempty(xy) ) 
  % map from 3d to 2d -- N.B. we use the distance on the sphere to ensure 
  % electrodes don't overlap near the equator
  if ( all(xyz(3,:)>-eps) ) cz=0; else cz= mean(xyz(3,:)); end % center
  r = abs(max(abs(xyz(3,:)-cz))*1.1); if( r<eps ) r=1; end;  % radius
  h = xyz(3,:)-cz;  % height
  rr=sqrt(2*(r.^2-r*h)./(r.^2-h.^2)); % arc-length to radial length ratio
  xy = [xyz(1,:).*rr; xyz(2,:).*rr];
end

% Compute linear mapping from old to new positions: R - [nChout x nChin]
switch (opts.method)
 case 'sphericalInterpolate';       
  R=sphericalInterpolate(electrodePos,xyz,opts.smthness);%[nChout x nChin]
 case {'spline','sphericalSplineInterpolate'}; 
  R=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order);
 case {'slap','csd'};               
  R=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order,'slap');  
 case 'nearest';                    
  R=nearestPos(electrodePos,xyz,opts.order)./opts.order;
 case 'hjorth';                     
  nNearest=opts.order; if(isempty(nNearest)) nNearest=5;end
  R=nearestPos(electrodePos,xyz,1)*(1+1/nNearest) - nearestPos(electrodePos,xyz,nNearest)./nNearest;
 otherwise; error(sprintf('Unrecoginised downsample method: %s',opts.method));
end
nChIn=size(R,2); nChOut=size(R,1);

% include non-mapped stuff at the end
Rout=zeros(size(R,1)+numel(keepIdx),size(R,2)+numel(keepIdx));
Rout(1:size(R,1),idx)=R; % map the actual channels
Rout(size(R,1)+1:end,keepIdx)=eye(numel(keepIdx));
R=Rout';
iseeg=false(size(R,2),1); iseeg(1:nChOut)=true;
return;
function testCase()
[Cname,ll,xy,xyz]=readCapInf('cap64');
R=spatdownsample(xyz);
