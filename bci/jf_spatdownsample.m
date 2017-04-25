function [z]=jf_spatdownsample(z,varargin);
% spatiallly downsample to a new size
%
% Options:
%  capFile  -- [str] the new electrode-layout to use, e.g. 'cap64'         ('')
%  dest_names -- channel names of the destination electrodes  (position info from capFile if not given)
%  pos3d    -- 3d positions of the destination electrodes (from capFile is not given) ([])
%  smthness -- [1x1] smoothness parameter for the downsample (.5)
%  dim      -- the dimension to downsample
%  idx      -- indices of this dim to include in the downsample (1:size(X,dim))
%              'eegonly' : use [di(dim).extra.iseeg] to identify electrodes to use.
%  method   -- [str] one of:                                ('sphericalSplineInterpolate')
%               sphericalInterpolate - gaussian about each electrode
%               spline,sphericalSplineInterpolate - what it says
%               splineCAR            - as for spline but without the constant term
%               nearest              - average of nearest order source electrode
%               slap, csd            - surface-laplacian based on sphericalSplineInterpolate     
%               hjorth               - nearest electrode - average of nearest order electrodes
opts=struct('dim','ch','idx','eegonly','method','sphericalSplineInterpolate',...
            'capFile',[],'capDir',[],'dest_names',[],'pos3d',[],'smthness',[],'order',[],'tol',[],...
            'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);
dim = n2d(z.di,opts.dim); 
idx=opts.idx; if ( isempty(idx) ) idx=1:size(z.X,dim); end;
if ( ischar(idx) && strcmp(idx,'eegonly') )
   if ( isfield(z.di(dim).extra,'iseeg') ) idx=[z.di(dim).extra.iseeg]; 
   else idx=1:size(z.X,dim); end;
end;
if ( islogical(idx) ) idx=find(idx); end;

electrodePos=[z.di(dim).extra(idx).pos3d];
keepIdx=setdiff(1:size(z.X,dim),idx);

xy=[];
if ( ~isempty(opts.capFile) && isempty(opts.pos3d) )   
  if ( isempty(opts.dest_names) ) % only the electrodes mentioned in dest_names are wanted
    [Cname latlong xy xyz]=readCapInf(opts.capFile,opts.capDir);
  else % get pos info for these electrodes from the given capFile
    di=addPosInfo(opts.dest_names,opts.capFile,[],[],[],opts.capDir);
    Cname=di.vals;
    xyz  =[di.extra.pos3d];
  end
elseif ( ~isempty(opts.pos3d) )
  Cname={}; for ci=1:size(opts.pos3d,2); Cname{ci}=sprintf('%d',ci); end;
  if ( ~isempty(opts.dest_names) ) Cname=opts.dest_names; end;
   xyz=opts.pos3d;
   if ( size(xyz,1)==2 ) 
      xy=xyz;      
      latlong= [sqrt(sum(xy.^2,1)); atan2(xy(2,:),xy(1,:))];
      xyz= [sin(latlong(1,:)).*cos(latlong(2,:)); sin(latlong(1,:)).*sin(latlong(2,:)); cos(latlong(1,:))]; %3d
   end
else % use the electrodePos
   Cname=z.di(dim).vals(idx);
   xyz=electrodePos;
   if ( isfield(z.di(dim).extra,'pos2d') ) xy=[z.di(dim).extra(idx).pos2d]; end;
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
  R=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order,'spline',opts.tol,opts.verb);

 case {'splineCAR'};
  R=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order,'splineCAR',opts.tol,opts.verb);

 case {'slap','csd'};               
  R=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order,'slap',opts.tol,opts.verb);  

 case 'nearest';                    
  R=nearestPos(electrodePos,xyz,opts.order);
  if ( ~isempty(opts.order) ) R=R./opts.order; end;

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
di=mkDimInfo(size(R),'ch',[],[],'ch_ds',[],{Cname{:} z.di(dim).vals{keepIdx}});
di(1)=z.di(n2d(z.di,opts.dim));
if( ~isempty(keepIdx) )
   fn=fieldnames(z.di(dim).extra(1));
   for fi=1:numel(fn);
     [di(2).extra(numel(Cname)+1:end).(fn{fi})]=z.di(dim).extra(keepIdx).(fn{fi});
   end
end
[di(2).extra(1:numel(Cname)).pos3d]=num2csl(xyz);
[di(2).extra(1:numel(Cname)).pos2d]=num2csl(xy);
if( numel(di(2).extra)>numel(Cname) ) [di(2).extra(numel(Cname)+1:end).pos2d]=deal([-1;-1]); end;
iseeg=false(size(R,2),1); iseeg(1:nChOut)=true;
[di(2).extra.iseeg]=num2csl(iseeg);

% apply the spatial filter to the data
ozdi=z.di;
z=jf_linMapDim(z,'dim','ch','mx',R,'di',di,'sparse',1);

% update the prep info
info=z.prep(end).info;
info.R=R; info.di=di;
summary=sprintf('%s mapped %s -> %s',opts.method,sprintf('%d %ss',numel(ozdi(n2d(ozdi,'ch')).vals),ozdi(n2d(ozdi,'ch')).name),sprintf('%d %ss',numel(z.di(n2d(z.di,'ch')).vals),z.di(n2d(z.di,'ch')).name));
tmpprep = jf_addprep([],mfilename,summary,opts,info);
z.prep(end)=tmpprep.prep;
z.summary=jf_disp(z);
return;

function R=nearestPos(src,dest,k)
% map the positions onto the sphere
%  src    - [3 x N] old electrode positions
%  dest   - [3 x M] new electrode positions
%  k      - [int] return k-nearest electrodes for each output  (1)
% Outputs:
%  R      - [M x N] linear mapping matrix between old and new co-ords
if ( nargin < 3 || isempty(k) ) k=1; end;
src   = repop(src,'./',sqrt(sum(src.^2)));
dest  = repop(dest,'./',sqrt(sum(dest.^2)));
cosDS = dest'*src; % angles between destination positions [MxN]
R     = zeros(size(cosDS));
if ( k==1 ) % special case, use hungarian algorithm to find best match between pairs electrodes
  for ei=1:size(R,1);
	 [ans,ij]=max(cosDS(:)); [i,j]=ind2sub(size(cosDS),ij);
	 R(i,j)=1;
	 cosDS(i,:)=-1; cosDS(:,j)=-1; % remove this pair from the set of possible pairs
  end
else % k-nearest, so allow overlapping sets
  for i=1:size(R,1);  
    [ans,nni]=sort(cosDS(i,:),'descend'); nni=nni(1:k);
	 R(i,nni)=1;
  end
end
return;


%----------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
z=jf_retain(z,'dim','ch','vals',{'2' '3' '4' '5' '6'});
z=jf_spatdownsample(z,'method','hjorth');
z=jf_spatdownsample(z,'dest_names',{'Cz' 'CPz'},'capFile','1010'); % named electrodes and positions

[Cnames256 ll256 xy256 xyz256]=readCapInf('cap256');
[Cnames64 ll64 xy64 xyz64]=readCapInf('cap64');
R=sphericalInterpolate(xyz256,xyz64,.95);
clf;axes('outerposition',[.0 .35 1 .65]);imagesc(R');ylabel('256ch');axes('outerposition',[.0 .0 1 .35]);plot(1:size(R,1),R'+randn(size(R))'*.01,'b.');set(gca,'Ylim',[.1 1],'Xlim',[1 size(R,1)]); xlabel('64ch'); ylabel('weight');
