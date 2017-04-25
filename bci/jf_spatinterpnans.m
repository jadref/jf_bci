function [z]=jf_spatinterpnans(z,varargin);
% sphericl spline to spatiallly interperate missing data marked with nan's 
%
% Options:
%  capFile  -- [str] the new electrode-layout to use, e.g. 'cap64'                    ('')
%  dest_names -- channel names of the destination electrodes                          ([])  
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
            'subIdx',[],'verb',1);
opts=parseOpts(opts,varargin);
dim = n2d(z.di,opts.dim); 
chidx=opts.idx; if ( isempty(chidx) ) chidx=1:size(z.X,dim(1)); end;
if ( ischar(chidx) && strcmp(chidx,'eegonly') )
   if ( isfield(z.di(dim(1)).extra,'iseeg') ) chidx=[z.di(dim(1)).extra.iseeg]; 
   else chidx=1:size(z.X,dim(1)); end;
end;
if ( islogical(chidx) ) chidx=find(chidx); end;

electrodePos=[z.di(dim(1)).extra(chidx).pos3d];
keepIdx=setdiff(1:size(z.X,dim(1)),chidx);

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
   Cname=z.di(dim(1)).vals(chidx);
   xyz=electrodePos;
   if ( isfield(z.di(dim(1)).extra,'pos2d') ) xy=[z.di(dim(1)).extra(chidx).pos2d]; end;
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

% Compute info for mapping from old to new positions: R - [nChout x nChin]
% N.B. to compute the spatial filter use:  W = pinv(Gss)*Gds
[R,Gss,Gds,Hds]=sphericalSplineInterpolate(electrodePos,xyz,opts.smthness,opts.order,'spline',opts.tol,opts.verb-1);

badch=isnan(z.X);
% accumulate away dims for which we fix the bad-ness
if( numel(dim)>1 )
  for di=2:numel(dim); badch=any(badch,dim(di)); end;
end

% pre-build the spatial filter matrix
nChIn=size(electrodePos,2); nChOut=size(xyz,2);
% include non-mapped stuff at the end
W=zeros(nChIn+numel(keepIdx),nChOut+numel(keepIdx)); % [ (#src+keep) x (#dest+keep) ] =>  x' = W'*x;
W(size(W,1)+1:end,keepIdx)=eye(numel(keepIdx)); % pass-through mapp for the ignored channels

szX  =size(z.X);
% TODO []: make work with arbitary order of ch-dim and summed-out dims
if( dim(1)~=1 || ~all(sort(dim(:)','ascend')==[1:numel(dim)]) ) 
   error('only supported for consequetive dims currently!'); 
end;
idx  =repmat({':'},1,numel(dim)+1);
N    =size(badch(:,:),2);
nW   =0; Ws={};
sampi=1;
if( opts.verb>0 ) fprintf('spatinterpnans:'); end;
while (sampi<N);
  % get the range of elements to with the same badch set
  starti=sampi;
  badchi=badch(:,starti);
  for sampi=starti+1:N;
    if( ~all(badch(:,sampi)==badchi) ) break; end; % stop 1-past range of same goodness
    if (opts.verb>0 ) textprogressbar(sampi,N,100); end;
  end
  if( sampi==N ) sampi=N+1; end; %end of data is special case...
  if( ~any(badchi) ) continue; end; % don't process when nothing is bad
  % update the set to apply to.
  idx{end}=starti:sampi-1;  
  % compute the updated spatial filter
  goodchi=~badchi;
  C   = [Gss(goodchi,goodchi) ones(sum(goodchi),1);... % add the constant terms to Gss
         ones(1,sum(goodchi)) 0];
  iC  = pinv(C); % solve the system, i.e. mapping to coefficient space
  Wi  = [Gds(:,goodchi) ones(size(Gds,1),1)]*iC(:,1:end-1); % [nDest x nSrc]
  % compute from scratch to compare
  R=sphericalSplineInterpolate(electrodePos(:,goodchi),xyz,opts.smthness,opts.order,'spline',opts.tol,opts.verb-1);
  mad(Wi,R)

  % insert into the whole channel mapping set
  W(chidx,chidx)  =0;
  W(goodchi,chidx)=Wi';
                          % save it for later
  nW=nW+1; Ws(:,nW)={goodchi Wi};
  % apply the spatial fitler to this subset of data
  X = z.X(idx{:}); X(badchi,:)=0; % extract this set, and clear the NaNs to 0's
  z.X(idx{:})=tprod(W,[-dim(1) dim(1)],X,[1:dim(1)-1 -dim(1) dim(1)+1:numel(szX)]);  
end
if( opts.verb>0 ) fprintf('\n'); end;

                                % update the prep info
info.Ws=Ws;info.Gss=Gss;info.Gds=Gds;info.Hds=Hds;
summary=sprintf('%s mapped %s -> %s in %d blocks',opts.method,sprintf('%d %ss',size(Gds,2),z.di(dim(1)).name),sprintf('%d %ss',size(Gds,1),z.di(dim(1)).name),nW);
z=jf_addprep(z,mfilename,summary,opts,info);
return;

%----------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();oz=z;
z.X(1,:,1:10)=NaN; % mark with NaNs...
zi=jf_spatinterpnans(z); 
zi=jf_spatinterpnans(z,'dim',{'ch' 'time'});
clf;mimage(oz.X(:,:,[1 11]),zi.X(:,:,[1 11]),'diff',1)
%z=jf_spatinterpnans(z,'dest_names',{'Cz' 'CPz'},'capFile','1010'); % named electrodes and positions
