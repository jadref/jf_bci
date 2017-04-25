function [z]=jf_invSrcLoc(z,varargin);
% perform a beamforming style source localisation on the input EEG data
%
% Options:
%  dim -- dimension which contains the channels
%  fwdMx -- [str] file name containing the forward or system matrix
%           OR
%           [nElect x nSrc x nOri] matrix containing the fwd/sys matrix
%  fwdDi -- [dimInfo] struct containing the meta-info about fwdMx
%  srdDepth -- [float] max depth of sources to fit for (inf)
%  fwdMxfids -- [str] file name containing the fid locations for the fwdMx
%               OR
%               [3 x 3] matrix of 3-d coords of the fids, NAS,EARL,EARR
%  fids      -- same as fwdMxfids but for the data in z
%  idx       -- [int] subset of electrodes to use in the inversion
%  method    -- [str] beamforming method to use, currently only 'lcmv' ('lcmv')
%  lambda    -- [float] regularisation paremter for the inverse method
%  nai       -- [bool] record 'Neural-activity-indicies' (1)
%  maxPowOri -- [bool] only report the time course in the maxPow direction for each
%               source location
opts=struct('dim','ch',...
            'fwdMx','../signalproc/sourceloc/rsrc/fwdMx','fwdDi',[],'srcDepth',[],...
            'fwdMxfids',[],'fids','fids',...
            'subIdx',[],...
            'method','lcmv','lambda',0,'nai',1,'maxPowOri',1,'verb',0);
[opts]=parseOpts(opts,varargin);

% channel dim
dim=n2d(z.di,opts.dim);
% extract the data we want
X=z.X;
if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
  idx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
  X=X(idx{:});
  Xidx=idx{dim(1)};
else
  Xidx=1:size(X,dim(1));
end;
szX=size(X); nd=ndims(X);

% Get the forward matrix (and its meta-info) 
mfiled=fileparts(mfilename('fullpath')); % get dir where this file is, for relative paths
if( ischar(opts.fwdMx) )
   fn = opts.fwdMx; 
   if ( ~isequal(opts.fwdMx(max(1,end-4):end),'.mat') ) fn=[fn '.mat']; end;
   if ( exist(fn,'file') )                      fwdMx = load(opts.fwdMx); 
   elseif ( exist(fullfile(mfiled,fn),'file') ) fwdMx = load(fullfile(mfiled,fn)); 
   else error('couldnt find the fwdMx file');
   end;
   A  =fwdMx.X;
   Adi=fwdMx.di;
elseif ( isstruct(opts.fwdMx) )
   A  =opts.fwdMx.X;
   Adi=opts.fwdMx.di;
elseif ( isnumeric(opts.fwdMx) ) 
   A = opts.fwdMx;
   if ( ~isempty(opts.fwdDi) )
      if( ischar(opts.fwdDi) ) Adi=load(opts.fwdDi);
      else                    Adi=opts.fwdDi;
      end
   else % no Di so make it up
      Adi=mkDimInfo(size(A),'ch',[],[],'ori',[],[],'ch_src',[],[]);
   end
end

% remove too deep sources, if wanted
if ( ~isempty(opts.srcDepth) && ~isinf(opts.srcDepth) )
   if ( isfield(Adi(n2d(Adi,'ch_src')).extra,'d2tri') )
      keep= [Adi(n2d(Adi,'ch_src')).extra.d2tri]<(opts.srcDepth.^2);
      [idx,Adi]=subsrefDimInfo(Adi,'dim','ch_src','idx',keep);
      A=A(idx{:});
   else
      warning('Dont have source depth info, cant remove too deep sources');
   end
end

% get the positions of the fwdMx electrodes and the data's electrodes
electPos=[z.di(n2d(z.di,'ch')).extra(Xidx).pos3d]; % 3d pos of the electrodes
fwdMxPos=[Adi(1).extra.pos3d];               % 3d pos of the fwdMx dests
fwdMxTri=[Adi(1).info.tri];

% map the electrode positions into the fwdMx positions space
% 1st use the fids to align the 2 sets of points
fwdMxfids = Adi(1).info.fids;
if ( ~isempty(opts.fids) )
   if( isnumeric(opts.fids) ) electfids = opts.fids;
   elseif(ischar(opts.fids) )  [ans ans ans electfids]=readCapInf(opts.fids);
   else error('electfids not useable');
   end      
elseif ( isfield(z.di(n2d(z.di,'ch')).info,'fids') ) 
   electfids=z.di(n2d(z.di,'ch')).info.fids; 
else
   electfids=fwdMxfids; % assume in the same space
end
[R t]=rigidAlign(electfids,fwdMxfids);
electPos = repop(R*electPos,'+',t); % electrods mapped into fwdMx space

% check the registeration
% clf;trisurf(fwdMxTri',fwdMxPos(1,:),fwdMxPos(2,:),fwdMxPos(3,:));hold on;scatPlot(electPos,'g.','markersize',20); %hold on; scatPlot(fwdMxPos,'k.','markersize',40);


% project the data electrodes onto the fwdMx vertices & re-compute the A appropriately
W = zeros(size(fwdMxPos,2),size(electPos,2),'single'); % map from fwdMx -> electPos
mapped=false; % mapped fwdMx to this set of electrodes
if ( iscell(Adi(1).vals) && ischar(Adi(1).vals{1}) )
   [ans,miz,miA]=intersect(lower(z.di(n2d(z.di,'ch')).vals),lower(Adi(1).vals)); % check if names intersect
   if ( numel(miz)==size(X,dim(1)) ) % names match
      W(sub2ind(size(W),miA,1:size(W,2)))=1;
      mapped=true;
   end
end
if ( ~mapped ) % check if the positions intersect
   d2 = repop(sum(electPos.^2)','+',repop(sum(fwdMxPos.^2),'-',2*electPos'*fwdMxPos)); % dis btw bits
   [miz,miA]= find( abs(sqrt(d2))<eps ); % close enough
   if ( numel(miz)==size(X,dim(1)) ) % matched
      W(sub2ind(size(W),miA,1:size(W,2)))=1; mapped=true; 
   end
end;
if ( ~mapped ) % do a full triangle interpolation to map to the fwdMx space
   % Now project the electrode positions onto the 
   fwdMxTri=Adi(1).info.tri;
   [lambda1,lambda2,Pnorm,d2tri]=pntTriDis(electPos,fwdMxPos,fwdMxTri); % get pnts in tri coords
   sd2tri=(Pnorm.^2+d2tri); [mind2tri,mind2Trii]=min(sd2tri,[],2); % get the closest tri
   ii=sub2ind(size(lambda1),(1:size(lambda1,1))',mind2Trii);       % get dir index into lambda
   electTriPos = cat(1,mind2Trii',lambda1(ii)',lambda2(ii)');      % tri-pos of the electrodes [3 x nElect]
   % build the interpolation matrix
   W(sub2ind(size(W),fwdMxTri(2,electTriPos(1,:)),1:size(W,2)))=electTriPos(2,:);
   W(sub2ind(size(W),fwdMxTri(3,electTriPos(1,:)),1:size(W,2)))=electTriPos(3,:);
   W(sub2ind(size(W),fwdMxTri(1,electTriPos(1,:)),1:size(W,2)))=1-electTriPos(2,:)-electTriPos(3,:);
   clear lambda1 lambda2 Pnorm d2tri sd2tri
end
% apply the interpolation to the fwdMx & its vertex positions
A = tprod(A,[-1 2 3],W,[-1 1],'n');
ofwdMxPos=fwdMxPos; 
fwdMxPos = fwdMxPos*W;
if( isempty(opts.idx) ) Adi(1)=z.di(1); else [ans,Adi(1)]=subsrefDimInfo(z.di(1),'dim',dim,'idx',opts.idx); end;
[Adi(1).extra.pos3d]=num2csl(fwdMxPos); % update meta info

% check the projection
% clf;trisurf(fwdMxTri',ofwdMxPos(1,:),ofwdMxPos(2,:),ofwdMxPos(3,:));hold on;scatPlot(electPos,'g.','markersize',20); hold on; scatPlot(fwdMxPos,'k.','markersize',40);

% compute whole data covariance, over dim(2:end)
idx=-(1:ndims(z.X)); idx(dim(1))=1; idx2=idx;  idx2(dim(1))=2; 
if ( numel(dim)>1 ) % deal with per_xxx computation
   idx(dim(2:end))=dim(2:end); idx2(dim(2:end))=dim(2:end);
   if(any(dim(2:end)<=dim(1)+1)) error('per-x dim-order problem'); end;
end   
Cxx = tprod(X,idx,[],idx2,'n');

% compute the inverse matrix
switch (opts.method);
 case 'lcmv';  W = lcmv(A,Cxx,opts.lambda);
 otherwise; error('Unsupported method: %s',opts.method); 
end
if( opts.nai ) % convert to nerual-activity index
   % normalise the filters to get the neural-activity-index, i.e. give each source equal power
   W=repop(W,'./',tprod(W,[-1 -2 3:ndims(W)],[],[-1 -2 3:ndims(W)],'n'));
end
if( opts.maxPowOri ) % only use the max-power direction for each voxel
   [W,ori]=maxPowDir(W,Cxx); % remove the ori
   [Adi(3).extra.ori]=num2csl(ori); % store the orientation info
   % plot the orientations on each voxel location -- the spiky hedgehog!
   % clf; srcPos=[Adi(3).extra.pos3d]; scatPlot(srcPos,'b.','markersize',10); hold on;for i=1:size(srcPos,2); scatPlot([srcPos(:,i) srcPos(:,i)+ori(:,i)*10],'g-','linewidth',2);end;
end

clear fwdMx; % free up some ram

% apply this set of filters to the data
Xidx=1:ndims(X); Widx=1:ndims(W);
Xidx(dim(1))=-dim(1);  Widx(1)=-dim(1); % dim1 is accumulated
Widx(2:3)=dim(1)+[0 1];                 % & replaced by W_{ori,src}
Xidx(dim(1)+1:end)=Xidx(dim(1)+1:end)+1;% rest of X's dims are shifted down by 1
Widx(4:end)=dim(2:end)+1;               % and per_xxx dims matched with correct inverse solution
z.X = tprod(X,Xidx,W,Widx,'n');         % apply filter to the data

odi=z.di;
z.di =z.di([1:dim(1) dim(1):end]);
z.di(dim(1))   = mkDimInfo(size(z.X,dim(1)),1,'ori',[],[]);       % orientations
if( size(X,dim(1))==3  && ~opts.maxPowDir ) z.di(dim(1)).vals={'x','y','z'}; end;
z.di(dim(1)+1) = Adi(3); %mkDimInfo(size(z.X,dim(1)+1),1,'ch_src',[],[]);  % sources

summary=sprintf('%s->%s %ss',dispDimInfo(odi([dim end])),...
                sprintf('[%d %ss x %d %ss]',numel(z.di(dim(1)).vals),z.di(dim(1)).name,...
                        numel(z.di(dim(1)+1).vals),z.di(dim(1)+1).name));
info=struct('W',W,'Cxx',Cxx,'odi',odi(dim));
info.testFn={'jf_linMapDim' 'mx' W 'di' Adi 'Warning:notRight'};
if( opts.maxPowOri ) info.ori=ori; end;
z =jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
z=jf_load('eeg/vgrid/nips2007/1-rect230ms','jh','flip_rc_sep');
z.di(1)=addPosInfo(z.di(1),'easycap_74_xyz');
s=jf_invSrcLoc(z,'idx',[z.di(1).extra.iseeg],'fids','easycap_74_fids');


fwdMx=load('~/source/matfiles/signalproc/sourceloc/temp/fwdMx_cap64');
fwdMx=jf_retain(fwdMx,'dim','ch_src','idx',[fwdMx.di(3).extra.d2tri]<15.^2); % restrict to near cortical sources
s=jf_invSrcLoc(z,'idx',[z.di(1).extra.iseeg],'fids','easycap_74_fids','fwdMx',fwdMx);

