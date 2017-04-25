function [z]=jf_ica(z,varargin);
% Independent Components analysis method
%
% Options:
%  dim    -- [str] dimension to work along ('ch')
%  method -- [str] ica-method to use one of {fastica,infomax,tdsep} (tdsep)
%  nfilt  -- [1x1] number of ica components to keep (inf)
%  nonLinearity -- [str] for fastICA which non-linearity to use. (pow3)
%  whitening    -- [bool] do we whiten the data before starting. (true)
%  tds    -- [1x1] for tdsep the number of temporal factors to use (5)
%  verb   -- [1x1] verbosity level
%  subIdx -- {idx} subindicies of X to use for training the ICA
opts=struct('dim','ch','method','infomax','subIdx',[]);
icaOpts=struct('nfilt',[],'nonLinearity','pow3','maxIter',[],...
               'whitening',1,'verb',0,'tds',5,'bias',0);
[opts,icaOpts]=parseOpts({opts,icaOpts},varargin);

% Extract the dimensions to work along
if ( iscell(opts.dim) || ischar(opts.dim) ) % convert name to dim
   if ( ischar(opts.dim) ) opts.dim={opts.dim}; end;
   for i=1:numel(opts.dim)  dim(i)=strmatch(opts.dim{i},{z.di.name}); end;
elseif ( isnumeric(opts.dim) )
   dim=opts.dim;
end
dim(dim<0)=dim(dim<0)+ndims(z.X)+1; % convert neg dim specs

if ( isempty(icaOpts.nfilt) ) icaOpts.nfilt = size(z.X,dim); end;

% setup X as [nCh x ....] so can use X(:,:) to get 2-d rep
Xtrn = z.X;
if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data  
  idx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
  Xtrn=Xtrn(idx{:});
end;
if ( dim~=1 ) % permute the ch dim to the front and use it.
   Xtrn=permute(Xtrn,[dim 1:dim-1 dim+1:ndims(Xtrn)]);
end

info=struct();
switch ( lower(opts.method) ) 
 case {'fast','fastica'};
  %error('dont know how to make this one work?');
  verbStr={'off','on'};                       % map verb level -> verb str
  dispStr={'off','signals','basis','filters'};% map verb level -> disp str
  if ( icaOpts.whitening ) phases='all'; else phases='pca'; end;
  if ( isempty(icaOpts.maxIter) ) icaOpts.maxIter=1000; end;
  [mix,sf]=fastica(Xtrn(:,:),'g',icaOpts.nonLinearity,...
                   'only',phases,...
                   'maxNumIterations',icaOpts.maxIter,...
                   'numOfIC',icaOpts.nfilt,...
                   'verbose',verbStr{min(end,max(1,icaOpts.verb))},...
                   'displayMode',verbStr{min(end,max(1,icaOpts.verb))});
  sf=sf'; mix=mix';
  info.mix=mix;
  
 case {'infomax','infomaxica'};
  verbStr={'off','on'};     % map verb level -> verb str
  spherStr={'off','on'};    % map spher flat -> str
  biasStr={'off','on'};     % bias correct, i.e. center over time?
  if ( isempty(icaOpts.maxIter) ) icaOpts.maxIter=512; end;
  [R,W,compvar,bias]=...
      runica(Xtrn(:,:),'ncomps',icaOpts.nfilt,...
             'sphering',spherStr{icaOpts.whitening+1},...
             'maxsteps',icaOpts.maxIter,'bias',biasStr{min(end,max(1,icaOpts.verb))},...
             'verbose',verbStr{min(end,max(1,icaOpts.verb))});
  sf=(R*W)'; mix=pinv(sf);
  info.R=R; info.W=W; info.mix=mix; info.compvar=compvar;
  
 case {'tdsep','tdsepica'};
  if ( isscalar(icaOpts.tds) ) tds=0:icaOpts.tds-1; else tds=icaOpts.tds; end;
  [mix,D]=tdsep2(Xtrn(:,:),tds); 
  mix=mix'; sf=pinv(mix);
    
 otherwise; error('Unrecognised ica type: %s',opts.method); 
end
 
 
% Construct a spatial filter dimInfo structure.
sfDi = mkDimInfo(size(sf),...
                 z.di(dim).name,z.di(dim).units,z.di(dim).vals,...
                 'ch_ica',[],1:size(sf,2));
sfDi(1).extra = z.di(dim).extra; %save the extra info

% apply the filter, i.e. map to ica space
sfX   = tprod(z.X,[1:dim-1 -dim dim+1:ndims(z.X)],sf,[-dim 1]); 

% record the result
z.X = sfX;   
z.di(dim).name='ch_ica';
z.di(dim).vals=1:size(sfX,1);
z.di(dim).extra=repmat(struct(),1,size(sfX,1));

summary=sprintf('over %s, %d ica components',z.di(dim).name,size(sf,2));
info.sf=sf; info.sfDi=sfDi;%=struct('sf',sf,'sfDi',sfDi);
info.testFn={'jf_linMapDim' 'mx' sf 'di' sfDi};
z =jf_addprep(z,mfilename,summary,mergeStruct(opts,icaOpts),info);

return;
%------------------------------------------------------------------------
function testCase();
% make a typical ica toy problem

sources = { {'sin' 15};
            {'saw' 18};
            {'square',20};
            {'coloredNoise' 1};
            {'coloredNoise' 1./(1:50)};
            {'none' 1}
            }
d=8;
elect_loc = [+.5 .2; -.5 .2; +.3 .2; -.3 .2; 0 .2;...
             cos(linspace(0,pi,d))' sin(linspace(0,pi,d))']';
N=100; T=300;
mix=ones(N,1);
[X,A,S,elect_loc]=mksfToy(sources,mix,T,elect_loc);
z=jf_mksfToy(mix,'y2mix',mix,'fs',1,'N',N,'nCh',10,'nSamp',T,...
             'sources',sources,'pos2d',elect_loc);


% make a spatial filtering toy problem
mix=cat(3,[1 2;.5 2],[.5 2;1 2],[0 0;0 0]); % power switch with label + noise
Y=ceil(rand(N,1)*L); oY         = lab2ind(Y);   % True labels
z=jf_mksfToy(Y,'y2mix',mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'period',[fs/16;fs/24],'phaseStd',pi/2);

c=jf_ica(z);
clf;image3ddi(c.X,c.di,1,'disptype','imaget','clim','minmax');

muX = cat(3,mean(c.X(:,:,oY(:,1)>0),3),mean(c.X(:,:,oY(:,2)>0),3));
clf;image3ddi(muX,c.di,1,'disptype','plot','clim',[])

A=dv2auc(oY(:,1),c.X,strmatch('epoch',{w.di.name}));
clf;image3ddi(A,c.di,1,'disptype','plot','clim',[0 1])

clf;jplot([c.prep(end).info.sfDi(1).extra.pos2d],c.prep(end).info.sf(:,:));