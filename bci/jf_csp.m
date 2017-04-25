function [z]=jf_csp(z,varargin);
% compute common spatial patterns & return filtered data
%
% [z]=jf_csp(z,varargin);
%
% Options:
%  outputVar -- [bool] return the variance in the filtered direction (1)
%  nfilt     -- [1x1 or 2x1 int] number of filtered components to return (6)
%               [1x1] - this number of filters **for each class**,
%               [2x1] - nfilt(1) from bottom, nfilt(2) from top spectrum
%  subIdx    -- {} subset of the data to use for filter computation
%  log       -- [bool] return log power?                             (true)
%  dim       -- {str} dimensions for trials, and channels  ({'epoch','ch'})
%  cent      -- [bool] center data before comptuations (0)
%  ridge -- [float] size of ridge (as fraction of mean eigenvalue) to add for numerical stability (1e-7)
%  denomType -- [str] type of demoninator to use in the filter computation. 
%           One of: 'All' = , 'Rest', '1v1'
%  singThresh -- [float] threshold to detect singular values in inputs (1e-3)
%  powThresh  -- [float] fractional power threshold to remove directions (1e-4)
% 
% See also: CSP
opts=struct('outputVar',1,'log',1,'nfilt',6,'subIdx',[],'method','csp','verb',0);
cspOpts=struct('dim',{{'epoch','ch'}},'cent',0,'ridge',[],'demonType','all',...
					'singThresh',[],'powThresh',[]);
[opts,cspOpts]=parseOpts({opts,cspOpts},varargin);

% Extract the dimensions to work along
dim=n2d(z,cspOpts.dim,[],[],1);
szX=size(z.X);

if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
  idx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
  cX = z.X(idx{:}); cY=z.Y(idx{n2d(z.di,{z.Ydi(1:end-2).name},0,0)},:);
else
  cX = z.X; cY=z.Y;
end
if ( strcmp(opts.method,'csp') )
  % Call CSP to do the actual work
  [sf,d,Sigmai,Sigmac,SigmaAll]=csp(cX,cY,dim,cspOpts.demonType,cspOpts.ridge,cspOpts.cent,cspOpts.singThresh,cspOpts.powThresh);
elseif ( strcmp(opts.method,'cspfe') ) % just use CSP feature extractor
  [sf,d,Sigmai,Sigmac]=cspfe(cX,cY,dim); Sigmai=[]; SigmaAll=[];
end


% Construct a spatial filter dimInfo structure.
szsf=size(sf);
sfDi = mkDimInfo(szsf(1:2),...
                 z.di(dim(2)).name,z.di(dim(2)).units,z.di(dim(2)).vals,...
                 'ch_csp',[],1:size(sf,2),'class',[],1:size(sf,3));
sfDi=[sfDi(1:end-1); z.di(dim(3:end)); sfDi(end)];
sfDi(1).extra = z.di(dim(2)).extra; %save the extra info
if ( ndims(sf)==2 ) sfDi(3).name=''; end; % no name if only 1 class

% get number of filters to return
nfilt=opts.nfilt; 
% nfilt is total number, get the number per sub-problem
if( numel(nfilt)==1 ) nfilt=ceil(nfilt/size(d,2)); end;
% Extract the sub-set of filters we're going to use
sfAll=[]; 
keep=false(1,size(sf,2),size(sf,3)); % indicator of which coefficients we used
for fi=1:size(d,3); % for each feature
  sffi=[];
  for ci=1:size(d,2); % for each class
   if ( ~isempty(nfilt) )
      % Order the features by eigenvalue
      nSF=sum(d(:,ci)~=0); % d==0 indicates invalid
      
      if( numel(nfilt)==1 ) % #filt to keep, best=largest eigenvalue *for this problem*
        [ans,si]=sort(d(1:nSF,ci),'descend');
        
      elseif ( numel(nfilt)==2 ) % #filt from each end
         si(1:2:nSF)=1:ceil(nSF/2); % re-order to pick from each end
         si(2:2:nSF)=nSF:-1:ceil(nSF/2)+1;   
         [ans,sd]   =sort(d(1:nSF,ci),'descend'); % sort by eigvalue
         si(1:nSF)  =sd(si);
      
      else error('nfilt should be 1 or 2 element array');
      end

      d(1:nSF,ci)=d(si,ci); sf(:,1:nSF,ci)=sf(:,si,ci);% Re-order the features
      nfilt   = min(sum(nfilt),nSF);
   else
      nfilt   = sum(d>0);
   end

   % record the set of all filters we're going to use
   keep(1,1:nfilt,ci,fi)=true;
   sffi = cat(2,sffi,sf(:,1:nfilt,ci)); 
end
sfAll=cat(3,sfAll,sffi);
end

% apply the filter, i.e. map to csp space
idxX  = 1:ndims(z.X); idxX(dim(2))=-dim(2);
idxsf=[-dim(2);dim(2)]; if(numel(dim)>2)idxsf=[idxsf;dim(3:end)]; end;
sfX   = tprod(z.X,idxX,sfAll,idxsf);

if ( ~opts.outputVar ) % record the result
   z.X = sfX;   
   z.di(dim(2)).name='ch_csp';
   z.di(dim(2)).vals=1:size(sfX,1);
   z.di(dim(2)).extra=repmat(struct(),1,size(sfX,1));
else
   % map to feature variances, [sf x epoch]  
   if ( isequal(dim,[3 1]) && size(z.X,2)==size(z.X,1) && ndims(z.X)<=3 ) % covariance input
     sfXXsf=tprod(sfX,[1 -2 2],sfAll,[-2 1]); 
   else % *NOT* covariance input
     szX=size(z.X);
     tpIdx = -(1:ndims(sfX)); tpIdx(dim)=[numel(dim) 1:numel(dim)-1];
     sfXXsf=tprod(sfX,tpIdx,[],tpIdx)./prod(szX(setdiff(1:end,dim))); % ave over removed dims
   end
   z.X   = sfXXsf; % [ nfilt x nFeat x nEp]
   if ( opts.log ) z.X=log(abs(z.X)); end;
   z.di  = z.di([dim(2);dim([3:end 1]);end]);
   z.di(1).name='ch_csp';
   z.di(1).vals=1:size(sfXXsf,1);
   z.di(1).extra=repmat(struct(),1,size(sfXXsf,1));
   if ( ~opts.log ) 
     z.di(end).units='muV.^2';
   else
     z.di(end).units='db';
   end
end

summary=['[' sprintf('%d ',sum(sum(keep,2),4)./size(keep,4)) '] csp'];
if ( ~opts.outputVar ) summary=[summary ' coefficients'];
else summary=[summary ' variances'];
end
info=struct('sf',sf,'sfDi',sfDi,'d',d,'Sigmac',Sigmac,'SigmaAll',SigmaAll,'keep',keep);
if ( ~opts.outputVar )
  info.testFn={'jf_linMapDim' 'mx' sfAll 'di' sfDi};
else
  info.testFn={'jf_sfvar' 'dim' setdiff(1:numel(szX),dim) 'mx' sfAll 'di' sfDi};
end
z =jf_addprep(z,mfilename,summary,mergeStruct(opts,cspOpts),info);

return;
%------------------------------------------------------------------------
function testCase();
% make a spatial filtering toy problem
N=300; L=2; fs=100;
mix=cat(3,[1 2;.5 2],[.5 2;1 2],[0 0;0 0]); % power switch with label + noise
Y=ceil(rand(N,1)*L); oY         = lab2ind(Y);   % True labels
z=jf_mksfToy(Y,'y2mix',mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'period',[fs/16;fs/16],'phaseStd',pi/2);

c=jf_csp(z);
clf;image3ddi(c.X,c.di,1,'disptype','imaget','clim','minmax');

clf;jplot([c.prep(end).info.sfDi(1).extra.pos2d],c.prep(end).info.sf(:,:));

% filter-bank version
s=jf_spectrogram(z,'width_ms',250);
c=jf_csp(s,'dim',{'epoch' 'ch' 'freq'});


% compare wht+csp to csp directly
c=jf_csp(z,'nfilt',8);
sf1=c.prep(end).info.sf(:,c.prep(end).info.keep);
sf1=repop(sf1,'/',sqrt(sum(sf1.^2)));
c2=jf_csp(jf_whiten(z),'nfilt',8,'method','cspfe');
sf2=c2.prep(end).info.sf(:,c2.prep(end).info.keep);
W  =c2.prep(n2d({c2.prep.method},'jf_whiten')).info.W;
sf2=W*sf2;
sf2=repop(sf2,'/',sqrt(sum(sf2.^2)));
clf;imagesc(sf1'*sf2);
