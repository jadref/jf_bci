function [z]=jf_ERP(z,varargin);
% compute class-dependent ERPs
%
% Options:
%   dim -- [str] name of the dimension of z.X which contains the trails
%   Y   -- [size(z.X,dim) x 1] or [size(z.X,dim) x nSubProb] labeling to 
%           use for the plotting                                             (z.Y)
%   Ydi -- [di-struct ndims(Y)+1 x 1] dimension info for the Y               (z.Ydi)
%          OR
%          {str} list of dimension names which contain the trials            ('epoch')
%   method -- how to compute the mean: either 'mean','robust'                ('mean')
opts=struct('dim',[],'Y',[],'Ydi',[],'method','mean','zeroLab',0,'verb',0);
opts=parseOpts(opts,varargin);
Ydi=opts.Ydi; if ( isempty(Ydi) ) if( isfield(z,'Ydi') ) Ydi=z.Ydi; elseif(~isempty(opts.dim)) Ydi=opts.dim; end; end;
Y  =opts.Y;   if ( isempty(Y) && isfield(z,'Y') )     Y  =z.Y;   end;
if ( isempty(Y) && ~isempty(Ydi) ) % no labels -> do global ERP  
  if ( ~isstruct(Ydi) ) dim=n2d(z,Ydi); else dim=n2d(z,{Ydi.name}); end;
  Y=ones(size(z.X,n2d(z.di,dim)),1); Ydi=z.di(dim);
end;
if ( ~isnumeric(Y) ) Y=single(Y); end;
keyY=unique(Y(:));  % convert to spMx if not already done
if( ~isempty(Y) && ( numel(keyY)>3 || ~( isequal(keyY(:),[-1 1]') || isequal(keyY,[-1 0 1]') || isequal(keyY,[0 1]')) ...
    || (numel(Ydi)==ndims(Y)) || ~n2d(Ydi,'subProb',0,0) ) ) 
  if ( isstruct(Ydi) ) Ydi={Ydi(1:min(end,ndims(Y))).name}; end;
   oY=Y;
  [Y,Ydi]=addClassInfo(Y,'Ydi',Ydi,'zeroLab',opts.zeroLab); 
end;
% convert to proper di
if ( ischar(Ydi) ) Ydi={Ydi}; end;
if ( iscell(Ydi) ) 
   szY=size(Y); if(numel(szY)==2 && szY(2)==1) Ydi{2}='subProb'; end; Ydi = mkDimInfo(szY,Ydi); 
end
if ( isstruct(Ydi) && ~isempty(Y) )
   wdi  = Ydi;
   spD=n2d(Ydi,'subProb');
  % each unique sp encoding is roughly 1 class....
  if ( isfield(Ydi(spD).info,'label')) 
    spKey =Ydi(spD).info.label;% and labels
  elseif ( isfield(Ydi(spD).info,'spKey') )
    spKey =Ydi(spD).info.spKey;% use the marker number
  else
    spKey = 1:size(Y,spD);
  end

  szY=size(Y); szY(end+1:spD)=1;
  [Yu,ans,idx] = unique(reshape(Y,[prod(szY(1:spD-1)),szY(spD)]),'rows');
  idxKey=1:size(Yu,1);
  zIdx=find(all(Yu==0,2));
  Yu(zIdx,:)=[]; idxKey(zIdx)=[];% remove all 0's rows
  Yu=Yu'; % [nSp x nCls]
  if ( isfield(Ydi(spD).info,'spMx') ) % use spMx to compute mapping to labels
     spMx=Ydi(spD).info.spMx;
     if ( ndims(spMx)>2 ) spMx=reshape(spMx,[],size(spMx,ndims(spMx))); end;
     clsi=false(size(Yu,2),size(spMx,2));
     for ci=1:size(Yu,2);
        clsi(ci,:)=Yu(:,ci)'*spMx./(Yu(:,ci)'*Yu(:,ci))>.999999;
     end
  else % order by 1st occurance of a +label in left right order
     clsi=false(size(Yu,2),size(Yu,1)+1);
     for ci=1:size(Yu,2); 
        tmp=find(Yu(:,ci)>0,1); if(isempty(tmp))tmp=size(Yu,1)+1; end;
        clsi(ci,tmp) = true;
     end
  end
  % re-build the spKey to match the found groups of labels     
  for ci=1:size(Yu,2);
     if ( sum(clsi(ci,:))>0 )
        cli=find(clsi(ci,:));
        if ( iscell(spKey) )
           nspKey{ci}=sprintf('%s',spKey{cli(1)});
           if ( numel(cli)>1 )  nspKey{ci}=[nspKey{ci} sprintf('+%s',spKey{cli(2:end)})]; end;
        elseif( isnumeric(spKey) )
           nspKey{ci}=sprintf('%d+',spKey(min(end,cli(1:end))));
        end
     else
        nspKey{ci}='help!';
     end
  end
  spKey=nspKey;
  oY = Y;
  Y  = -ones([prod(szY(1:spD-1)),size(Yu,2)]);
  for ci=1:size(Y,2); Y(idx==idxKey(ci),ci)=1; end;
  Y  = reshape(Y,[szY(1:spD-1) size(Yu,2)]);
  %wght=double(Y>=mean(sum(abs(spMx),1))); % BODGE! may break with certain spMxs
  wght=double(Y>0);
  wdi(spD).vals=spKey;
  if ( n2d(wdi,'class',0,0) ) % guard against name re-use
     wdi(spD).name='class_2'; 
  end
elseif ( ~isempty(Y) )
   wght = double(Y>0); spKey=1:size(wght,2);
   % deal with binary problem special case
   if (size(Y,max(ndims(Y),2))==1) 
      wght=cat(ndims(Y),wght,double(Y<0)); spKey=[1 -1];
   end 
   wdi  = mkDimInfo(size(wght),'epoch',[],[],'class',[],spKey);
else % just do a std ERP
  if ( isempty(Ydi) )     dim=n2d(z,'epoch'); % epochs if nowt else
  elseif ( iscell(Ydi) )  dim=n2d(z,Ydi); 
  elseif ( isstruct(Ydi)) dim=n2d(z,{Ydi.name});
  end;
  wght = ones(size(z.X,dim),1);
  wdi  = mkDimInfo(size(wght),z.di(dim).name,[],[],'class',[],[]);
end
wght = repop(wght,'./',msum(wght,1:ndims(wght)-1));
%wdi(end-1) = mkDimInfo([],1,'class',[],Ydi(end).info.label(Ydi(end).info.marker));
switch ( opts.method ) 
 case {'mean','ave'};       z = jf_linMapDim(z,'mx',wght,'di',wdi,'summary','ERP');
 case {'median','robust'};  
  dim=n2d(z,{wdi(1:end-2).name});
  if ( numel(dim)~=1 ) error('only 1-d for now'); end;
  szX=size(z.X); nSp=size(wght,ndims(wght)); 
  mu=zeros([szX(1:dim-1) nSp szX(dim+1:end)]);
  idx={}; for d=1:ndims(mu); idx{d}=1:size(mu,d); end;
  for spi=1:nSp;
    idx{dim}=find(wght(:,spi)~=0);
    mu(idx{1:dim-1},spi,idx{dim+1:end})=median(z.X(idx{:}),dim);
  end
  z.X=mu; z.di(dim)=wdi(2);
  otherwise; error('Unrec erp mode');
end
% remove invalid labeling info 
if ( isfield(z,'Y') ) z.Y=[];  end;
if ( isfield(z,'Ydi') ) z.Ydi=[]; end;
if ( isfield(z,'foldIdxs') ) z.foldIdxs=[]; end;

return;
