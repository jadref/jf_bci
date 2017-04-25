function [z]=jf_compressDims(z,varargin);
% compress 2 separate dimensions of z into a single new one
% Options:
%  dim     -- compress entries from dim(1) into dim(2)
%  relabel -- do we generate new unique Y labels (0)
%  dimname  -- [str] new name to give the new combined dimension (dim(1).label '_' dim(2).label)
%  summary -- additional summary info
opts=struct('dim',[],'dimname',[],'relabel',0,'summary','','subIdx',[],'verb',0);
[opts]=parseOpts(opts,varargin);

dim=[]; % get the dim to work along
dim=n2d(z,opts.dim);
% [ans,si]=sort(dim,'ascend');
% if ( ~isequal(si(:),1:numel(si)) )
%    dim=dim(si);
% end
if ( isempty(dim) ) 
  dim=find(size(z.X)==1,1); % first singlention dimension
  if ( dim==1 ) dim(2)=2; else dim(2)=dim(1)-1; end;
end

% Do the compression
% 1 -- permute so it works
if ( ~all(abs(diff(sort(dim,'ascend')))==1) ) 
   [md,mi]=min(dim); permd=sort(dim([1:md-1 md+1:end]),'ascend');
   z=jf_permute(z,'dim',[1:md permd setdiff(md+1:ndims(z.X),permd)]);
   dim=md+(0:numel(dim)-1); % new set of dims to compress
   warning('permuted dim order before compress!'); 
end

% 2 -- merge dims
% update the data
sz=size(z.X); 
z.X=reshape(z.X,[sz(1:min(dim)-1) prod(sz(dim)) sz(max(dim)+1:end) 1]);

% update the di
ndi = z.di(dim(1));
dimname=opts.dimname; 
if ( isempty(dimname) ) 
   dimname=sprintf('%s%s',z.di(dim(1)).name,sprintf('_%s',z.di(dim(2:end)).name)); 
end;
ndi.name=dimname;
ndi.vals=repmat(ndi.vals',[1,sz(dim(2:end))]); % rep down
if( isnumeric(ndi.vals) ) 
   ndi.vals=repop(single(ndi.vals),'+',(1:sz(dim(1)))'*.1);
elseif ( iscell(ndi.vals) && ischar(ndi.vals{1}) ) 
   for i=1:size(ndi.vals,1); for j=1:size(ndi.vals,2);
         if ( iscell(ndi.vals) ) str=ndi.vals{i,j};
         elseif ( isnumeric(ndi.vals) ) str=sprintf('%g',ndi.vals(i,j));
         else str=disp(ndi.vals(i,j));
         end
         if ( iscell(z.di(dim(2)).vals) ) str=[str '_' z.di(dim(2)).vals{j}];
         elseif( isnumeric(z.di(dim(2)).vals) ) 
            str=[str '_' sprintf('%g',z.di(dim(2)).vals(j))];
         else str=[str '_' disp(ndi.vals{i,j})];
         end
         ndi.vals{i,j}=str;
      end
   end
end
if ( ~isempty(ndi.extra) ) 
	ndi.extra=ndi.extra(:); 
	extra=ndi.extra; for i=2:prod(sz(dim(2:end))); ndi.extra=cat(2,ndi.extra,extra); end;
	for i=1:size(ndi.extra,1); 
     for j=1:size(ndi.extra(:,:),2);
       if ( iscell(z.di(dim(2)).vals) )
         ndi.extra(i,j).(z.di(dim(2)).name)=z.di(dim(2)).vals{j};
       else
         ndi.extra(i,j).(z.di(dim(2)).name)=z.di(dim(2)).vals(j);
       end
       %if ( ~isfield(ndi.extra,'src') || isempty(ndi.extra(i,j).src) ); ndi.extra(i,j).src=j;
       %else ndi.extra(i,j).src=j+ndi.extra(i,j).src./10;
       %end
     end	
	end
	if ( dim(1)>dim(2) ); ndi.extra=ndi.extra'; end; % transpose if compressed into later dim
end
ndi.vals=ndi.vals(:)';
ndi.extra=ndi.extra(:)';

odi = z.di(dim); % save the removed value
% remove the compressed dim
ii=1:numel(z.di); ii(dim(2:end))=[];
z.di=z.di(ii); z.di(min(dim))=ndi;

if ( isfield(z,'Ydi') )
   oYdi=z.Ydi;
   Ydim=n2d(z.Ydi,{odi.name},0,0);
   if ( all(Ydim==0) ) % no overlap, do nothing 
   elseif(sum(Ydim>0)==numel(dim)) % just squahsed Y's dims, need to reshape
      szY=size(z.Y);
      Ydim(Ydim==0)=[];
		Ykeep = n2d(z.Ydi,odi(1).name); if ( Ykeep==0 ); Ykeep=min(Ydim); end; Ydim=sort(Ydim,'ascend');
      z.Y=reshape(z.Y,[szY(1:min(Ydim)-1) prod(szY(Ydim)) szY(max(Ydim)+1:end) 1]);
      if ( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) ) 
         szfI=size(z.foldIdxs);
			if ( prod(szY(Ydim))>prod(szfI(Ydim)) ) % need to replicate singlentons first
				z.foldIdxs=repmat(z.foldIdxs,[ones(1,min(Ydim)-1) szY(Ydim)./szfI(Ydim) ones(1,numel(szfI)-max(Ydim))]);
				szfI=size(z.foldIdxs);
			end
         z.foldIdxs=reshape(z.foldIdxs,[szfI(1:min(Ydim)-1) prod(szfI(Ydim)) szfI(max(Ydim)+1:end) 1]);
      end
      if ( isfield(z,'outfIdxs') && ~isempty(z.outfIdxs) ) 
         szfI=size(z.outfIdxs);
			if ( prod(szY(Ydim))>prod(szfI(Ydim)) ) % need to replicate singlentons first
				z.outfIdxs=repmat(z.outfIdxs,[ones(1,min(Ydim)-1) szY(Ydim)./szfI(Ydim) ones(1,numel(szfI)-max(Ydim))]);
				szfI=size(z.outfIdxs);
			end
         z.outfIdxs=reshape(z.outfIdxs,[szfI(1:min(Ydim)-1) prod(szfI(Ydim)) szfI(max(Ydim)+1:end) 1]);
      end
      % update dimInfo
		nYdi  = z.Ydi(Ykeep);
      nYdi.vals = repmat(nYdi.vals',[1,sz(dim(2:end))]); % BODGE:
		% copy the extra info: BODGE
		tmp=nYdi.extra(:); 
		nYdi.extra=tmp;for i=2:prod(sz(dim(2:end))); nYdi.extra=cat(2,nYdi.extra,tmp); end;		
      nYdi.vals = nYdi.vals(:)';
      nYdi.extra= nYdi.extra(:)';
      nYdi.name = ndi.name;
      z.Ydi = [z.Ydi(1:min(Ydim)-1); nYdi; z.Ydi(max(Ydim)+1:end)];
   else % squashed some outside Y into Y, need to repmat
      if ( ~opts.relabel ) % just replicate the existing info to make the new labs
			if ( Ydim(1)>0 ) 
			  szY=size(z.Y);
			  if ( dim(1)>dim(2) ); % added in from preceeding dimension
				 z.Y=repmat(reshape(z.Y,[1,szY]),[prod(sz(dim(2:end))),ones(1,ndims(z.Y))]); % shift right and duplicate
			  else % added in from post dimension
				 z.Y=repmat(reshape(z.Y,[szY(1:Ydim(1)),1,szY(Ydim(1)+1:end)]),[ones(1,Ydim(1)),prod(sz(dim(2:end))),ones(1,ndims(z.Y)-Ydim(1))]); % shift left and duplicate
			  end
			  z.Y=reshape(z.Y,[prod(sz(dim)) size(z.Y,3) size(z.Y,4)]); % re-shape to single entry
			  if ( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) )
				 szf=size(z.foldIdxs);
				 if ( dim(1)>dim(2) ); % added in from preceeding dimension
					z.foldIdxs=repmat(reshape(z.foldIdxs,[1,szf]),[prod(sz(dim(2:end))),ones(1,ndims(z.foldIdxs))]); % shift right and duplicate
				 else % added in from post dimension
					z.foldIdxs=repmat(reshape(z.foldIdxs,[szf(1:Ydim(1)),1,szf(Ydim(1)+1:end)]),[ones(1,Ydim(1)),prod(sz(dim(2:end))),ones(1,ndims(z.foldIdxs)-Ydim(1))]); % shift left and duplicate
				 end
				 z.foldIdxs=reshape(z.foldIdxs,[prod(sz(dim)) size(z.foldIdxs,3) size(z.foldIdxs,4)]);
			  end
			  if ( isfield(z,'outfIdxs') && ~isempty(z.outfIdxs) ) 
				 szf=size(z.outfIdxs);
				 if ( dim(1)>dim(2) ); % added in from preceeding dimension
					z.outfIdxs=repmat(reshape(z.outfIdxs,[1,szf]),[prod(sz(dim(2:end))),ones(1,ndims(z.outfIdxs))]); % shift right and duplicate
				 else % added in from post dimension
					z.outfIdxs=repmat(reshape(z.outfIdxs,[szf(1:Ydim(1)),1,szf(Ydim(1)+1:end)]),[ones(1,Ydim(1)),prod(sz(dim(2:end))),ones(1,ndims(z.outfIdxs)-Ydim(1))]); % shift left and duplicate
				 end
				 z.outfIdxs=reshape(z.outfIdxs,[prod(sz(dim)) size(z.outfIdxs,3) size(z.outfIdxs,4)]);
			  end
			  z.Ydi(Ydim(1)).vals = repmat(z.Ydi(Ydim(1)).vals(:),[1,prod(sz(dim(2:end)))]);
			  if ( dim(1)>dim(2) ); % added in from preceeding dimension
				 z.Ydi(Ydim(1)).vals = z.Ydi(Ydim(1)).vals';
			  end
			  z.Ydi(Ydim(1)).vals = z.Ydi(Ydim(1)).vals(:)';
			  if ( isfield(z.Ydi(Ydim(1)),'extra') )
				 tmp=z.Ydi(Ydim(1)).extra; tmp=tmp(:); for ii=1:prod(sz(dim(2:end))); tmp(:,ii)=tmp(:,1); end; z.Ydi(Ydim(1)).extra=tmp;
				 if ( dim(1)>dim(2) ); % added in from preceeding dimension
					z.Ydi(Ydim(1)).extra=z.Ydi(Ydim(1)).extra';
				 end
				 z.Ydi(Ydim(1)).extra = z.Ydi(Ydim(1)).extra(:)';
			  end
			else 
			  warning('compression removed the labels!');
			  z.Y=[]; z.foldIdxs=[]; z.Ydi=[];
			end
      else % compute new labels based upon the merged info
         Yl        = repmat((1:sz(dim(2)))',1,sz(dim(1)));Yl=Yl(:); % new labels
         markerdict= odi(2).vals;
         [z.Y,key]=lab2ind(Yl);
         z.Ydi    = mkDimInfo(size(z.Y),'epoch',[],[],'subProb',[],markerdict);
         [z.Ydi(1).extra.marker]=num2csl(Yl);
         z.Ydi(2).info=struct('marker',key,'label',{markerdict},'spType','1vR','sp',{num2cell(key)});
         if ( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) ) z.foldIdxs = gennFold(Yl,10,'perm',0); end;
      end   
      Ydim(Ydim==0)=[];
      z.Ydi(min(Ydim)).name = ndi.name;
   end
end

info = struct('odi',odi);
summary = sprintf('[%s] into %s',[sprintf('%s+',odi(1:end-1).name) odi(end).name],z.di(min(dim)).name);
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
z = jf_addprep(z,mfilename,summary,opts,info);
return;

%--------------------------------------------------------------------------
function testCase()
fs=128;
mix=cat(3,[1 2;.5 2],[.5 2;1 2],[0 0;0 0]); % power switch with label + noise
Y=ceil(rand(N,1)*L); oY         = lab2ind(Y);   % True labels
z=jf_mksfToy(Y,'y2mix',mix,'fs',fs,'N',size(Y,1),'nCh',10,'nSamp',3*fs,...
             'period',[fs/16;fs/16],'phaseStd',pi/2);

nFeat = 2;
mx = randn(size(z.X,1),2); % raw
zl=jf_linMapDim(z,'dim','ch','mx',mx);

% inc di
mxDi = mkDimInfo(size(mx),'ch',[],[],'nfeat',[],[])
zl=jf_linMapDim(z,'mx',mx,'di',mxDi);

% with sparsity
smx=mx; smx(randn(size(mx))<0)=0; % 50% sparse
zl=jf_linMapDim(z,'mx',smx,'di',mxDi,'sparse',0);
szl=jf_linMapDim(z,'mx',smx,'di',mxDi);
