function z=jf_splitDim(z,varargin)
% split a dimension into equal sized bits (with same label)
%
% Options:
%  dim    -- the dimension to spit                                              ('epoch')
%  seqLen -- the length of the sequence, i.e. how many elements in each new dim ([])
%  di     -- [dimInfo struct] the dimension info for the new dimension          ('seq')
%  matchY -- [bool] do we ensure the Ys are the same for each group of elements (1)
%  seqFold-- [bool] do we enforce sequence based folding?                       (0)
opts=struct('dim','epoch','seqLen',[],'di','seq','matchY',1,'seqFold',0,'subIdx',[],'verb',0);
opts=parseOpts(opts,varargin);
dim=opts.dim; dim=n2d(z.di,dim);

seqLen = opts.seqLen;
if ( opts.matchY ) % sequences should all have same label
  if ( isfield(z,'Ydi') & 0 ) ; % use the assigned labels
    
  elseif( isfield(z,'Y') && size(z.Y,2)==1 ) % extract labels from the dim meta-info
    Yl=z.Y;
  else
    Yl=single([z.di(dim(1)).extra.marker]); Yl=Yl(:);
  end   
   edgeInd =[true;(diff(Yl)~=0);true]; % label transition indicator, inc 1st last
   edgeIdx =find(edgeInd); % linear index
   dedgeIdx=diff(edgeIdx);% distance between label changes
   if ( isempty(seqLen) ) seqLen = median(dedgeIdx); end;
   % add in extra seq finishs for extra long bits
   for i=1:numel(edgeIdx)-1; edgeInd(edgeIdx(i)+seqLen:seqLen:edgeIdx(i+1)-seqLen)=1; end;
   % remove start edgeInd for sequences which are too short
   edgeInd(edgeIdx(dedgeIdx<seqLen))=0; edgeIdx(dedgeIdx<seqLen)=[];   

   % re-find the edges which are left
   edgeIdx=find(edgeInd(1:end-1)); % find start seqs of same label
else % just equally spaced bits
   edgeIdx=[1:seqLen:size(z.X,dim(1))]';
end

% construct a list of indicies we want
seqIdx=repop(edgeIdx,'+',(0:seqLen-1))';
% select this subset of z
if( numel(seqIdx)~=size(z.X,dim) ) z = jf_retain(z,'dim',dim,'idx',seqIdx(:)); end;

% re-shape the data
szX=size(z.X);
z.X = reshape(z.X,[szX(1:dim(1)-1) size(seqIdx) szX(dim(1)+1:end)]);
% setup the meta-info
odi = z.di;
z.di= z.di([1:dim(1) dim(1):end]);
z.di(dim(1)).info.vals = reshape(z.di(dim(1)).vals,size(seqIdx));
z.di(dim(1)).info.extra= reshape(z.di(dim(1)).extra,size(seqIdx));
z.di(dim(1)).vals = z.di(dim(1)).vals(seqIdx(:,1));
z.di(dim(1)).extra= z.di(dim(1)).extra(seqIdx(:,1));
% make the meta-info for the new dimension 
di=opts.di; 
if(iscell(di)) z.di(dim(1)).name=di{1}; di=di{2}; end;
if(isempty(di)) di=mkDimInfo(size(seqIdx,2),1,'seq',[],[]);
elseif(ischar(di)) di=mkDimInfo(size(seqIdx,2),1,di,[],[]);
end;
[di.extra.marker]=num2csl(reshape([odi(dim(1)).extra.marker],size(seqIdx)));
z.di(dim(1)+1) = di;

if ( isfield(z,'Ydi') && ~isempty(n2d(z.Ydi,z.di(dim(1)).name,0,0)) ) %reshape Y and Ydi as well
   dimY=n2d(z.Ydi,z.di(dim(1)).name);
   z.Ydi=z.Ydi([1:dimY dimY dimY+1:end]);
   z.Ydi(dimY).vals = z.Ydi(dimY).vals(seqIdx(:,1));
   z.Ydi(dimY).extra= reshape(z.Ydi(dimY).extra,size(seqIdx));
   z.Ydi(dimY+1)    = di;
   szY=size(z.Y); %idx={}; for d=1:ndims(z.Y); idx{d}=1:szY(d); end; idx{dimY}=seqIdx;
   z.Y = reshape(z.Y,[szY(1:dimY-1) size(seqIdx) szY(dimY+1:end)]);
elseif ( isfield(z,'Y') && size(z.Y,1)==szX(dim(1)) )
   szY=size(z.Y); z.Y = reshape(z.Y(seqIdx,:),[size(seqIdx) szY(2:end)]); 
end;
if ( isfield(z,'foldIdxs') && size(z.foldIdxs,1)==szX(dim(1)) )
   szfIdxs=size(z.foldIdxs); z.foldIdxs = reshape(z.foldIdxs,[size(seqIdx) szfIdxs(2:end)]);
   if ( opts.seqFold ) z.foldIdxs=sign(mean(z.foldIdxs,1)); end; % whole seq has fixed folding
end
summary = sprintf('%d %ss -> [%d %ss x %d %ss]',numel(seqIdx),z.di(dim(1)).name,...
                  size(z.X,dim(1)),z.di(dim(1)).name,size(z.X,dim(1)+1),z.di(dim(1)+1).name);
info = struct('seqIdx',seqIdx);
z = jf_addprep(z,mfilename,summary,opts,info);
return;
%--------------------------------------------------------------------------
function testCase()

% make a toy problem to split
z = jf_import('test','test','test',randn(10,100,300),mkDimInfo([10 100 300],'ch',[],[],'time','ms',[],'epoch_letter',[],[]));
Yl = ones(3,1)*sign(randn(100,1))'; % sequences of same labels
[z.Y,z.Ydi]=addClassInfo(Yl(:),'Ydi','epoch_letter');
[z.di(n2d(z.di,'epoch_letter')).extra.marker]=num2csl(Yl(:)); % setup marker stuff
z.foldIdxs=gennFold(Yl(:),10);
oz=z;

z = jf_splitDim(oz,'dim',{'epoch_letter'},'seqLen',3);
z = jf_splitDim(oz,'dim',{'epoch_letter'},'seqLen',2,'di',{'epoch','seq'});
z = jf_splitDim(oz,'dim',{'epoch_letter'},'matchY',1,'di',{'epoch','seq'});
