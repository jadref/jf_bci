function [blockIdx,epD,blkFound]=getBlockIdx(z,blockIdx)
if( nargin<2 ) blockIdx='block'; end;
blkFound=false; epD=[];
if ( ischar(blockIdx) )
   blockType=blockIdx;
   if ( strcmp(blockType,'fold') )
		blockIdx=z.foldIdxs;
   elseif ( strncmp(blockType,'block',numel('block')) )

      if ( isfield(z,'Ydi') ) % Infer dim from Ydi
         epD = n2d(z.di,{z.Ydi(1:end-1).name},0,0); epD(epD==0)=[]; % get dim to work along
         epD = z.di(epD(end)).name; % BODGE: default to last-matching-dim
      else
         epD = z.di(end-1).name; % default to last dim of X
      end
		if ( isfield(z.di(n2d(z,epD)).extra,'src') ) % source file based blocks
         blockIdx=floor([z.di(n2d(z,epD)).extra.src]);
         blkFound=true;
		else
         blockIdx=ones(numel(z.di(n2d(z,epD)).vals),1);
      end
      blockIdx=blockIdx(:); % ensure is [n x 1]
   end
end
return;
