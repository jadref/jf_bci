function [z]=jf_stimSeq2regressor(z,varargin);
opts=struct('dim',-1,'nFold',[]);
[opts,varargin]=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);

% get the stim-seq
[Y,Ydi,target,targetSeq]=stimSeq2regressor(z,'zeroLab',1,'compressSymbStim',0,varargin{:});

% N.B. Y = [ nSamp x nSeq x nSymb x nStim ]
% limit to the true target for each sequence to get the true-target values
if ( size(Y,2)==1 ) % same stim-seq for every epoch
  Ytgt = Y(:,:,target,:); % -> [ nSamp x 1 x nSeq x nStim ]
  Ytgt = reshape(Ytgt,[size(Ytgt,1) size(Ytgt,3) size(Ytgt,4)]); % [ nSamp x nSeq x nStim ]
  Yditgt= [Ydi(1); z.di(n2d(z,-1)); Ydi(4:end)];
else
  Ytgt = zeros(size(Y,1),size(Y,2),size(Y,4)); % [ nSamp x nSeq x nStim ]
  for ei=1:numel(target);
	 Ytgt(:,ei,:) = squeeze(Y(:,ei,target(ei),:));
  end
  Yditgt = Ydi([1 2 4:end]);
end
 % convert 2 stims = into positive/negative examples
if( n2d(Yditgt,'stim',0,0)>0 && size(Ytgt,n2d(Yditgt,'stim'))==2 )% convert 2 stims = into positive/negative examples
   Ytgt=-Ytgt(:,:,1) + Ytgt(:,:,2); % +1/-1 labels
   Yditgt(n2d(Yditgt,'stim'))=[];
end
										  % add this new label info to z
z=jf_addClassInfo(z,'Y',Ytgt,'Ydi',Yditgt(1:end-1));

% update the folding info if needed
if( ~isempty(opts.nFold) ) 
   z = jf_addFolding(z,'nFold',sprintf('lndo_%s',z.di(dim(end)).name),'foldSize',opts.nFold);
end;
