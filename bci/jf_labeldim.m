function [z]=jf_labeldim(z,varargin)
% copy class label information into the specified dim
%
% Options:
%  dim - dim to copy labels into
opts=struct('dim',[],'subsrefOpts',[],'ignoredClass',0,'subIdx',[],'verb',0,'mcSp',1);
opts=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);
if ( numel(dim)>1 ) error('Multi-dims not supported yet'); end;
if ( isfield(z,'Ydi') )
  epD=n2d(z,{z.Ydi(1:end-1).name},1,0); epD(epD==0)=[];
else
  epD=n2d(z,'epoch');
end
szY = size(z.Y);
epDY = sum(dim>epD); % dim index to insert the new labels in Y
z.Y = reshape(z.Y,[szY(1:epDY)       1        szY(epDY+1:end)]); % insert the dim
z.Y = repmat( z.Y,[ones(1,epDY) size(z.X,dim) ones(1,ndims(z.Y)-epDY)]);
if ( ~isempty(opts.subsrefOpts) )
  idx={}; for di=1:ndims(z.Y); idx{di}=1:size(z.Y,di); end;
  [tmpidx]=subsrefDimInfo(z.di(dim),'mode','reject',opts.subsrefOpts{:});   idx(epDY+1)=tmpidx;
  if ( isempty(opts.ignoredClass) || isequal(opts.ignoredClass,0) )
	 z.Y(idx{:})=0; % give these elements 0 label = ignored
  else
	 z.Y(idx{:})=-1; 
	 % give ignored elements the label indicated
	 spD=n2d(z.Ydi,'subProb');
	 idx{spD+(epDY<spD)}=opts.ignoredClass;
	 z.Y(idx{:})=1;
  end
end
if ( isfield(z,'Ydi') )
  z.Ydi = z.Ydi(:); % ensure is col vec
  z.Ydi = [z.Ydi((1:epDY)); z.di(dim); z.Ydi((epDY+1:end))];
  if ( opts.mcSp ) % make a multi-class decoding
	 spD=n2d(z.Ydi,'subProb');
	 if ( isfield(z.Ydi(spD).info,'spMx') )
		spMx= z.Ydi(spD).info.spMx;
		if ( epDY<spD ) 
		  z.Ydi(spD).info.spMx = repmat(shiftdim(spMx,-1),[size(z.X,dim) ones(1,ndims(spMx))]);
		  z.Ydi(spD).info.mcSp = true;
		  z.Ydi(spD).info.spD  = {z.di(dim).name 'subProb'};
		end
	 else
		error('not supported yet');
	 end
  end
end
% copy folding
if( isfield(z,'foldIdxs') && ~isempty(z.foldIdxs) )
  szf = size(z.foldIdxs);
  z.foldIdxs = reshape(z.foldIdxs,[szf(1:epDY)   1            szf(epDY+1:end)]);
  z.foldIdxs = repmat( z.foldIdxs,[ones(1,epDY) size(z.X,dim) ones(1,ndims(z.foldIdxs)-epDY)]);
end;
if( isfield(z,'outfIdxs') && ~isempty(z.outfIdxs) )
  szf = size(z.outfIdxs);
  z.outfIdxs = reshape(z.outfIdxs,[szf(1:epDY)   1            szf(epDY+1:end)]);
  z.outfIdxs = repmat( z.outfIdxs,[ones(1,epDY) size(z.X,dim) ones(1,ndims(z.outfIdxs)-epDY)]);
end;
z =jf_addprep(z,mfilename,sprintf('label %ss',z.di(dim).name),opts,[]);
return;
%---------------------------
function testCase
  z=jf_windowData(z,'dim','time','width_ms',1000,'overlap',.5);

  l=jf_labeldim(z,'dim','window');  jf_disp(l)
  l=jf_labeldim(z,'dim','window','subsrefOpts',{'range','between','vals',[500 1500]});  jf_disp(l)
  
