function [z]=jf_artChRm(z,varargin);
% remove any signal correlated with the input signals from the data
%
% Options:
%  dim -- dim(1) = the dimension along which to correlate/deflate
%         dim(2) = the time dimension for spectral filtering/detrending along
%  idx -- the index/indicies along this dimension to use as artifact channels
%  vals-- the value of the index/indicies along this dimension to use as artifact channels
%  bands   -- spectral filter (as for fftfilter) to apply to the artifact signal ([])
%  detrend -- detrend the artifact before removal                        (0)
%  center  -- center in time (0-mean) the artifact signal before removal (0)
opts  =struct('method','artChRegress');
rmopts=struct('detrend',0,'center',0,'bands',[],'fs',[]);
subsrefOpts=struct('dim',{{'ch' 'time' 'epoch'}},'vals',[],'idx',[],'range',[],'mode','retain','valmatch','exact');
ignoreOpts=struct('subIdx',[]);
[opts,subsrefOpts,ignoreOpts,varargin]=parseOpts({opts,subsrefOpts,ignoreOpts},varargin);

dim=n2d(z,subsrefOpts.dim); dim(dim==0)=[];
if ( numel(dim)<2 && (opts.detrend || opts.center || ~isempty(opts.bands)) )
  dim(2) = n2d(z,'time'); % assume time dim
end

% compute the artifact signal and its forward propogation to the other channels
if ( isempty(rmopts.fs) && ~isempty(rmopts.bands) ) rmopts.fs=getSampRate(z); end

% get the indices we will use for the artifact channel
[idx]=subsrefDimInfo(z.di,'dim',dim(1),subsrefOpts);

% call the artChRm function to do the actual work
if(strcmpi(opts.method,'artchrm'))
  [z.X,info] = artChRm(z.X,dim,idx{dim(1)},rmopts,varargin{:});
elseif ( strcmpi(opts.method,'artchregress') )
  [z.X,info] = artChRegress(z.X,dim,idx{dim(1)},rmopts,varargin{:});
else
  error('Unrecognised removal method: %s',opts.method);
end

summary = ['over ' sprintf('%s',z.di(dim(1)).name)];
if ( islogical(idx{dim(1)}) ) idx{dim(1)}=find(idx{dim(1)});end;
if ( numel(idx{dim(1)})<5 ) 
   summary=sprintf('%s (%s)',summary,vec2str(z.di(dim(1)).vals(idx{dim(1)}),',')); 
else
   summary=sprintf('%d %s',numel(idx{dim(1)}),summary);
end
if ( numel(dim)>1 ) summary=sprintf('%s x %s',summary,sprintf('%s ',z.di(dim(2:end)).name)); end
z=jf_addprep(z,mfilename,summary,mergeStruct(rmopts,subsrefOpts),info);
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_mksfToy();
d=jf_deflate(z,'dim','time','mx',shiftdim(z.X(2,:,1)));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','plot','legend','ne');clickplots

d=jf_deflate(z,'dim',{'time','epoch'},'mx',shiftdim(z.X(2,:,:)));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','plot','legend','ne');clickplots

d2=jf_artChRm(z,'vals','2');
clf;image3ddi(d2.X(:,:,1),d2.di,1,'disptype','plot','legend','ne');clickplots

d=jf_deflate(z,'dim',{'time','epoch'},'mx',permute(z.X(1:2,:,:),[2 3 1]));
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','plot','legend','ne');clickplots

di=z.di; di(1).name='def_dir';
d=jf_deflate(z,'dim',{'time','epoch'},'mx',z.X([1:2],:,:),'di',di);
clf;image3ddi(d.X(:,:,1),d.di,1,'disptype','plot','legend','ne');clickplots


% test
% plot artifact channel before removal
clf;for i=1:10:size(z.X,n2d(z,'epoch')); clf;jf_plot(jf_reref(jf_retain(jf_retain(z,'dim','ch','vals','%EOG*%'),'dim','epoch','idx',i+(0:9)),'dim','time'),'disptype','plot');waitkey; end
% plot data before removal
clf;for i=1:10:size(z.X,n2d(z,'epoch')); clf;jf_plot(jf_retain(z,'dim','epoch','idx',i+(0:9)),'disptype','plot');waitkey; end

a=jf_artChRm(z,'dim','ch','vals','%EOG*%','bands',[.1 .5 6 8]);
% plot artSig used to decorrelate
clf;for i=1:10:size(z.X,n2d(z,'epoch')); clf;image3d(a.prep(end).info.artSig(:,:,i+(0:9)),1,'Yvals',a.di(2).vals,'disptype','plot');waitkey; end
% plot left over signal in the artifact channels
clf;for i=1:10:size(z.X,n2d(z,'epoch')); clf;jf_plot(jf_reref(jf_retain(jf_retain(a,'dim','ch','vals','%EOG*%'),'dim','epoch','idx',i+(0:9)),'dim','time'),'disptype','plot');waitkey; end
% plot data after removal
clf;for i=1:10:size(z.X,n2d(z,'epoch')); clf;jf_plot(jf_retain(a,'dim','epoch','idx',i+(0:9)),'disptype','plot');waitkey; end

