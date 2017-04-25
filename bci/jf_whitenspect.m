function [z]=jf_whitenspect(z,varargin)
% whiten the spectrum of the input data
opts=struct('dim','time','bands',[],'muSpect',[],'muReg',[],'reg',-1e-2,'smthlen',[],'smthhz',6,...
				'muCh','eegonly','verb',1,'center',1,'detrend',0,'blockIdx',[],'subIdx',[]);
opts=parseOpts(opts,varargin);
if(~isempty(opts.muReg))opts.reg=opts.muReg;end;

dim=n2d(z,opts.dim);
szX=size(z.X); szX(end+1:max(dim))=1;

% get if we should work over a subset of channels
muCh=opts.muCh;
chD=[];
if ( strcmp(opts.muCh,'eegonly') )
   chD=n2d(z,'ch'); 
   muCh=[z.di(chD).extra.iseeg];
   if ( all(muCh) ) muCh=[]; end;
end;

%smooth muSpect
smth=opts.smthlen;
if ( isempty(smth) && ~isempty(opts.smthhz) ) 
   fs   = getSampleRate(z);
   smth = opts.smthhz * 2 / (fs/szX(dim(1))*2) ; 
end;

if ( isempty(opts.blockIdx) ) 
   if ( ~isempty(opts.subIdx) ) warning('subIdx ignored!'); end; % compute trans on sub-set of the data
   [z.X,muSpect,smth]=whitenspect(z.X,dim,smth,opts.center,opts.reg,muCh,chD,opts.verb);

elseif ( ~isempty(opts.blockIdx) )  % whiten in blocks
  blockIdx=getBlockIdx(z,opts.blockIdx);
  % convert to indicator
  if ( size(blockIdx,2)==1 ) blockIdx=gennFold(blockIdx,'llo'); end
  if ( ndims(blockIdx)==3 )
	 if ( size(blockIdx,2)~=1 ) error('Not supported yet!'); end
	 blockIdx=blockIdx(:,:);
  end
  % whiten each of the blocks independently
  for bi=1:size(blockIdx,2);
	 idx=subsrefDimInfo(z.di,'dim','epoch','idx',blockIdx(:,bi)>0); % which subset
	 [Xbi,muSpectbi] = whitenspect(z.X(idx{:}),dim,smth,opts.center,opts.reg,muCh,chD,opts.verb);
	 % BODGE: using cell's is a hack, and will probably break other things later...
	 z.X(idx{:})=Xbi;
	 muSpect{bi}=muSpectbi;
  end
end

% summary info
info= struct('muSpect',muSpect,'smth',smth);
summary = sprintf('over %s',z.di(dim(1)).name);
if ( numel(dim)>1 ) summary=[summary sprintf(' per %s',z.di(dim(2:end)).name)]; end;
z   = jf_addprep(z,mfilename,summary,opts,info);
return;
%---------------------------------------------------------------------
function testCase()

zn=jf_whitenspect(z,'reg',-.1);
figure(1);clf;jf_plot(jf_mean(jf_welchpsd(z,'width_ms',1000),'dim','epoch'),'lineWidth',1);
figure(2);clf;jf_plot(jf_mean(jf_welchpsd(zn,'width_ms',1000),'dim','epoch'),'lineWidth',1);
