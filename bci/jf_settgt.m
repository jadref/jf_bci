function [z]=jf_settgt(z,varargin)
opts=struct('dim',{{'time' 'epoch'}},'tgt','emg','interp','none','bands',[],'fs',[],...
				'feature','raw','normalizeTgt',1,'wght',[],'idx',[]);
[opts,varargin]=parseOpts(opts,varargin);
dim=n2d(z,opts.dim);

if ( ~isempty(opts.bands) ) % pre-filter z if needed..
  % filter and downsample both data and target to the same rate
  z     =jf_fftfilter(z, 'dim',dim(1),'bands',opts.bands,'fs',opts.fs);
end

% load the target data
fprintf('Trying: %s as target to predict.\n',lower(opts.tgt));
if ( isstr(opts.tgt) ) 
  if ( exist([opts.tgt '.mat'],'file') ) % try loading the file
	  y = load(opts.tgt);
	  Y  = y.X;
	  Ydi= y.Ydi;
	  if ( ~isempty(opts.bands) ) 
		 y=jf_fftfilter(y,'bands',opts.bands,'fs',opts.fs);
	  end
	  Yfs=getSampleRate(y);

  elseif ( ~isempty(jf_load(z.expt,z.subj,[z.label '_' opts.tgt],z.session,-1,-1)) ) % try loading the pre-proc'd data
	y = jf_load(z,[z.label '_' opts.tgt],-1);
	% BODGE: special code for the opto expt...
	if ( strcmp(opts.tgt,'opto') )
	  y=jf_compressDims(y,'dim',{'dim','ch'}); % combine x,y,z and name into 1 dimension	 
	end
	if ( ~isempty(opts.bands) ) 
		y=jf_fftfilter(y,'bands',opts.bands,'fs',opts.fs);
	end
	Y = y.X;
	Ydi=y.Ydi;
	Ydi(1).name=[opts.tgt '_' Ydi(1).name];
	Yfs=getSampleRate(y);

  elseif ( strcmp(opts.tgt,'stimSeq') ) % get from the stim-seq
	 [Y,Ydi]=stimSeq2regressor(z,varargin);
	 Yfs=getSampleRate(z);

  else
	 error('Couldnt find the target type to load');
  end  

  if (any(isnan(Y(:)))) Y=interpnans(Y,n2d(Ydi,'time'),'extrap'); end;
end

% remove unneeded channels
if ( ~isempty(opts.idx) ) 
	Y=Y(opts.idx,:,:,:);
	Ydi(1).vals=Ydi(1).vals(opts.idx);
end

% fix sizes if needed
if ( Yfs ~= getSampleRate(z) ) error('Huh sample rates dont match'); end;
if ( size(z.X,dim(1))<size(Y,dim(1)) ) 
  Y   =Y(1:size(z.X,dim(1)),:,:,:);
elseif ( size(z.X,dim(1))>size(Y,dim(1)) )
  z   =jf_retain(z,'dim',dim(1),'idx',1:size(Y,dim(1)));
end

% transform the target feature as wanted
if ( ~isempty(opts.feature) )
	switch opts.feature;
	  case 'raw';
	  case 'velocity';      Y(:,1:end-1,:)=diff(Y,1,dim(1));
	  case 'acceleration';  Y(:,1:end-1,:)=diff(Y,2,dim(1));
	  otherwise; error('Unrecognised feature');
	end
end

% make each target channel unit power in each epoch, so correlation computation is easy
% also means the sse has a sensible scale
if ( ~isempty(opts.normalizeTgt) )
  % Limit to the indicated valid range of the data
  wght=opts.wght;
  if ( ~isempty(wght) && any(strcmpi(wght,{'valid_ms','stimTime_ms'})) )% xxx says time-range to use
	 fn=wght;
	 wght=[];
	 if( isfield(z.di(dim(end)).extra,fn) )
		wght   = false(size(z.X,dim(1)),size(z.X,dim(end)));
		for ei=1:numel(z.di(dim(end)).extra);
		  valid_ms=z.di(dim(end)).extra(ei).(fn);		
		  [ans,vstartIdx] = min(abs(z.di(dim(1)).vals-valid_ms(1))); % start at time==0
		  [ans,vendIdx]   = min(abs(z.di(dim(1)).vals-valid_ms(end))); % end time for this epoch
		  wght(vstartIdx:vendIdx,ei)=true;
		end
	 end
  end
  Y2=Y; Y2(isnan(Y2))=0; if( ~isempty(wght) ) Y2(wght==0)=0; end
  switch (opts.normalizeTgt)
	 case {0,'none'}; Y2(:)=1; % don't normalize
	 case {1,'epoch'}; % normalize for each epoch
		Y2=tprod(Y2,[1 -(2:ndims(Y2)-1) ndims(Y2)],[],[1 -(2:ndims(Y2)-1) ndims(Y2)]); Y2(Y2==0)=1;
	 case {2,'ch'}; % normalize for each channel
		Y2=sum(Y2(:,:).^2,2);
	 case {3,'all'}; % normalize total power
		Y2=Y(:)'*Y(:);
	 otherwise; error('Unrecognized normalization type');				  
  end
  Y2(abs(Y2)<eps)=1; % guard divide by 0
  Y=repop(Y,'/',sqrt(abs(Y2))); % Unit power in valid range
end

% finally set this as the target for the input data
z.Y=Y; z.Ydi=Ydi; z.label = [z.label '_' opts.tgt];
z=jf_addprep(z,mfilename(),sprintf('tgt: %s',opts.tgt),[],[]);
