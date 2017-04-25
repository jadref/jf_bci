function [z,opts]=jf_extractLinearClassifier(z,varargin)
% extract a linear classifier from a trained data-set
% Options:
opts=struct('verb',0);
[opts]=parseOpts(opts,varargin);

% Extract the classifier weight vector
clspi=m2p(z,'jf_cvtrain');
soln=z.prep(clspi).info.res.opt.soln;

% use the classifier objective function info to decide how to transform into a cov-weighting
for spi=1:numel(soln);
  switch ( z.prep(clspi).opts.objFn );
 
   case {'klr_cg','l2svm_cg'}; % normal linear-kernel method
    if ( ~strcmpi(z.prep(m2p(z,'jf_compKernel')).opts.kerType,'linear') )
      error('Only supported for linear kernels'); 
    end
    wdi = z.prep(m2p(z,'jf_compKernel')).info.odi;
    X = z.prep(m2p(z,'jf_compKernel')).info.X;
    szX = size(X);
    % check for n-d training
    if ( isfield(z,'Ydi') )
      trD = n2d(wdi,{z.Ydi(setdiff(1:end-1,n2d(z.Ydi,'subProb'))).name});
    else
      trD = ndims(X);
    end
    xidx=1:ndims(X); xidx(trD)=-(1:numel(trD)); 
    W_cls{spi}=tprod(X,xidx,reshape(soln{spi}(1:end-1),[szX(trD) 1]),-(1:numel(trD)),'n'); b=soln(end);

    wdi(trD(2:end))=[]; wdi(trD(1))=z.Ydi(n2d(z.Ydi,'subProb')); % remove extra trDims
    
   case {'LSigmaRegKLR','LSigmaRegKLR_als'};
    W_cls{spi}=parafac(soln{spi}{1:end-1}); % put the factors back together
    b=soln{1}{end};  % get the bias
    wdi = z.prep(clspi).info.wdi;
    wdi(n2d(z,'epoch'))=z.Ydi(n2d(z.Ydi,'subProb'));
    wdi(end).name='wght';

   case {'mkKLR'};
    kerstep=m2p(z,'jf_compKernel');
    if ( ~strcmpi(z.prep(kerstep).opts.kerType,'linear') )
      error('Only supported for linear kernels'); 
    end
    wdi = z.prep(kerstep).info.odi;
    X   = z.prep(kerstep).info.X;
    szX = size(X);
    % check for n-d training
    if ( isfield(z,'Ydi') )
      trD = n2d(wdi,{z.Ydi(setdiff(1:end-1,n2d(z.Ydi,'subProb'))).name});
    else
      trD = ndims(X);
    end
    xidx=1:ndims(X); xidx(trD)=-(1:numel(trD)); 
    W_cls{spi}=tprod(X,xidx,reshape(soln{spi}{1}(1:end-1),[szX(trD) 1]),-(1:numel(trD)),'n'); b=soln(end);

    wdi(trD(2:end))=[]; wdi(trD(1))=z.Ydi(n2d(z.Ydi,'subProb')); % remove extra trDims

    if ( ~isempty(z.prep(kerstep).opts.grpDim) ) % apply the feature weighting
      W_cls{spi}=repop(W_cls{spi},'*',shiftdim(soln{spi}{2}(:),-n2d(z.prep(kerstep).info.odi,z.prep(kerstep).opts.grpDim)+1));
    end

   otherwise;
    error('Unrecognised objective function');
  end
end
% combine the weights into 1 matrix
z.X = cat(n2d(wdi,'subProb'),W_cls{:}); 
z.di= wdi;

% update the meta-info
summary='';
info=[];
z=jf_addprep(z,mfilename,summary,opts,info);
return;
%--------------------------------------------------------------------------
function testCase()
z=jf_load('external_data/mpi_tuebingen/vgrid/nips2007/1-rect230ms','jh','flip_opt');
z=jf_load('external_data/mlsp2010/p300-comp','s1','trn');

% make a simple ERP style toy problem
sources={{'gaus' 80 40} ...  % ERP peak
         {'coloredNoise' exp(-[inf(1,0) zeros(1,1) linspace(0,5,40)])} }; % noise signal
n2s=1; y2mix  =cat(3,[.5  n2s],[0  n2s]); % per class ERP magnitude % [ nSrcLoc x nSrcSig x nLab ]
Yl     =sign(randn(100,1));  % epoch labels
z=jf_mksfToy('Y',Yl,'sources',sources,'y2mix',y2mix,'nCh',10,'nSamp',300,'fs',100);

r=jf_cvtrain(jf_compKernel(z));
w=jf_clsfrWeights(r)

