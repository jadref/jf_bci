function [z]=evidCSPLRClass(z,trnIdx,tstIdx,varargin)
opts=struct('nfilt',2);
cgopts=struct('pCond','adapt');
opts=parseOpts(varargin,opts,cgopts);
% the format of the output structure
loss=struct('func','class_loss','train',1,'train_se',1,...
            'test',1,'test_se',1);

fprintf('To avoid overfitting using %d of %d trials\n',numel(trnIdx),ntrials(z));

% convert trnIdx/tstIdx to logical spec
if ( ~islogical(trnIdx) ) 
   ans=trnIdx; trnIdx=false(ntrials(z),1); trnIdx(ans)=true; 
end
if ( ~islogical(tstIdx) ) 
   ans=tstIdx; tstIdx=false(ntrials(z),1); tstIdx(ans)=true; 
end

% Extract the data
[N,indim,nEx]=size(z.x); Y=z.y;
fX=rcovFilt(permute(z.x,[3 2 1]));

% seed with CSP solution
XX1=sum(fX(:,:,trnIdx & Y()>0),3);XX=sum(fX(:,:,trnIdx),3);
[U,D]=eig(XX1,XX); [sD,sI]=sort(abs(.5-diag(D)));D=diag(D);D=D(sI);U=U(:,sI);
sf0=U(:,end:-1:end-opts.nfilt+1); 

% ALPHA: init
% evidence max
phiX=cspFeatFn(sf0,fX); 
[w,b,alpha0]=evidenceLR(phiX(:,trnIdx)',Y(trnIdx)>0,5,0);

% SFA: init
sfa0=[sf0(:);log(alpha0)]; sfa=sfa0;

% train
sfa=nonLinConjGrad(@(w) evidenceCSPLRFn(w,fX(:,:,trnIdx),Y(trnIdx)>0),sfa0,'plot',0,'verb',1,'maxEval',5000,'maxIter',inf,'curveTol',1e-3,'pCond',[],'alpha0',1e-4);

% Results extraction
[f,df,ddf,wb]=evidenceCSPLRFn(sfa,fX(:,:,trnIdx),Y(trnIdx)>0,3,0);
w=wb(1:end-1);b=wb(end);sf=reshape(sfa(1:end-1),indim,opts.nfilt);alpha=exp(sfa(end));
phiX=cspFeatFn(sf,fX); 

% Add some info about what we've done to the object
summarystr=sprintf('%s optimized %d CSP filters and classifier',mfilename,opts.nfilt);
info=struct('bandbyclass',struct(),'filter',sf,'folded_roc',struct(),'wb',wb);
z=addprepinfo(z, mfilename, summarystr, opts, info, 'notyetimplemented', info);
z.x=phiX'; % set the new data to the filtered bits

% evaluate performance
dv=phiX'*w+b;
% N.B. dv2conf reports class rate
[conf,trnbin]=dv2conf(dv(trnIdx),Y(trnIdx),[],[],1);  
[conf,tstbin]=dv2conf(dv(tstIdx),Y(tstIdx),[],[],1);

loss.train=1-trnbin(1);  %N.B. error rates!
loss.test =1-tstbin(1);  %N.B. error rates!
%s.wb=[w;b];

% add the results info to z
summary=sprintf('%s optimised class loss %s',mfilename,lossstr('class loss',loss.test));
z = setresults(z, mfilename, summary, [], loss);
