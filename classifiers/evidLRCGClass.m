function [z]=evidLRCGClass(z,trnIdx,tstIdx,varargin)
% the format of the output structure
loss=struct('func','class_loss','train',1,'train_se',1,...
            'test',1,'test_se',1);
% get the training set
if ( nargin < 2 | isempty(trnIdx) ) 
 trnIdx=false(ntrials(z),1);trnIdx(getouterfold(z))=true; tstIdx=~trnIdx;
end;
X=z.x; Y=z.y;
% train
[w,b,alpha]=evidenceLRCG(X(trnIdx,:),Y(trnIdx),50,1,1);

% add prep info?
z=addprepinfo(z,mfilename(),sprintf('Optimise classifer by LR'),[],struct('w',w,'b',b,'alpha',alpha),'WTF?',[]);

% evaluate performance
dv=X*w+b;
[conf,trnbin]=dv2conf(dv(trnIdx),Y(trnIdx));  % N.B. dv2conf reports class rate
[conf,tstbin]=dv2conf(dv(tstIdx),Y(tstIdx));

loss.train=1-trnbin(1);  %N.B. error rates!
loss.test =1-tstbin(1);  %N.B. error rates!
%s.wb=[w;b];

% add the results info to z
summary=sprintf('%s optimised train:%s, test:%s',mfilename,lossstr('class loss',loss.train),lossstr('class loss',loss.test));
z = setresults(z, mfilename, summary, struct('w',w,'b',b,'alpha',alpha), loss);
