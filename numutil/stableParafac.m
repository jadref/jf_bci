function [Popt,sse,stab,degen,P]=stabelParafac(X,dim,varargin)
% find a good parafac decompositin of the input tensor data
%
% Options:
%  fIdxs   - fold guide
%  ranks   - [nRank x 1] set of ranks to test
%  Cs      - [nCs x 1] set of regularisation parameters to test
%  seedNoise- [nseed x 1] set of seed noises to add to each consequetive restart
%  nRestarts - [int] number of restarts to use
%  objFuzz - [sse stab degen] fuzz factor to use in multi-critera optimisation to
%            pick the 'best' set of hyper-parameters
opts=struct('fIdxs',[],'ranks',[],'nRestarts',3,'Cs',[],'seedNoise',[0 .5],'verb',1,'objFuzz',[1 .05 .3]);
[opts,varargin]=parseOpts(opts,varargin);
nd=ndims(X);
szX=size(X);

idx={}; for d=1:nd; idx{d}=1:szX(d); end; % index expression for folding
ranks=opts.ranks; if ( isempty(ranks) ) ranks=1:min(szX); end;
Cscale=sqrt(abs(X(:)'*X(:)));
Cs=opts.Cs; if ( isempty(Cs) ) Cs=[0 5.^([-8 -6 -5 -4])]; end;

% setup the folding
fIdxs=opts.fIdxs; nFold=size(fIdxs,2);
if ( numel(fIdxs)==1 ) nFold=fIdxs; fIdxs=[]; end;
if ( isempty(fIdxs) ) % generate requested folding
  if ( nFold==0 ) 
    fIdxs=gennFold(ones([szX(dim),1]),2,'repeats',2); 
  else
    fIdxs=gennFold(ones([szX(dim),1]),nFold); 
  end;
end
nFold=size(fIdxs,2);
if ( size(fIdxs,1) ~= szX(dim) ) 
  error('folding size doesnt agree with X size');
end
fitDim=setdiff(1:nd,dim);

if ( opts.verb>0 ) fprintf('(C,rank) fold\n'); end;
for nCompi=1:numel(ranks);
  nComp=ranks(nCompi); % number components to fit here
  for ci=1:numel(Cs);
    if ( opts.verb > 0 )
      fprintf('(%2d,%2d)\t',ci,nComp);
    end
    for fi=1:nFold;
      idx{dim} = find(fIdxs(:,fi)>0); % N.B. use testing sets to *NO* overlap!
      Xtrn = X(idx{:});

      % for this fold, do a few re-starts to
      % a) assess convergence stability
      % b) pick the best-fitting solution
      for ri=1:opts.nRestarts; % number of random re-starts to do
        [Pi{1:nd+1,ri}]=parafac_als(Xtrn,'rank',nComp,'C',Cscale*Cs(ci),'seedNoise',opts.seedNoise(min(end,ri)),varargin{:}); 
        % print the fit quality...
        [ssei(ri),sseip(ri)]=parafacSSE(Xtrn,Pi{1:nd+1,ri});
        if ( opts.verb > 0 ) fprintf('%.5f',sseip(ri)); if ( ri<opts.nRestarts ) fprintf('/'); end; end
      end
      if ( opts.verb > 0 ) fprintf('\t'); end
      % assess stability of the different re-starts & pick the best one to be the fold solution
      [ans,minri]=min(ssei); % lowest sse solution is picked
      P{ci,nCompi,fi}=Pi(1:nd+1,minri);%parafacOrtho(Pi{1:nd+1,minri});%
      sse(ci,nCompi,fi)=sseip(minri);
    end

    % compute the cross fold solution correlations
    c=zeros(nFold,nFold); cc=zeros([nComp,nFold,nFold]); degeni=zeros([nComp nComp nFold]); effranki=zeros(nFold,1);
    for i=1:nFold;
      % cross fold correlation -> solution stability
      for j=1:i-1;
        [cij,ccij]=parafacCorr(P{ci,nCompi,i}([1 1+fitDim]),P{ci,nCompi,j}([1 1+fitDim])); % similarity ignoring split dim
        c(i,j)=cij; c(j,i)=cij;
        cc(:,i,j)=ccij; cc(:,j,i)=ccij;  % [nComp x nFold x nFold]
      end
      c(i,i)=1; cc(:,i,i)=1; % with itself has corr=1
      % % mimage(P{ci,nCompi,j}{1+fitDim},'disptype','plot'); % plot the model
      % within fold btw-component correlation -> solution degeneracy
      [ans,ans,corrMx]=parafacCorr(P{ci,nCompi,i}([1 1+fitDim]),P{ci,nCompi,i}([1 1+fitDim]));
      degeni(:,:,i)=sum(abs(corrMx),3)-eye(size(corrMx,1))*size(corrMx,3); % [nComp x nComp x nFold]
      effranki(i) = sum(P{ci,nCompi,i}{1}>max(P{ci,nCompi,i}{1})*1e-6);
    end
    cc = shiftdim(sum(cc,1))./size(cc,1);    % per folding pair-wise correlation [nFold x nFold]
    stab(:,:,ci,nCompi)=cc;  % stability is amount of correlation between foldings [nFold x nFold x nC x nComp]
    degen{ci,nCompi}=sum(degeni,3)/nFold./nComp./nComp; % amount of solution degeneracy [nC x nComp]
    effrank(ci,nCompi)=sum(effranki)/nFold; % [nC x nComp]
    obj(ci,nCompi,:)= [mean(tstsse(ci,nCompi,:),3) ... % sse
                       sum(sum(stab(:,:,ci,nCompi)))./nFold./nFold ... % cross fold stability
                       sum(degen{ci,nCompi}(:))]; % solution degeneracy
    if ( opts.verb  > 0 )
      fprintf('\n(%2d,%2d) rank=%g\tsse=%g/%g\tstab=%g\tdegen=%g\n',ci,nComp,...
              effrank(ci,nCompi),...
              mean(trnsse(ci,nCompi,:),3),obj(ci,nCompi,1),obj(ci,nCompi,2),obj(ci,nCompi,3));
    end
  end
  if ( opts.verb > 0 ) fprintf('\n'); end;
end

% use the computed measures to pick the 'optimal' fitting parameters
% Use multi-criteria optimisation -> find a non-dominated best trade-off between sse and stability
% search for best point, using multi-objective optimisation
% we want a point which is within a certain distance of the optimal for every criteria
% objFuzz tell's us for each criteria what this distance should be
step=3; t=1; bracket=0;
mins=[min(min(log10(obj(:,:,1)))) min(min(-obj(:,:,2))) min(min(obj(:,:,3)+(obj(:,:,3)==0)))];
for i=1:10;
  pts=log10(obj(:,:,1))<mins(1)+t*opts.objFuzz(1) ... % sse
      & -obj(:,:,2)<mins(2)+t*opts.objFuzz(2) ... % stab
      & obj(:,:,3)<mins(3)+t*opts.objFuzz(3); % degen
  nPts=sum(pts(:));
  %fprintf('%d) %g %d\n',i,t,sum(pts(:)));
  if ( nPts==1 ) break;
  elseif ( ~bracket && nPts==0 ) step=step*1.6; t=t+step; % forward until bracket
  elseif ( ~bracket && nPts>0 )  step=step*.62; t=t-step; bracket=1; 
  elseif ( bracket &&  nPts==0 ) step=step*.62; t=t+step; % golden ratio search
  elseif ( bracket &&  nPts>0  ) step=step*.62; t=t-step;
  end
end
optIdx=find(pts);
optIdx=optIdx(1); % if >1 pick the 1st
[optCi,optnCompi]=ind2sub([size(obj,1),size(obj,2)],optIdx);

% fit model with these parameters on the total data
fprintf('------------------\n');
fprintf('(%2d,%2d)*\t',optCi,ranks(optnCompi));
clear ssei;
for ri=1:opts.nRestarts; % number of random re-starts to do
  [Pi{1:nd+1,ri}]=parafac_als(X,'rank',ranks(optnCompi),'C',Cscale*Cs(optCi),'seedNoise',opts.seedNoise(min(end,ri)),varargin{:}); 
  % print the fit quality...
  [ssei(ri),sseip(ri)]=parafacSSE(X,Pi{1:nd+1,ri});
  if ( opts.verb > 0 ) fprintf('%.5f',sseip(ri)); if ( ri<opts.nRestarts ) fprintf('/'); end; end
end
if ( opts.verb > 0 ) fprintf('\n'); end
[ans,minri]=min(ssei); % lowest sse solution is picked
Popt=Pi(1:nd+1,minri);

return;

%----------------------------------------------------------
function x=rstd(x,thresh);
if ( nargin < 2 || isempty(thresh) ) thresh=2.5; end;
inc=true(size(x));
for i=1:4;
  mu=sum(x(inc))/sum(inc(:)); sigma=sqrt(sum((x(inc)-mu).^2)/sum(inc(:)));
  inc = (x < mu+thresh*sigma) & (x > mu-thresh*sigma);
end
x = (x-mu)./sigma;
return;

%----------------------------------------------------------
function testCase()
tmp=load('amino'); A=tmp.X; clear tmp
tmp=load('FIA');   A=tmp.FIA.data;
tmp=load('KojimaGirls'); A=tmp.KojimaGirls.data;

[optP,sse,stab,degen,P]=stableParafac(A,3,'ranks',1:4);
clf;mimage(optP{2:end},'disptype','plot');
clf;
subplot(211);imagesc(log(sum(sse,3)));title('log(sse)');xlabel('rank');ylabel('C');
subplot(212);imagesc(stab);title('stability');xlabel('rank');ylabel('C');


tS=rand(3,1); tU={abs(randn(10,3)) abs(randn(9,3)) abs(cumsum(randn(8,3),1))};
clf;subplot(131);plot(tU{1});subplot(132);plot(tU{2});subplot(133);plot(tU{3});
[optP,sse,stab,degen,P]=stableParafac(A,3,'ranks',1:8);
[c,cc,cmx]=parafacCorr({tS tU{:}},optP);
clf;subplot(131);plot(optP{2});subplot(132);plot(optP{3});subplot(133);plot(optP{4});


t={abs(rand(3,1)) abs(randn(10,3)) abs(randn(9,3)) abs(cumsum(randn(100,3),1))};
t=parafacStd(t{:});
A=parafac(t{:}); A=A+randn(size(A))*1e-3;
optP=parafac_als(A,3,'rank',3,'verb',1,'seedNoise',0)
[optP,sse,stab,degen,P]=stableParafac(A,3,'ranks',1:4);
clf;mimage(optP{2:end},'disptype','plot')
[c,cc]=parafacCorr(t,optP)

Ci=1;Ri=2;fi=1; clf;mimage(P{Ci,Ri,fi}{2:end},'disptype','plot');
