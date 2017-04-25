function [Popt,res]=cvParafac(X,varargin)
% find a good parafac decompositin of the input tensor data
%
%  [Popt,res]=cvParafac(X,varargin)
%
% Options:
%  fIdxs   - fold guide, N.B. we use the *testing set* (fIdxs>0) to train with!!!!
%  ranks   - [nRank x 1] set of ranks to test
%  Cs      - [nCs x 1] set of regularisation parameters to test
%  reorderC- [-1/0/1] re-oder C into:                                  (1)
%              -1=descending, 0=unchanged, 1=ascending order
%  seedNoise- [nseed x 1] set of seed noises to add to each consequetive restart
%  nRestarts - [int 2] number of restarts to use (outer / inner optimisations) (1)
%  objFuzz - fuzz factor to use in multi-critera optimisation to pick the 'best' set of hyper-parameters
%            larger values make parameter less important                      ([.03 .012 .05 5 .3])
%            Optimisation parameters are: 
%               [tst-error cross-fold-stability solution-degeneracy rank regularisation-stability]
%  outerSoln  - [bool] compute global solution using all the data (1)
%  reuseParms - [bool] re-use parameters from previous C's in subsequence runs for speed (1)
opts=struct('fIdxs',[],'ranks',[],'nRestarts',1,'objFn','parafac_als','Cs',[],'Cscale',[],'reorderC',1,'regType','C','seedNoise',[],...
            'verb',1,'objFuzz',[.03 .012 .05 5 .3],'ortho',0,'reuseParms',1,'outerSoln',1,'foldType','bicvFold');
foldOpts=struct('perm',0,'oneBlock',0);
[opts,foldOpts,varargin]=parseOpts({opts,foldOpts},varargin);
nd=ndims(X);
szX=size(X);

% increasing seed noise over restarts
if ( isempty(opts.seedNoise) ) opts.seedNoise=[0 .1 .1 .2 .2 .3 .3 .4 .4 .5 .5 1]; end;

Cscale=opts.Cscale; 
if ( isempty(Cscale) ) 
  Cscale=(X(:)'*X(:)).^(1/ndims(X));  % N.B. why this value?  
end;

idx={}; for d=1:nd; idx{d}=1:szX(d); end; % index expression for folding
ranks=opts.ranks; if ( isempty(ranks) ) ranks=1:min(szX); end;
Cs=opts.Cs; if ( isempty(Cs) ) Cs=[0 5.^([-7 -6.5 -6 -5 -4.5 -4])]; end;
% ensure C is in order of decreasing? regularisation
if ( opts.reorderC>0)      [Cs,si]=sort(Cs,'ascend'); 
elseif ( opts.reorderC<0)  [Cs,si]=sort(Cs,'descend'); 
end
if( ~all(si(:)==(1:numel(si))') ) warning('re-ordered Cs to increasing order for efficiency'); end;

% setup the folding
fIdxs=opts.fIdxs; nFold=size(fIdxs,ndims(fIdxs));
if ( numel(fIdxs)==1 || numel(fIdxs)==2 ) nFold=fIdxs; fIdxs=[]; 
end;
if ( isempty(fIdxs) ) % generate requested folding
  if ( numel(nFold)>1 ) nRep=nFold(2); nFold=nFold(1); else nRep=1; end;
  if ( nFold==0 ) nFold=3; nRep=3; end;
  if ( strcmp(opts.foldType,'bicvFold') )
    fIdxs=bicvFold(szX,abs(nFold),'repeats',nRep,foldOpts);
  else
    fIdxs=int8(gennFold(ones([prod(szX),1]),abs(nFold),'repeats',nRep,'perm',1));
  end
  if ( nFold<0 ) fIdxs=-fIdxs; end; % swap train/test if wanted
end
if ( ndims(fIdxs)>2 ) % reshape to 2-d
  fIdxs=reshape(fIdxs,[prod(szX),size(fIdxs,numel(szX)+1)]);
end
nFold=size(fIdxs,ndims(fIdxs));
if ( opts.verb>0 ) fprintf('nFold=%d\n',nFold); end;
if ( size(fIdxs,1) ~= prod(szX) ) 
  error('folding size doesnt agree with X size');
end

X2=X(:)'*X(:);
if ( opts.verb>0 ) 
   if ( opts.nRestarts>1 )
      fprintf('(fold/rank.restart) C trn/tst %%Err\n'); 
   else
      fprintf('(fold/rank) C trn/tst %%Err\n'); 
   end
end;

if ( opts.outerSoln )
  seeds={}; trnssei=[];
  for nCompi=1:numel(ranks);
      nComp=ranks(nCompi); % number components to fit here
                           % for this fold, do a few re-starts to
                           % a) assess convergence stability
                           % b) pick the best-fitting solution
      for ri=1:opts.nRestarts; % number of random re-starts to do
         if ( opts.verb > 0 )
            if ( opts.nRestarts<=1 )fprintf('(out/%3d)\t',nComp);
            else                    fprintf('(out/%3d.%2d)\t',nComp,ri);
            end
         end
         seed={}; objTol0=1e-4; % lower tol 1st time round
         for ci=1:numel(Cs);
           if ( opts.reuseParms && ci==1 ) seed=seeds; end; % seed forthis C
            % scale C by number points -- allow for very small training sets
            C = Cs(ci)*Cscale*numel(X)./mean(sum(fIdxs~=0));%*nComp.^2; %scale C so reg/component is roughly the same
            
            [P{1:nd+1}]=feval(opts.objFn,X,'rank',nComp,opts.regType,C,...
                                    'seed',seed,'objTol0',objTol0,...
                                    'seedNoise',opts.seedNoise(min(end,ri)),...
                                    'verb',opts.verb-2,varargin{:}); 
            Err=parafacSSE(X,P);
            if ( opts.ortho ) % ortho to simplify the solution 
              [tmp{1:nd+1}]=parafacOrtho(P);
              Err2=parafacSSE(X,tmp{:});
              if ( Err2<Err*1.1 ) P=tmp; end; % only use if not too much worse
            end 
            if ( opts.reuseParms ) if ( ci==1 ) seeds=P; end; seed=P; objTol0=1e-3; end;
            trnssei(ci,ri)=Err./X2;
            Pi(1:nd+1,ci,ri)=P;
            if ( opts.verb > 0 ) 
              fprintf('%.2f/NA  \t',trnssei(ci,ri)); 
            end;
          end % for Cs
         if ( opts.verb > 0 ) fprintf('\n'); end
      end
      [ans,minri]=min(trnssei,[],2); % lowest sse on *training* set solution is picked
      for ci=1:numel(Cs);
         Pout{ci,nCompi}=Pi(1:nd+1,ci,minri(ci));%parafacOrtho(P{1:nd+1,minri});%
         outsse(ci,nCompi)=trnssei(ci,minri(ci));
      end
   end
end

% Now do the loop over inner folds
Pi={};
for fi=1:nFold;
  trnIdx = reshape(fIdxs(:,fi)<0,szX);
  tstIdx = reshape(fIdxs(:,fi)>0,szX);
  trnX2  = sum(X(trnIdx).^2);
  tstX2  = sum(X(tstIdx).^2);
  seeds={}; seed={}; % reset the seed solution between folds
  trnssei=[]; tstssei=[]; Pi={};
  for nCompi=1:numel(ranks);
    nComp=ranks(nCompi); % number components to fit here
    % for this fold, do a few re-starts to
    % a) assess convergence stability
    % b) pick the best-fitting solution
    for ri=1:opts.nRestarts(min(end,2)); % number of random re-starts to do
      if ( opts.verb > 0 )
        if ( opts.nRestarts(min(end,2))<=1 )fprintf('(%3d/%3d)\t',fi,nComp);
        else                    fprintf('(%3d/%3d.%2d)\t',fi,nComp,ri);
        end
      end
      seed={}; objTol0=1e-4; % reset the seed value
      for ci=1:numel(Cs);
        C(ci,nCompi) = Cs(ci)*Cscale;%*nComp.^2; % scale C such that reg/component is roughly the same

        if ( opts.reuseParms && ci==1 ) seed=seeds; end; % seed forthis C
      
        
        [P{1:nd+1}]=feval(opts.objFn,X,'rank',nComp,opts.regType,C(ci,nCompi),...
                                'seed',seed,'objTol0',objTol0,'seedNoise',opts.seedNoise(min(end,ri)),...
                                'wght',trnIdx,'verb',opts.verb-2,varargin{:}); 
        if ( opts.ortho ) [P{:}]=parafacOrtho(P); end % ortho to simplify the solution
        if ( opts.reuseParms ) if ( ci==1 ) seeds=P; end; seed=P; objTol0=1e-4; end;
        % get the train/test set fit qualities
        Ae=parafac(P);
        Err = X-Ae;
        trnssei(ci,ri) = sum(Err(trnIdx).^2);    tstssei(ci,ri)=sum(Err(tstIdx).^2);
        trnsseip(ci,ri)= trnssei(ci,ri)./trnX2;  tstsseip(ci,ri)=tstssei(ci,ri)./tstX2;
        Pi(1:nd+1,ci,ri)=P;
        clear Ae Err;
        if ( opts.verb > 0 ) 
          fprintf('%.2f/%.2f\t',trnsseip(ci,ri),tstsseip(ci,ri)); 
        end;
      end % for Cs
      if ( opts.verb > 0 ) fprintf('\n'); end
    end

    % assess stability of the different re-starts & pick the best one to be the fold solution
    [ans,minri]=min(trnssei,[],2); % lowest sse on *training* set solution is picked
    for ci=1:numel(Cs);
      Ps{ci,nCompi,fi}=Pi(1:nd+1,ci,minri(ci));%parafacOrtho(Pi{1:nd+1,minri});%
      trnsse(ci,nCompi,fi)=trnsseip(ci,minri(ci));
      tstsse(ci,nCompi,fi)=tstsseip(ci,minri(ci));    
    end
  end % rank
end % fold

  % compute the summary info
for nCompi=1:numel(ranks);
  nComp=ranks(nCompi);
  for ci=1:numel(Cs);
    % compute the cross fold summary information
    c=zeros(nFold,nFold); cc=zeros([nComp,nFold,nFold]); degeni=zeros([nComp nComp nFold]); effranki=zeros(nFold,1);
    for i=1:nFold;
      P=Ps{ci,nCompi,i};
      % cross fold correlation -> solution stability
      for j=1:i-1;
        [cij,ccij]=parafacCorr(P,Ps{ci,nCompi,j}); 
        c(i,j)=cij; c(j,i)=cij;
        cc(:,i,j)=ccij; cc(:,j,i)=ccij;  % [nComp x nFold x nFold]
      end
      c(i,i)=1; cc(:,i,i)=1; % with itself has corr=1
      % % mimage(P{1+fitDim},'disptype','plot'); % plot the model
      % within fold btw-component correlation -> solution degeneracy
      [cii,ccii,corrMx]=parafacCorr(P);
      % compute the congruence coefficient
      degenii=prod((corrMx),3)-eye(size(corrMx,1)); % [nComp x nComp x nFold]
      % convert to covariance matrix...
      %degenii=repop(P{1},'*',repop(degenii,'*',P{1}'))./(P{1}(:)'*P{1}(:)); % include relative component size
      % OR normalise for the number of components
      degeni(:,:,i)=degenii;
      effranki(i) = sum(P{1}>max(P{1})*1e-6);
    end
    cc = shiftdim(sum(cc,1))./size(cc,1);    % per folding pair-wise correlation [nFold x nFold]
    stab(:,:,ci,nCompi)=cc;  % stability is amount of correlation between
                             % foldings [nFold x nFold x nC x nComp]
    %degeni(degeni>0)=degeni(degeni>0)*1; % positive degeneracy isn't tooo bad?
    degen{ci,nCompi}=degeni; % amount of solution degeneracy [nC x nComp] {[nComp x nComp]}
    effrank(ci,nCompi)=sum(effranki)/nFold; % [nC x nComp]
    % N.B. use robust average because of the issue of *bad* folds
    obj(ci,nCompi,:)= [robustCenter(tstsse(ci,nCompi,:),3) ... % sse
                       robustCenter(cc(:))  ... % cross fold stability
                       sum(abs(degeni(:)))./nFold./nComp]; % solution degeneracy
  end
end; 

if ( opts.verb>0 ) fprintf('---------------------\n'); end;
if ( opts.verb>0 )
   for nCompi=1:numel(ranks);
      nComp=ranks(nCompi);
      fprintf('(ave/%3d)\t',nComp);
      for ci=1:numel(Cs);
        fprintf('%.2f/%.2f\t',robustCenter(trnsse(ci,nCompi,:),3),robustCenter(tstsse(ci,nCompi,:),3));
      end
      fprintf('\n');
      fprintf('(stb/dgn)\t');
      for ci=1:numel(Cs);
        fprintf('%.2f/%.2f\t',obj(ci,nCompi,2),obj(ci,nCompi,3));
      end
      fprintf('\n');
   end
end

% use the computed measures to pick the 'optimal' fitting parameters
% Use multi-criteria optimisation -> find a non-dominated best trade-off between sse and stability
% search for best point, using multi-objective optimisation
% we want a point which is within a certain distance of the optimal for every criteria
% objFuzz tell's us for each criteria what this distance should be,
% i.e. inversly related to importance
obj = cat(3,obj(:,:,1)+log10(obj(:,:,1))/5,... % for large SSE linear is good, for small SSE log is good
          -obj(:,:,2),... % higher stability is good
          obj(:,:,3),... % lower degeneracy is good
          repmat(ranks,numel(Cs),1),... % lower rank is good
          filter([1 1]/2,1,obj(:,:,1),[],2)); % lower regularisation robustness is good
% N.B. special mins on the degeneracy to avoid the rank-1 = degen=0 dominating everything
mins=[min(min(obj(:,:,1))) min(min(obj(:,:,2))) min(min(obj(:,:,3)+(obj(:,:,3)==0))) min(min(obj(:,:,4))) min(min(obj(:,:,5)))];
[optCi,optnCompi]=multiCriteriaOpt(obj,opts.objFuzz,mins,opts.verb-1);

% fit model with these parameters on the total data
fprintf('------------------\n');
fprintf('(C=%3d,r=%2d)*\t',optCi,ranks(optnCompi));
clear ssei;
if ( opts.outerSoln ) % already done, just pick it up
   Popt=Pout{optCi,optnCompi};
   if ( opts.verb>0 ) fprintf('%2f\n',outsse(optCi,optnCompi)); end;
else
  % spend roughly as long as outerSoln would have taken
  nrestarts=opts.nRestarts+numel(Cs);
  for ri=1:nrestarts; % number of random re-starts to do - 
      [Pi{1:nd+1,ri}]=feval(opts.objFn,X,'rank',ranks(optnCompi),opts.regType,C(optCi,optnCompi)*numel(X)./mean(sum(fIdxs<0)),'seedNoise',opts.seedNoise(min(end,ri)),'verb',opts.verb-2,varargin{:}); 
      % print the fit quality...
      [ssei(ri),sseip(ri)]=parafacSSE(X,Pi{1:nd+1,ri});
      if ( opts.verb > 0 ) fprintf('%.5f',sseip(ri)); if ( ri<nrestarts ) fprintf('/'); end; end
   end
   if ( opts.verb > 0 ) fprintf('\n'); end
   [outsse,minri]=min(sseip); % lowest ssep solution is picked
   Pout={}; Popt=Pi(1:nd+1,minri);
end

% results structure
if ( nargout>1 )
  res.di    =mkDimInfo(size(Ps),'C',[],C','rank',[],ranks,'fold',[],[]);
  res.opts  =opts;
  res.C     =C;
  res.fIdxs =reshape(fIdxs,[szX size(fIdxs,2)]);
  res.stab  =stab;  % stability is amount of correlation between foldings [nFold x nFold x nC x nComp]
  res.degen =degen; % amount of solution degeneracy [nC x nComp]
  res.effrank=effrank; % [nC x nComp]
  res.obj   =obj; 
  res.opt.soln =Popt;
  res.opt.optCi=optCi;
  res.opt.optnCompi=optnCompi;
  res.opt.optRank=ranks(optnCompi);
  res.fold.soln=Ps;
  res.fold.trnsse=trnsse;
  res.fold.tstsse=tstsse;
  res.trnsse=mean(trnsse,3);
  res.tstsse=mean(tstsse,3);
  res.soln  =Pout; 
  res.outsse=outsse;
  res.totsse=X(:)'*X(:);
end
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
function [A]=parafac(S,varargin);
% Compute the full tensor specified by the input parallel-factors decomposition
%
% [A]=parafac(S,U_1,U_2,...);
%
U=varargin;
if ( nargin==1 && iscell(S) ) U=S(2:end); S=S{1}; end;
if ( numel(U)==1 && iscell(U{1}) ) U=U{1}; end;
if ( numel(S)==1 && size(U{1},2)>1 && S==1 ) S=ones(size(U{1},2),1); end;
nd=numel(U); A=shiftdim(S(:),-nd);  % [1 x 1 x ... x 1 x M]
for d=1:nd; A=tprod(A,[1:d-1 0 d+1:nd nd+1],U{d},[d nd+1],'n'); end
A=sum(A,nd+1); % Sum over the sub-factor tensors to get the final result
return;


%----------------------------------------------------------
function testCase()
tmp=load('~/projects/bci/parafac/amino'); A=tmp.X; clear tmp; % [5 x 201 x 61] should be rank 3, also is non-negative
tmp=load('~/projects/bci/parafac/FIA');   A=tmp.FIA.data; % [] should be rank-6, local-minima problems
tmp=load('~/projects/bci/parafac/KojimaGirls'); A=tmp.KojimaGirls.data; % 2-factor degeneracy...
tmp=load('~/projects/bci/parafac/brod'); A=tmp.X; clear tmp; % [ 10 bread (rep) x 11 attributes x 8 judges ] - rank ?


[optP,res]=cvParafac(A,'ranks',1:8,'nRestarts',3);
clf;mimage(optP{2:end},'disptype','plot');
clf;
subplot(211);imagesc(log(sum(sse,3)));title('log(sse)');xlabel('rank');ylabel('C');
subplot(212);imagesc(stab);title('stability');xlabel('rank');ylabel('C');


t={rand(3,1) abs(randn(10,3)) abs(randn(9,3)) abs(cumsum(randn(8,3),1))};
A=parafac(t{:}); A=A+randn(size(A))*1e-2;
clf;mimage(t{:},'disptype','plot');
[optP,res]=cvParafac(A,'ranks',1:8,'nRestarts',10);
[c,cc,cmx]=parafacCorr({tS tU{:}},optP);
clf;subplot(131);plot(optP{2});subplot(132);plot(optP{3});subplot(133);plot(optP{4});


t={abs(rand(3,1)) abs(randn(10,3)) abs(randn(9,3)) abs(cumsum(randn(100,3),1))};
t=parafacStd(t{:});
A=parafac(t{:}); A=A+randn(size(A))*1e-3;
optP=parafac_als(A,3,'rank',3,'verb',1,'seedNoise',0)
[optP,res]=cvParafac(A,'ranks',1:4);
clf;mimage(optP{2:end},'disptype','plot')
[c,cc]=parafacCorr(t,optP)

Ci=1;Ri=2;fi=1; clf;mimage(res.fold.soln{Ci,Ri,fi}{2:end},'disptype','plot');

