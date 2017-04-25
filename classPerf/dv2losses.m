function [conf,binCls,eeoc,auc]=dv2conf(Y,dv,subProbs,probType,verb)
% function [conf,binCls,eeoc,auc]=dv2conf(Y,dv,subProbs,probType,verb)
% Compute the confusion matrix and other performance measures for a binary
% or multi-class classification problem.  The biggest output for each class
% is the predicted classification.  So this row of the confusion matrix is
% simply the # of times this classifier gave the highest decision value.
%
%Input:
% dv -- decision values,either [Nx1] for regress/binary or [NxL] for multiclass
% Y -- target values, either [Nx1] or [NxL]
% subProbs -- list of target subProbs for multi-class.  (def. unique(Y))
% probType -- type of problem, {'binclass','nclass','regress'}
% verb     -- verbosity level
%
%Output:
% conf   -- LxL confusion matrix -- rows=predictions, cols=true labels
% binCls -- Lx1 vector of binary classification performances
% eeoc   -- Lx1 vector of equal error operating point values
% auc    -- Lx1 vector of area under the curve vales
% $Id: dv2conf.m,v 1.14 2007-09-10 08:52:37 jdrf Exp $
if ( nargin < 2 ) help dv2conf; return; end;
if ( nargin < 3 ) subProbs=[]; else L=length(subProbs); end;
if ( isempty(subProbs) )
   if ( min(size(Y))==1 )
      subProbs=unique(Y); 
      subProbs(subProbs==0)=[]; % 0 class labels are ignored
      L=length(subProbs);
   else L=size(Y,2); end
end    
if ( isempty(dv) ) 
   conf=zeros(L); binCls=zeros(L,1); eeoc=zeros(L,1); auc=zeros(L,1); return ; 
end;
% decide on the type of problem we're looking at.
if ( nargin < 4 || isempty(probType) ) 
  if ( min(size(Y))==1 )     
    if ( min(size(dv)) == 1 )
      % the number of subProbs in Y determines if this is a regression prob
      if ( L == 2 ) probType='binclass'; else probType='regress'; end; 
    else % dv has multiple dimesions, so assume its n-class
      probType='nclass';
    end
  else % Y is multi-dimensional so assume its a n-class problem
     if ( size(Y,2)==2 ) probType='binclass'; subProbs=[0 1]; 
     else probType='nclass'; end
     %probType='nclass';
  end
else
  probType=lower(probType);
end
if ( nargin < 5 | isempty(verb) ) verb=0; end;

% convert targets to indicator array for classification problems
if ( any(strfind(probType,'class')) )
  if ( min(size(Y)) == 1 ) 
     if ( isempty(subProbs) ) 
        error('Need to give subProbs set with 1D input multi-class problem');
     end
    [Y]=lab2ind(Y,subProbs);
    L=size(Y,2);
  else
    Y=logical(Y); 
    L=size(Y,2);
    if ( isempty(subProbs) ) subProbs=1:L; end;
  end  
end

if ( min(size(dv))==1 ) N=length(dv); else  N=size(dv,1); end;
if ( size(dv,1)<size(dv,2) ) dv=dv'; end; % convert to column order
% computed the predicted output values
switch(probType)
  case 'binclass'; 
    if ( subProbs(1) > 0 ) ii=1; else ii=-1; end;
    I=(dv*ii<0)+1; 
 case 'nclass';
  if ( size(dv,2) > 1 ) [Val,I]=max(dv');
  else   % 1 classifier => decision is to closest value.      
    err=abs(repmat(dv,1,length(subProbs))-repmat(subProbs(:)',[size(dv,1),1]));
    [Val,I]=min(err,[],2); 
    % convert decis vals to n-d indicator matrix of nearest target value
    dv(:,1:L)=repmat(dv,1,L);%zeros(N,L); dv(sub2ind(size(dv),1:N,I'))=1;
    if ( L==2 ) dv=dv-mean(subProbs); dv(:,1)=-dv(:,1); end;
  end
 case 'regress';  I=dv;     
end

% compute the error measures.
if ( strfind(probType,'class') )
   for c=1:L;
      N=sum(Y(:,c)~=0);
      switch (probType)
        case 'nclass';
          binCls(c)=sum( dv(:,c).*Y(:,c)>0 )./N;
          [sdv,sidx]=sort(dv(:,c),'ascend');      
        case 'binclass';
          signc=sign(subProbs(c));
          binCls(c)=sum( dv(:).*Y(:,c).*signc>0 )./N;
          [sdv,sidx]=sort(dv(:)*signc,'ascend');
      end
      % compute the EEOC, i.e. point where tp=tn
      if ( sum(Y(sidx,c)>0)*sum(Y(sidx,c)<0) > 0 )
         stn=cumsum(Y(sidx,c)<0)./sum(Y(sidx,c)<0);
         stp=(1-cumsum(Y(sidx,c)>0)./sum(Y(sidx,c)>0));
         [m,mini]=min(abs(stp-stn)); eeoc(c)=mean([stp(mini),stn(mini)]);    
         auc(c)=-sum([stp(1); diff(stp)].*stn);
      else
         eeoc(c)=0;auc(c)=0;
      end
      % compute the confusion matrix
      for j=1:L;    
         conf(c,j)=sum(I(Y(:,c)>0)==j);% / length(pos);
      end      
   end
else % regression problem, conf is the sum abs error & binCls sum sqared error
  conf=[]; binCls=sum( abs(dv - Y) )./N; eeoc=sum( (dv-Y).^2 ); auc=[];
end

if( verb>0 || nargout<1 ) 
  if ( strfind(probType,'class') )
     cellSz=ceil(log10(max(conf(:))))+3;
     for i=1:L;
      if(i==1)fprintf('\nconf = ');else fprintf('       ');end;
      fprintf(['%' num2str(cellSz) 'd'],conf(i,:)); fprintf('\n');
    end;
  end
  fprintf('\nBin:\t\t');fprintf('%1.5f ',binCls);fprintf('/ %1.5f\n',mean(binCls));
  fprintf('EEOC:\t\t');fprintf('%1.5f ',eeoc);fprintf('/ %1.5f\n',mean(eeoc));
  fprintf('AUC: \t\t');fprintf('%1.5f ',auc);fprintf('/ %1.5f\n',mean(auc));
end

% Orginal simple version of teh same
% function [conf]=dv2conf(dv,Y)
% subProbs=unique(Y);L=length(subProbs);
% if ( size(dv,2) == L )
%   [Val,I]=max(decisVal');
% else
%   I=dv;
% end
% for c=1:L;
%   for j=1:L;    
%     conf(c,j)=sum(I(Y==c)==j);% / length(pos);
%   end
% end
