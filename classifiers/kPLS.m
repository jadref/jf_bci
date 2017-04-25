function [alpha,transMx,beta,tau] = kerPLSTrain(K,Y,plsDim,trainType,verb)
%function [alpha,transMx,beta,tau] = kerPLSTrain(K,Y,plsDim,trainType,verb)
%
% Performs dual (kernel) PLS discrimination
%
%INPUTS
% K - the training kernel matrix (N x N)
% Y - Matrix of desired outputs, either as labels list, or matrix of
%     outputs. (Nx1) or (NxL) where L is the # label types.
% plsDim - the number of PLS components to take
% trainType -- type of training, 'class','regress'.  When using
%               'class' Y must contain a discrete set of labels.  If
%               using 'regress' then Y is treated as a real valued output
%               vector. (class)
%
%OUTPUTs
% alpha - the dual vectors corresponding to the learning PLS regressor,
%         a test output is given by: \sum_i K(x_i,x)*alpha_i = K(x)'*alpha
% transMx - the dual coefficients corresponding to the PLS feature
%           directions,  A mapped feature vector is given by:
%             \sum_i K(x_i,x)*transMx_i = K(x)'*transMx
%
% Jason Farquhar 08/02/05
% The basic algorithm is taken from:
%   "Kernel Methods for Pattern Analysis" JS-T and N. Cristianini (P187)
%   based on the dualpls.m code from: www.kernel-methods.net
% 
% 22/02/05 -- Modified to return the feature mappings also
% 
% Example output usage:
%  dv=tstKer*alpha; [v,cv]=max(dv,[],2); cr=sum(cv==tstY)/length(tstY);
%

% K is an N x N kernel matrix
% Y is N x m containing the corresponding output vectors
% T gives the number of iterations to be performed
N=size(K,1);
if ( nargin < 3 ) help kerPLSTrain; return; end;
if ( nargin < 4 | isempty(trainType) ) trainType='class'; end;
if ( nargin < 5 | isempty(verb) ) verb=0; end;
% convert Y to matrix of indicators if wanted.
if ( strcmp(trainType,'class') ) % classifier training
  if ( size(Y, 1) == 1 | size(Y, 2) == 1 )   % If form [1 2 2 3 3 1 2 ...]
    L=unique(Y);
    if ( length(L) == 2 ) % binary problem.
      %N.B. unique sorts entries so if 1,(-1or0) 1 comes last
      Ytst = Y==L(end);
    else 
      for i=1:length(L);
        Ytst(:,i) = Y==L(i); % Changing to matrix form of size [N x L]
      end
    end
    Y=single(Ytst);%Y(Y==0)=-1;     % make non-class have target of -1....???
  else
    L=1:size(Y,2);
  end
elseif ( strcmp(trainType,'regress') ) % regression training
  L=1:size(Y,2);  % L is number of output dimensions.
else
  error('Unknown type of training...');
end

trainY=0;
KK = K; YY = Y;
for i=1:plsDim
  if ( verb ) fprintf('Dimension %d)   ',i); end;

  YYK = YY*YY'*KK;
  beta(:,i) = YY(:,1)/norm(YY(:,1));
  if size(YY,2) > 1, % only loop if dimension greater than 1
     bold = beta(:,i) + 1;
     while norm(beta(:,i) - bold) > 0.001,
        bold = beta(:,i);
        tbeta = YYK*beta(:,i);
        beta(:,i) = tbeta/norm(tbeta);
     end
  end
  tau(:,i) = KK*beta(:,i);
  val(i) = tau(:,i)'*tau(:,i);
  c(:,i) = YY'*tau(:,i)/val(i);
  w = KK*tau(:,i)/val(i);
  if (verb) fprintf('Deflating X...'); end
  KK = KK -tau(:,i)*w' -w*tau(:,i)' +tau(:,i)*tau(:,i)'*(tau(:,i)'*w)/val(i);
  if ( verb ) fprintf('done\n'); end;
  if ( verb ) 
     trainY = trainY + tau(:,i)*c(:,i)';
     trainerror = norm(Y - trainY,'fro')/sqrt(N);
     fprintf('Training error %f\n',trainerror);
  end
  
  YY = YY - tau(:,i)*c(:,i)';
end

% Compute the feature space feature directions?
% Undo the deflations to map feature directions back into the orginal
% feature space, so we can use them directly without incrementaly deflating
% first
% N.B. the main effect of this is to ensure that the mapped training set has
% diagonal covariance, i.e. the mapped features are uncorrelated!
% However as it mixes the beta's it does so by making the feature directions
% non-orthonogal... you pays your money and makes your choice!
% BODGE: N.B. to do this properly we need: 
%  beta*(T'*K*beta)^-1*diag(tau'*tau)*diag(a)^-1
% but since diag(a) isn't available we have to just use ignore it, this
% shouldn't be too much of a problem as a_i is just a re-scaling of the
% feature direction u_i anyway?
transMx = ( beta/(tau'*K*beta)*diag(val) ) ;
% Regression coefficients for new data, fast but approx way:
% alpha = transMx*diag(1/val)*tau'*Y;
% This is the correct version because the a's cancel out. 
alpha = beta * ((tau'*K*beta)\tau')*Y;
