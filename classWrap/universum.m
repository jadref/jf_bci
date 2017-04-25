function [wb,f,J,obj]=universum(X,Y,C,varargin)
% wrapper function for a normal binary classifier which makes it a universum classifier
% 
% [wb,f,J,obj]=universum(X,Y,C,varargin)
%
% This function wraps a normal binary classifier to make a universum
% classifier by simply duplicating every universum example and assigning 
% *both* target values -1,+1
% 
% Inputs:
%  X -- [d x N] data to classify, with d features and N examples
%  Y -- [N x 1] labels for the data to be classified
%       N.B. orginal labelling in Y is transformed such that:
%          Y==1 -> negative class
%          Y==2 -> positive class
%          Y>=2 -> universum class
%  C -- [1 x 1] regularisation parameter for the classifier
% Options:
%  alphab -- seed solution (kernel method)
%  wb     -- seed solution (direct method)
%  innerobjFn  -- [str] classification objective function to use for the inner classifier  (lr_cg)
%  kernel -- [bool] is the input X a kernel matrix ([])
% N.B. all other options are passed directly to 'innerobjFn'
% Outputs:
%  wb      - {size(X,1:end-1) 1} matrix of the feature weights and the bias {W;b}
%  f       - [Nx1] vector of decision values
%  J       - the final objective value
%  obj     - [J Ew Ed]

% gets called from cvtrainFn (line 154), 
% which gets called from cvtrainLinearClassifier (line 93)

opts = struct('alphab',[],'kernel',0,'innerObjFn','lr_cg'); % the options we care about
% parse the options, extract ones we want, ignore rest
[opts,varargin] = parseOpts(opts,varargin); 
if ( nargin<3 || isempty(C) ) C=0; end;

% HACK the re-labelling....
Y_u = Y;
if( any(Y<0) ) 
  u_idx= ~(Y==1 | Y==-1 | Y==0); % select universum trials
else
  u_idx= Y_u>2;  
  Y_u(Y==1)=-1;  Y_u(Y==2)=1; % re-label 1=-1, 2-1;
end

% duplicate the universum trials in X and Y
% Add duplicates to X
if ( opts.kernel || size(X,1)==size(X,2) ) % kernel matrix input
  opts.kernel=1;
  X_u = [X X(:,u_idx); X(u_idx,:) X(u_idx, u_idx)]; % modify kernel
else
  switch (ndims(X))
   case 2;  X_u = cat(2,X,X(:,u_idx)); % N.B. assumes X is [feat x examples]
   case 3;  X_u = cat(3,X,X(:,:,u_idx)); 
   otherwise; error('More than 3d not supported yet!');
  end  
end
% Modify the labelling so first copy has + label, and second -
Y_u = [Y_u; -ones(sum(u_idx),1)]; % concat -1 target labels for universum
Y_u(u_idx) = 1; % set original universum target labels to +1

alphab=opts.alphab;
if(~isempty(alphab)) % modify existing seed via zero-padding, call usvm
  alphab = [opts.alphab(1:end-1); zeros(sum(u_idx),1); opts.alphab(end)];
end 

% Call the unmodified inner classifier
[wb,f,J,obj] = feval(opts.innerObjFn,X_u,Y_u,C,'alphab',alphab,varargin{:});

% undo changes (i.e. duplications) to decision values and alpha+b
f = f(1:length(Y),:); % select original decision values

if( opts.kernel || ~isempty(alphab) )
  uwb=wb; % universum solution weighting
  wb = [uwb(1:length(Y));uwb(end)]; % select (non-duplicated) weights and bias
  wb(u_idx) = uwb(u_idx)+uwb(length(Y)+1:end-1);  % add in the effect of duplicated examples
end

return
function testCase()
[X,Y]=mkMultiClassTst([-1 0; 1 0; .2 .5],[400 400 50],[.3 .3; .3 .3; .2 .2]);[dim,N]=size(X);
wb=universum(X,Y)
clf;plotLinDecisFn(X,Y,wb(1:end-1),wb(end))