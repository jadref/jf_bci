function [dh ddh H] = estgrad(fFn, x, e, PC, verb,varargin)
% [dy ddy] = estgrad(fFn, x, e, verb, ....)
% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% Inputs:
%  fFn - function to return the feval and the derivatives of the form:
%             [y dy]=feval(fFn,x)
%  x   - [dx1] point about which to compute the perf
%  e   - [1x1] scaling for the change                                  (1e-3)
%  PC  - [dxd] pre-conditioning matrix to use, e=PC*e                    ([])
%  verb- [int] verbosity level                                           (0)
%
% Outputs:
%  dy  - [dx1] = finite-diff approx to the gradient = (f(x+e) - f(x-e))/2*|e|
%  ddy - [dx1] = finite-diff approx to the hessian  = (f(x+e) - 2f(x) + f(x-e))/(|e|*|e|)
if ( nargin < 3 || isempty(e) ) e=1e-3; end;
if ( nargin < 4 || isempty(PC) ) PC=1; end;
if ( nargin < 5 || isempty(verb) ) verb=1; end;
if ( iscell(fFn) ) 
  y=feval(fFn{1},x,fFn{2:end},varargin{:}); 
else 
  y=feval(fFn,x,varargin{:});  
end;
if ( numel(y)>size(y,1) )
  dh =zeros([size(y,1),size(x)]);
else
  dh =zeros([size(y,1),size(x)]);
end
ddh=zeros(size(dh));
yp =zeros(size(dh));
yn =zeros(size(dh));
if ( nargout>2 ) H=zeros([size(y,1),numel(x),numel(x)]); end;
tx = x;
if ( ~isempty(PC) ) dx=zeros(size(x)); end;
for j = 1:numel(x);
   if( verb>0 ) textprogressbar(j,numel(x)); end;
	
	% positive movement along dimension j
	if( ~isempty(PC) ) 
	  dx(j) = e;
	  tx    = x    + PC*dx;
	else
     tx(j) = x(j) + e;                               % perturb a single dimension	  
	end
   if ( iscell(fFn) ) 
      yp(:,j)=feval(fFn{1},tx,fFn{2:end},varargin{:}); 
   else 
      yp(:,j)=feval(fFn,tx,varargin{:});  
   end;
	
	% negative movement along dimension j
	if( ~isempty(PC) ) 
	  dx(j) = e;
	  tx    = x    - PC*dx;
	else
     tx(j) = x(j) - e;                               % perturb a single dimension	  
	end
   if ( iscell(fFn) ) 
      yn(:,j)=feval(fFn{1},tx,fFn{2:end},varargin{:}); 
   else 
      yn(:,j)=feval(fFn,tx,varargin{:});  
   end;
	
	% record the gradient and diag hessian estimates
   dh(:,j) = (yp(:,j) - yn(:,j))/(2*e);
   ddh(:,j)= (yp(:,j) - 2*y(:) + yn(:,j))/(e*e);     % diag hessian est

	if ( nargout>2 ) % compute the full hessian estimate
	  H(:,j,j)=ddh(:,j);
	  for i=1:j-1;
		 if( ~isempty(PC) ) 
			dx(i) = e;
			tx    = x    + PC*dx;
		 else
			tx(i) = x(i) + e;
		 end
		 if ( iscell(fFn) ) 
			yij=feval(fFn{1},tx,fFn{2:end},varargin{:}); 
		 else 
			yij=feval(fFn,tx,varargin{:});  
		 end;			 
		 H(:,i,j) = ((yij-yp(:,j))/e - (yp(:,i)-y)/e)/e; % Hij
		 H(:,j,i) = ((yij-yp(:,i))/e - (yp(:,j)-y)/e)/e; % Hij
		 
		 if ( ~isempty(PC) ) % reset the test point
			tx = x; dx(i)=0; 
		 else 
			tx(i) = x(i); 
		 end    
	  end
	end
   if ( ~isempty(PC) ) % reset the test point
	  tx = x; dx(j)=0; 
	else 
	  tx(j) = x(j); 
	end    
end
return;

function testCase();
[X,Y]=mkMultiClassTst([-1 0;1 0;0 1;0 -1],[400 400 400 400],[.2 .2],'gaus',[-1 1 -1 1]);[dim,N]=size(X);

wb=randn(size(X,1)+1,1);
wb=zeros(size(X,1)+1,1);
wb=[X*Y;0]./sum(abs(Y)); % prototype

oX= X;
X = repop(X,'*',[100000;ones(size(X,1)-1,1)]);
[dh,ddh]=estgrad(@(wb) rlr(X,Y,0,wb),wb);
[dh,ddh,H]=estgrad(@(wb) rlr(X,Y,0,wb),wb);
max(abs(ddh))./min(abs(ddh)),condest(shiftdim(H))

% validate the hessian estimates
Htrue=X*X';
[dh,ddh,H]=estgrad(@(x) x'*Htrue*x,wb(1:end-1)); % pure quadratic estimate
clf;imagesc('cdata',Htrue*2-shiftdim(H)); % N.B. don't forget the factor 2
max(abs(ddh))./min(abs(ddh)),condest(shiftdim(H))
% with a diag pre-conditioner
[dh,ddh,H]=estgrad(@(x) x'*Htrue*x,wb(1:end-1),[],diag(sqrt(1./diag(Htrue)))); 
max(abs(ddh))./min(abs(ddh)),condest(shiftdim(H))


% Pre-conditioners from the lr_cg code
C=1;
% non-PC results
[dh,ddh,H]=estgrad(@(wb) rlr(X,Y,C,wb),wb);
max(abs(ddh))./min(abs(ddh)),condest(shiftdim(H))

% PC results
wght=.25/2; % fixed
g  =1./(1+exp(-Y(:)'.*(wb(1:end-1)'*X+wb(end))));wght=g.*(1-g); % adaptive
if ( numel(wght)==1 ) wPC=1./sum(X.*X,2)/wght;            bPC=1./(size(X,2).*wght);
else                  wPC=1./sum(X.*repop(wght,'*',X),2); bPC=1./sum(wght);
end
%N.B. as we transform X we use the sqrt of the pre-conditioner to compute the updated condition number!
[Mdh,Mddh,MH]=estgrad(@(wb) rlr(X,Y,C,wb),wb,[],diag(sqrt([wPC;bPC])));
max(abs(Mddh))./min(abs(Mddh)),condest(shiftdim(MH))

[U,s]=eig(shiftdim(MH));s=diag(s);

lr_cg(X,Y,1,'wPC',1,'bPC',1); % no PC
lr_cg(X,Y,1,'PCmethod','wb0');% zeros PC
lr_cg(X,Y,1,'PCmethod','wb'); % seed solution
