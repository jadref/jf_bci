function [wb,f]=kerPerceptron(wb,K,Y,C,varargin);
% function [wb]=kerPerceptron(wb,K,Y,C,varargin);
% 
% The kernel perceptron classification algorithm
% C is a softness parameter which we treat as added to the diag of K
% we also add 1 to K to simulate a bias term
opts=struct('maxIter',inf,'maxEval',inf,'tol',0,'tol0',0,...
            'verb',0,'X','wght',[],'perm',[]);
[opts,varargin]=parseOpts(opts,varargin);
if(~isempty(varargin))error('Unrecognised Option(s)'); end;

% The code
[dim N]=size(K);
f=zeros(N,1);
if ( ~isempty(opts.wght) ) Y=Y.*wght; end;
if ( isempty(opts.prm) ) prm=1:N; end;
wb0=wb;
for iter=1:opts.maxIter; % num times through the data
   owb=wb;
   updated=0;
   for i=prm;  % loop over points (in potentially permuted order)
      % N.B. K_eff = K + 1 + eye*C(1)
      %      so f=wb*K_eff = wb'*K + sum(wb) + wb*(C(1)) 
      if ( Y(i)*f(i) < 0 ) 
         updated = 1;
         wb(i)   = wb(i) + Y(i);   
         %wb(end) = wb(end)+Y(i); % bias is just sum alphas
         f       = f + Y(i)*K(i,:) + Y(i);
         f(i)    = f(i) + C(1)*Y(i);
      end
   end
   if ( updated==0 | norm(wb-owb)<opts.tol | norm(wb-wb0)<opts.tol0) 
      break ; 
   end;
end
wb(end)=sum(wb);
return;

%----------- option parsing
function [opts,inOpts]=parseOpts(opts,inOpts)
i=1;unrec=[];
while i<=numel(inOpts);  % refined option parser with structure flatten
   if ( iscell(inOpts{i}) ) % flatten cells
      inOpts={inOpts{1:i} inOpts{i}{:} inOpts{i+1:end}};
   elseif ( isstruct(inOpts{i}) )% flatten structures
      cellver=[fieldnames(inOpts{i})'; struct2cell(inOpts{i})'];
      inOpts={inOpts{1:i} cellver{:} inOpts{i+1:end} };
   elseif( isfield(opts,varargin{i}) ) 
      opts{j}.(varargin{i})=varargin{i+1}; i=i+1;
   else
      unrec(end+1)=i;
   end
   i=i+1;
end
inOpts=inOpts(unrec);
return;