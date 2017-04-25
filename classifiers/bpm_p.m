function [wb,J]=bpm_p(K,Y,C,varargin)
% function []=bpm_p()
%
% Multiple perceptron training version of the bayes point machine
opts=struct('perm',10,'maxIter',inf,'maxEval',inf,'tol',0,'tol0',0,...
            'verb',0,'X','wght',[],'alphab',[]);
[opts,varargin]=parseOpts(opts,varargin);
if(~isempty(varargin))error('Unrecognised Option(s)'); end;

[dim N]=size(K);
f=zeros(N,1);
if ( ~isempty(opts.wght) ) Y=Y.*wght; end;
if ( numel(opts.perm)==1 ) % compute the required number of permutations
   for i=1:opts.perm;
      prms(:,i)=randperm(N);
   end
else
   prms=opts.perm; % use the input permutations
end;
if ( isempty(opts.alphab) ) wb=zeros(N+1,1); else wb=opts.alphab; end;


% Run the kernel perceptron algorithm numSamp times
wbs=repmat(wb,[1,size(prms,2)]); % set the initial solns
for samp=1:size(prms,2);
   
   % get the permutation to use
   prm=prms(:,i);

   % Run the kernel perceptron algorithm on this permutation
   wb0=wbs(:,samp);
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
            f        = f + Y(i)*K(i,:) + Y(i);
            f(i)     = f(i) + C(1)*Y(i);
         end
      end
      if ( updated==0 | norm(wb(:)-owb)<opts.tol | norm(wb(:)-wb0)<opts.tol0) 
         break ; 
      end;
   end
   wb(end)=sum(wb);

   wbs(:,samp)=wb;   
end

% The returned solution is simply the average of the found ones.
wb=mean(wbs,2);