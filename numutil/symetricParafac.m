function [S,varargout]=symetricParafac(symDim,S,varargin)
% make an input parafac decomposition symetric in pairs of dimensions
%
% [S,varargout]=symetricParafac(symDim,S,U1,U2,...)
%
% Inputs:
%  symDim - [2 x M] set of pairs of dimensions to symetrize
%  S      - [R x 1] set of component (hyper)-singular values
%  U1     - [d1 x R] singular-vectors for the 1st dimension of the inputs
%  U2     - [d2 x R] singular-vectors for the 2nd dimension of the inputs
%  etc..
%
% Output:
%  S      - [R x 1] set of component (hyper)-singular values 
%           N.B. may be negative, c.f. negative eigen-values
%  U1     - [d1 x R] singular-vectors for the 1st dimension of the inputs
%  U2     - [d2 x R] singular-vectors for the 1st dimension of the inputs
%  etc..
if ( numel(symDim)==2 && size(symDim,1)==1 ) symDim=symDim'; end;
S=S(:); % ensure is col vector
actComp=(abs(S)>max(abs(S(:)))*1e-9); inactComp=find(~actComp); actComp=find(actComp); 
if ( isempty(actComp) ) varargout=varargin; return; end;
U=varargin;
for di=1:size(symDim,2); % project back onto the symetric sub-space for symetric dim
   d = symDim(1,di); symd=symDim(2,di); % dim and its symetric partner
   
   sgn = sign(sum(U{d}(:,actComp).*U{symd}(:,actComp))); % preserve if opposite directions
   % N.B. project to the symetric sub-space using the power method to find the best Rank 1 symetric approx
   % for the symetric equiv, i.e. u' = (xy'+yx')/2*u;
   % N.B. this approx may be *very* bad if the symetric input is inherently rank-2!!!!!
   x=U{d}(:,actComp); y=repop(sgn,'*',U{symd}(:,actComp)); u=(x+y)./4; % start from good seed
   for j=1:25;
      u = .5*(repop(x,'*',sum(y.*u))+repop(y,'*',sum(x.*u))); nrm=sqrt(sum(u.*u)); u=repop(u,'./',nrm);
   end
   %u  =repop(u,'.*',sqrt(nrm)); % make the norms right         
   expVar = sum(x.*y)./nrm; % explained variance for each component
   ok = expVar>.95; % ranks for which the rank-1 decomp is good aprox
   for ri=1:numel(ok);
      ci=actComp(ri); sgnci=sgn(ri); % only update if sym version explains most of the variance
      if ( ok(ri) || ~isempty(inactComp) )
         if ( ok(ri) ) % rank-1 approx is OK
            P=u(:,ri); D=nrm(ri); sgnci=sgn(ri);
         elseif ( ~isempty(inactComp) ) % need a rank-2 approx
            % add the new component in inact comps or at end
            ci=[ci inactComp(1)]; inactComp=inactComp(2:end); 
            %if ( isempty(inactComp) ) ci=[ci numel(S)+1]; else ci=[ci inactComp(1)]; inactComp=inactComp(2:end); end; 
            % copy the current rank into the new postion
            S(ci(2))=S(ci(1)); for dii=1:numel(U); U{d}(:,ci(2))=U{d}(:,ci(1)); end; % copy into the new rank
            % get the rank-2 approx, i.e. what to put in the 2 places
            [P,D]=eig(.5*(x(:,ri)*y(:,ri)'+y(:,ri)*x(:,ri)')); D=diag(D); [ans,si]=sort(abs(D),'descend'); 
            D=D(si(1:2)); P=P(:,si(1:2)); sgnci=sgnci.*sign(D); D=abs(D);
         end
         % update the solution
         U{d}(:,ci)=P; U{symd}(:,ci)=U{d}(:,ci);
         S(ci)=S(ci).*D(:).*sgnci; % rescale the magnitude
      end
   end
   % for non-rank-1 bits we need to decompose both eigenvectors and generate 2 new components   
end
% try to make all the strengths positive
nonSymD = setdiff(1:numel(U),symDim);
if (any(nonSymD) && any(S<0))
  Unsd=U{nonSymD(1)};
  Unsd(:,S<0)=-Unsd(:,S<0); 
  S=abs(S); U{nonSymD(1)}=Unsd;
end
varargout=U;
return;

%------------
function testCase()
S=[1 1];  U={[1 2 3;4 5 6]',[1 2 3;7 8 9]'}
U2={[] []};
[S2,U2{1},U2{2}]=symetricParafac([1 2]',S,U{:})

mimage(parafac(.5*[S S],[U{1} U{2}],[U{2} U{1}]),parafac(S2,U2),'diff',1)

S=[1 1];U={[1 2 3;4 5 9]',[1 2 3;7 13 -7]'}
[S2,U2{1},U2{2}]=symetricParafac([1 2]',S,U{:}) % rank-2 needed, but not possible
[S2,U2{1},U2{2}]=symetricParafac([1 2]',[S(:);0],U{:}) % rank-2 needed, but possible

mimage(parafac(.5*[S S],[U{1} U{2}],[U{2} U{1}]),parafac(S2,U2),'diff',1)