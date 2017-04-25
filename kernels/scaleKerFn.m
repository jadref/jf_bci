function [K,dK,ddK]=scaleKerFn(hp,fat,X,Z);
% a simple scaling feature function which just re-scales the input
% dimensions by the given numbers
if ( isstruct(fat) ) hp=hp(fat.scale); end;
if ( numel(hp) ~= size(X,1) ) error('One scale parameter per dim'); end;

sf=exp(hp); sf=sf(:); % hp contains log factors
if ( nargin >3 && ~isempty(Z) )  
   phiX=repop(X,sf.^2,'.*'); phiZ=Z;% phiZ=tprod(Z,hp(:),[1 2 3],[1]);
else
   phiX=repop(X,sf,'.*');    Z=[]; phiZ=[]; 
end
K =tprod(phiX,phiZ,[-(1:ndims(phiX)-1) 1],[-(1:ndims(phiX)-1) 2]);
if ( nargout > 1 ) 
   dK=2*repop(tprod(X,Z,[3 -(2:ndims(X)-1) 1],[3 -(2:ndims(X)-1) 2]),...
              shiftdim(sf.^2,-2),'.*');
end;
if ( nargout > 2 ) ddK=zeros(size(dK)); end;

%----------------------------------------------------------------------------
function testCase()
X=randn(1,100);hp=.3;
X=randn(2,100);hp=[.3,.5];
[K,dKdhp,ddKdhp]=scaleKerFn(hp,[],X);
jf_checkgrad(@(x) scaleKerFn(x,[],X),hp,1e-5,0,1);
