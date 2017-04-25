function [regMx]=mkQuadReg(regtype,sz)
% [regMx]=mkQuadReg(regtype,sz)
% Inputs:
%  regType -- [cell array of of]
%           {str}
%            'none' -- no reg
%            'eye'  -- constant diagonal, 0 elsewhere
%            'smooth' -- [-.5 1 -.5] 2nd deriv finite diff smoothing reg
%            'smootheye' -- smooth + eye
%           [sz x d] -- values to put on the [-d/2 : d/2] diagonal entries 
% Outputs:
%  regMx -- [sz x sz] quadratic regulariser matrix (sparse)
%
% N.B. to get an idea of what type of solution a particular regulariser
% perfers simply plot its eigenvectors.
% i.e. [v,d]=eig(full(mkQuadReg('eye',10))); imagesc(v);
if ( iscell(regtype) )
   regMx=[];
   for i=1:numel(regtype)
      regMxi=mkQuadReg(regtype{i},sz);
      if ( isscalar(regMxi) ) regMxi=regMxi*speye(sz); end;
      if ( isempty(regMx) ) regMx=regMxi; else regMx=regMx+regMxi; end;
   end
   
elseif ( isnumeric(regtype) )
   if ( numel(regtype)==1 )  % scalar
      regMx=regtype;
   elseif ( size(regtype,1)==size(regtype,2) && size(regtype,1)==sz ) % square
      regMx=regtype; % square is reg to use
   elseif( size(regtype,1)==sz || size(regtype,1)==1 ) 
      % non-square -- contains kernel to use.
      ker=regtype;
      regMx=spdiags(repmat(ker,sz/size(regtype,1),1),-floor(size(ker,2)/2):floor(size(ker,2)/2),sz,sz);      
      %val=sum(ker(1:ceil(end/2))); regMx([1 end])=val;% approx edge effect corr
   else
      error('regMx size wrong');
   end
elseif ( ischar(regtype) ) 
   switch lower(regtype)
    case 'none'
     regMx=0;
    case 'eye'; % perfers minimum norm solutions
     %      didx=diagIdx(size(pSigma),0); pSigma(didx(1:numel(sf0)))=+1;
     regMx=1;
    case 'smooth'; 
     % N.B. for this smoother: R ~ <dw/dx>.^2*.75*numel(w)
     ker=[-.5 1 -.5];
     regMx=spdiags(repmat(ker,sz,1),-floor(numel(ker)/2):floor(numel(ker)/2),sz,sz);
     regMx([1 end])=.5;       % (roughly) correct for edge effects
    case 'smootheye'; 
     ridge=.1;
     ker=[-.5 1+ridge -.5];
     regMx=spdiags(repmat(ker,sz,1),-floor(numel(ker)/2):floor(numel(ker)/2),sz,sz);
     regMx([1 end])=.5+ridge; % (roughly) correct for edge effects
    otherwise; error('Unrecognised sigma init type: %s',regtype);
  end
else
   error('Unrecognised regMx init type');
end
return;
%-----------------------------------------------------------------------------
function []=testCases();

mimage(mkQuadReg('eye',10),mkQuadReg([-.5 1 -.5],10),mkQuadReg(repmat([-.5 1 -.5],10,1),10),mkQuadReg({.1,'smooth'},10),mkQuadReg(randn(10,1),10))


% some interesting stuff for making regularlisers
% Smoothing regularisers
width=1;K=exp(-.5*([-50:50]/width).^2);K=K/sum(K);
smth=spdiags(repmat(K,size(tf0,1),1),-floor(numel(K)/2):floor(numel(K)/2),size(tf0,1),size(tf0,1));
reg=inv(smth);
Kreg=reg(38:62,50); serialize(full(Kreg))

Kreg=exp(-.5*(([1:100]-50)/width).^2).*(-1).^[1:100]; % A^-2
reg =spdiags(repmat(K,size(tf0,1),1),-floor(numel(K)/2):floor(numel(K)/2),size(tf0,1),size(tf0,1));
smth=reg.^-.5;
mimage(reg,smth);

% Sigma=.5
K=[ 5.07837e-11;-3.75244e-10;2.7727e-09;-2.04876e-08;1.51384e-07;-1.11859e-06;8.26529e-06;-6.10727e-05;0.00045127;-0.00333446;0.0246383;-0.181994;1.32059;-0.181994;0.0246383;-0.00333446;0.00045127;-6.10727e-05;8.26529e-06;-1.11859e-06;1.51384e-07;-2.04876e-08;2.7727e-09;-3.75244e-10;5.07837e-11 ];
% width=1
K=[ 0.00208126;-0.00343141;0.00565736;-0.00932717;0.0153767;-0.0253465;0.0417653;-0.0687516;0.11287;-0.183941;0.293789;-0.443716;0.570517;-0.443716;0.293789;-0.183941;0.11287;-0.0687516;0.0417653;-0.0253465;0.0153767;-0.00932717;0.00565736;-0.00343141;0.00208126 ];
% width=2
K=[ 0.104834;-0.118076;0.132536;-0.148143;0.164739;-0.182046;0.199634;-0.216901;0.233062;-0.247179;0.25825;-0.265345;0.267792;-0.265345;0.25825;-0.24718;0.233063;-0.216903;0.199637;-0.182048;0.164741;-0.148146;0.132539;-0.11808;0.104839 ];
% width=3
K=[ -53102.5;67944.5;-43228.5;3043.38;14871.8;-3674.64;-8695.05;2994.43;6409.47;-2302.53;-5305.05;1527.99;4765.9;-567.307;-5026.68;184.369;5419.03;-114.19;-6105.16;378.825;7574.44;-1254.51;-11804.9;5175.67;31746.1 ]
% Shrunk versions..

K=[ -0.011109   0.135335   -0.606531   1   -0.606531   0.135335   -0.011109 ];%sig=1 approx
K=[-.5 2 -.5]; % sig=1 approx
% additional roughness penalty only on temporal HPs
tfpSigma=spdiags(repmat(K,size(tf0,1),1),-floor(numel(K)/2):floor(numel(K)/2),size(tf0,1),size(tf0,1));
