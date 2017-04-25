function [vs,cors]=dualcca(Ks,gamma,varargin)

%function [vs,cors]=dualcca(Ks,gamma,varargin)
%
% Performs (multiway) CCA between the kernels in the different cells of Ks.
% The number of components computed can be specified in varargin.
% gamma is the regularization parameter.
% The output vs is a cell array, containing the part of the CCA vectors for
% the i-th kernel matrix in the i-th cell.
%
%INPUTS
% Ks = a cell array containing the kernel matrices
% gamma = the regularization parameter
% varargin = an optional argument specifying the number of components;
%            when this is not specified, all components are given
%
%OUTPUTS
% vs = a cell array containing the dual coordinates of the canonical directions
% cors = the corresponding correlations
%
%
%For more info, see www.kernel-methods.net
%
%Author: Tijl De Bie, february 2004


n=size(Ks{1},1);
m=length(Ks);

if length(varargin)>=0
    ncomp=varargin{1};
else
    ncomp=k*n;
end

% Generate LH
VK=zeros(n*m,n);
for i=1:m
    VK((i-1)*n+1:i*n,:)=Ks{i};
end
LH=VK*VK';
for i=1:m
    LH((i-1)*n+1:i*n,(i-1)*n+1:i*n)=0;
end

% Generate RH
RH=zeros(n*m,n*m);
for i=1:m
    RH((i-1)*n+1:i*n,(i-1)*n+1:i*n)=(Ks{i}+gamma*eye(n))*Ks{i}+1e-6*eye(n);
end
RH=(RH+RH')/2;

% Compute the generalized eigenvectors
[Vs,cors]=eigs(LH,RH,ncomp,'LA');
cors=diag(cors);

for i=1:m
    vs{i}=Vs((i-1)*n+1:i*n,:);
end
