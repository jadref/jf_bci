function [loss]=dv2loss(Y,dv,dim,losstype)
% convert decision values to loss
if ( nargin<4 ) losstype=''; end;
if ( nargin<3 ) dim=[]; end;
conf=dv2conf(Y,dv,dim);
loss=conf2loss(conf,1,losstype);
return;
