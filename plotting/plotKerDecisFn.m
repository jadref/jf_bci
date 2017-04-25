function []=plotKerDecisFn(X,Y,alphab,kerFn,hps,fat,wght,varargin)
%function []=plotKerDecisFn(X,Y,alphab,kerFn,hps,fat,varargin)
% Cheap and nasty function to plot kernel decision functions
if ( nargin < 7 ) wght=[]; end;
xs=[];ys=[];zs=[];
isheld=ishold;
plot2dFn(@(x) alphab(1:end-1)'*feval(kerFn,hps,fat,X,x)+alphab(end),xs,ys,zs);
hold on;
labScatPlot(X,Y,wght);
if(isheld) hold on; else hold off; end;
