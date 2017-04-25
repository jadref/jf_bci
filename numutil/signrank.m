function [z_delta]=signrank(x,y,eps)
% Wilcoxon signed rank test
%
% Implemtation based on <http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>
if ( nargin<2 || isempty(y) || numel(y)==1 ) 
  if ( nargin>1 && numel(y)==1 ) eps=y; y=[]; end;
  if ( size(x,2)==2 )  y=x(:,2); x=x(:,1);
  elseif ( size(x,1)==2 ) y=x(2,:)'; x=x(1,:)'; 
  end;
end
if ( nargin<3 || isempty(eps) ) eps=0; end;
dif=(x-y); % diff
dif=dif(abs(dif)>eps); % remove draws
[ans,si]=sort(abs(dif),'ascend'); % compute ranking
% XXX -- should correct rank for ties
W = abs(sign(dif(si))'*(1:numel(dif))');
sigma_w = sqrt(numel(dif)*(numel(dif)+1)*(2*numel(dif)+1)/6);
% find the z-score for this statistic value
h=.01;xs=[0:h:10]; normcdf=.5+cumsum(exp(-.5*(xs).^2).*h./sqrt(2*pi));normcdf=normcdf+min(0,(1-normcdf(end)));
z_delta = normcdf(find(xs>(W-.5)./sigma_w,1));
