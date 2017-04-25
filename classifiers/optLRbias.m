function [b]=optLRbias(Y,dv)
% compute the optimal chang in the bias from a given classifiers output
%
%  [b]=optbias(Y,dv)
ind=(Y~=0); 
if ( size(Y,2)==1 ) % ignore 0 labelled points
   Y=repmat(Y(ind),1,size(dv,2)); dv=dv(ind,:);
else 
   Y=Y(ind); dv=dv(ind); 
end
      
b=zeros(1,size(dv,2)); db=repmat(inf,size(b)); Ed=repmat(inf,size(b));
for iter=1:10;
   oEd=Ed; ob=b; odb=abs(db);
   % compute the updated solution
   dvb   = repop(dv,'+',b);
   g     = 1./(1+exp(-(Y.*dvb))); g=max(g,eps); % stop log 0
   Yerr  = Y.*(1-g);
   Ed    = -sum(log(g)); % the true objective funtion value
   db    = -sum(Yerr);
   ddb   = sum(g.*(1-g));

   % convergence test
   if ( iter==1 ) db0=norm(db); Ed0=Ed; end;
   if ( Ed > oEd ) b=ob; break; end; % don't take value increase steps
   if ( norm(db) < eps || norm(odb-abs(db))./db0 < eps ) break; end; % relative convergence test

   %if ( 1 )
   %   fprintf('%d) b=[%s] Ed=[%s] db=%s\n',iter,sprintf('%0.5g,',b),sprintf('%0.5g,',Ed),sprintf('%0.5g,',db));
   %end
   
   % now do the newton step
   b  = b - db./ddb;
end

return;
%-----------------------------------------------------------------
function testCase()
Y = sign(randn(1000,1));
dv= Y+randn(size(Y))+2; % simulate LR style classifier with bias shift
mub=optLRbias(Y,dv);

