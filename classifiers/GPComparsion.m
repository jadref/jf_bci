cost=1;

[X,Y]=mkToy(); Y=(Y==1)*2-1; trnIdx=true(size(Y));

clf; labScatPlot(X',Y); hold on;

% grid for eval of non-line decision functions.
[t1 t2] = meshgrid(linspace(min(X(1,:)'),max(X(1,:)),100),...
                   linspace(min(X(2,:)'),max(X(2,:)),100));
t = [t1(:) t2(:)];                        % these are the test inputs

% Fixed cost

% RUN logistic laplacian GP and extract the linear boundary
[p2 ans,ans,gpevid,best_a]=binaryLaplaceGP(log(cost), 'covLINone', 'logistic', X(:,trnIdx)', Y(trnIdx), X(:,~trnIdx)');
wgp=X(:,trnIdx)*best_a;bgp=sum(best_a);

drawLine(wgp,bgp,min(X'),max(X'),'r');

[p2,ans,ans,gpevid,best_a]=binaryLaplaceGP(log(cost),'covLINone','logistic',X(:,trnIdx)', Y(trnIdx),t);
[c,h]=contour(t1,t2,reshape(p2,size(t1)),[.4:.1:.6]);clabel(c,h);


% RUN cumGauss EP GP and extract the linear boundary
[p2 ans,ans,gpevid,best_a]=binaryEPGP(log(cost), 'covLINone', X(:,trnIdx)', Y(trnIdx), X(:,~trnIdx)');
wgp=X(:,trnIdx)*best_a;bgp=sum(best_a);

drawLine(wgp,bgp,min(X'),max(X'),'y');

[p2,ans,ans,gpevid,best_a]=binaryEPGP(log(cost),'covLINone',X(:,trnIdx)', Y(trnIdx),t);
[c,h]=contour(t1,t2,reshape(p2,size(t1)),[.4:.1:.6]);clabel(c,h);



% RUN direct logistic regression and get its boundary with the same cost
[wlr,blr,cost2,lrevid,lrpost]=evidenceLR(X(:,trnIdx)',Y(trnIdx),1,cost,1);

drawLine(wlr,blr,min(X'),max(X'),'g');

clf;labScatPlot(X',Y); hold on;

% Evid optimised cost
nlogcost=nonLinConjGrad({'binaryLaplaceGP','covLINone','logistic',X(:,trnIdx)',Y(trnIdx)},log(cost),'verb',1)
nlogcost=minimize(log(cost), 'binaryLaplaceGP',-60,'covLINone','logistic',X(:,trnIdx)', Y(trnIdx));
[p2 ans,ans,gpevid,best_a]= binaryLaplaceGP(nlogcost, 'covLINone', 'logistic', X(:,trnIdx)', Y(trnIdx), X(:,~trnIdx)');
wgp=X(:,trnIdx)*best_a;bgp=sum(best_a); cost=exp(nlogcost);

drawLine(wgp,bgp,min(X'),max(X'),'b');

% EPGP
nlogcost=minimize(log(cost), 'binaryEPGP',-60,'covLINone',X(:,trnIdx)', Y(trnIdx));
[p2 ans,ans,gpevid,best_a]= binaryEPGP(nlogcost, 'covLINone', X(:,trnIdx)', Y(trnIdx), X(:,~trnIdx)');
wgp=X(:,trnIdx)*best_a;bgp=sum(best_a); cost=exp(nlogcost);

drawLine(wgp,bgp,min(X'),max(X'),'y');


% RUN direct logistic regression and get its boundary with the same cost
[wlr,blr,ncost,lrevid,lrpost]=evidenceLR(X(:,trnIdx)',Y(trnIdx),30,cost,1);

drawLine(wlr,blr,min(X'),max(X'),'c');


