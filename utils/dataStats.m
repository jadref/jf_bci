function [stats]=dataStats(X)
stats(1,:)=min(X);
stats(2,:)=mean(X);
stats(3,:)=max(X);
stats(4,:)=var(X);