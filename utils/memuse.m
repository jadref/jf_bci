function [mem] = memuse
[s,w]=system(['ps -ovsz=,rsz=,rss= -p ' sprintf('%d',getpid)]);
mem=sscanf(w,'%d');
return;