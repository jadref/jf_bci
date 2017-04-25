function []=fact3Dplot(cw,sw,ww,cwxys,swxs,wwxs,varargin)
% plot a 3-d factored model
%
%  fact3DPlot(cw,sw,ww,cwxys,swxs,wwxs,layout,plottype)
%
% Inputs:
%  cw    -- [nCh x M] channel weights
%  sw    -- [nFreq x M] spectral weights
%  ww    -- [nWin x M] window (time) weights
%  cwxys -- [2 x nCh] channel locations
%  swxs  -- [1 x nFreq] spectral weight positions
%  wwxs  -- [1 x nWin] window weight positions
%  layout -- [w h] layout for the plots
%  disptype -- [str] one of 'plot' 'mcplot'
if ( nargin < 4 ) cwxys=[]; end;
if ( nargin < 5 ) swxs=[]; end;
if ( nargin < 6 ) wwxs=[]; end;
opts=struct('layout',[],'disptype',[],'labels',[],'hlim',.6,'legend',1,'axeslabels',{{'frequency (hz)' 'time (ms)'}});
opts=parseOpts(opts,varargin);
nfilt=size(cw,2);
if ( isempty(sw) && isempty(ww) ) hlim=1; wlim=.9; else hlim=opts.hlim; wlim=.9; end;
layout=opts.layout; disptype=opts.disptype;
if ( isempty(layout) ) 
  wpos=get(gcf,'position'); 
  aspect=(wpos(3)./(wpos(4)*hlim));
  w=ceil(sqrt(nfilt*aspect)); 
  h=ceil(nfilt/w); w=ceil(nfilt/h); % equally spread the components
else w=layout(1); h=layout(2); 
end
for i=1:nfilt;
   hdls(i)=axes('outerposition',[mod(i-1,w)/w*wlim 1-ceil(i/w)/h*hlim wlim/w hlim/h]);
   if ( ~isempty(cwxys) )
      jplot(cwxys,cw(:,i));%,'labels',sprintf('%d',i));
      axis equal
   else
      plot(cw(:,i));title(sprintf('%d',i));
   end
   if( ~isempty(opts.labels) ) title(opts.labels{min(end,i)}); end;
end
clims=[-1 1]*max(abs(cw(:)));set(hdls,'CLim',clims);

% Add the colorbar
pos=[wlim 1-hlim 1-wlim-.03 hlim];
cb=colorbar('peer',hdls(end),'outerposition',pos);

pidx=w*(h-1);
if (~isempty(sw))
   if ( isempty(ww) ) 
      axes('outerposition',[0 0.03 1 1-hlim]);
   else
      axes('outerposition',[0 0.03 .5 1-hlim]);
   end
   if ( isempty(swxs) ) swxs=[0:size(sw,1)-1]'; end
   if ( strcmp(lower(disptype),'mcplot') ) 
     mcplot(swxs,squeeze(sw),'center',0);
     ylabel('component');
   else
     if (isnumeric(swxs))
       plot(swxs,squeeze(sw));       
     else
       plot(ww);
       tickIdx=unique(floor(get(gca,'XTick')));
       tickIdx(tickIdx<1)=[]; tickIdx(tickIdx>numel(swxs))=[];
       set(gca,'XTick',tickIdx,'XTickLabel',swxs(tickIdx));       
     end
     mabs=[min([sw(:);0]) max(abs(sw(:)))];
     if(mabs(1)<-1e-4*mabs(2)) mabs(1)=-mabs(2); end;
     axis([0 1 mabs],'autox')   
   end
   grid on;
   if ( ~isempty(opts.axeslabels) )
     xlabel(opts.axeslabels{1});%title([z.fulldir(end-10:end)]);
   end
   if ( opts.legend ) 
     for i=1:size(sw,2); legtxt{i}=sprintf('Comp %d',i); end; 
     if ( size(sw,2) > 1 ) legend(legtxt); end;
   end
end
if (~isempty(ww))
   if ( isempty(sw) ) 
      axes('outerposition',[0 0.03 1 1-hlim]);
   else
      axes('outerposition',[.5 0.03 .5 1-hlim]);
   end
   if ( isempty(wwxs) ) wwxs=[0:size(ww,1)-1]; end
   if ( strcmp(lower(disptype),'mcplot') ) 
     mcplot(wwxs,ww,'center',0);
   else
     if ( isnumeric(wwxs) )
       plot(wwxs,ww);
     else
       plot(ww);
       tickIdx=unique(floor(get(gca,'XTick')));
       tickIdx(tickIdx<1)=[]; tickIdx(tickIdx>numel(wwxs))=[];
       set(gca,'XTick',tickIdx,'XTickLabel',wwxs(tickIdx));       
     end
     mabs=[min([ww(:);0]) max(abs(ww(:)))];
     if(mabs(1)<-1e-4*mabs(2)) mabs(1)=-mabs(2); end;
     axis([0 1 mabs],'autox')   
   end
   grid on;
   if ( ~isempty(opts.axeslabels) )
     xlabel(opts.axeslabels{2});%title([z.fulldir(end-10:end)]);
   end
   if ( opts.legend ) 
     for i=1:size(ww,2); legtxt{i}=sprintf('Comp %d',i); end; 
     if ( size(ww,2) > 1 ) legend(legtxt); end;
   end
 end
%suptitle(z.fulldir(end-10:end)); % global title
drawnow;
