function []=parafacPlot(P,di,varargin)
% plot a parafac tensor decomposition
%
%  parafacPlot(P,di,...)
%
% Inputs:
%  P  -- {S U1 U2 U3 ...} parafac decomposition with
%         S - [nComp x 1] component weights
%         U1- [D1 x nComp] component loadings over 1st dimension
%         U2- [D1 x nComp] component loadings over 2nd dimension
%         etc.
%  di  -- dimInfo struct describing the dimensions (see mkDimInfo)
% Options: 
%  layout   -- [nDim x 4] sub-rectangle of the figure window to use for each
%               dimensions plots.  Rect=[x y w h]
%  dispType -- {nDim x 1 str} plot type to use for each dimension.  One of:
%              'plot','mcplot','imagesc','topoplot'
%  labels -- {1 x nComp} labels for the components
%  hlim --
%  legend -- [nDim x 1 bool] put legend on this dimensions plots?
%  electPos -- [2 x size(X,topo)]
opts=struct('layout',[],'pSize',[],'dispType',[],'labels',[],'hlim',.6,'legend',[],'plotOpts',{{}},'electPos',[]);
opts=parseOpts(opts,varargin);
nComp=numel(P{1});
nDim =numel(P)-1;

if ( isfield(di,'di') ) di=di.di; end;

if ( isempty(opts.legend) ) opts.legend=false(nDim,1); opts.legend(end)=true; end;

% work out the part of the figure we're going to use for each dimension
layout=opts.layout;
pSize=opts.pSize; pSize(end+1:nDim)=1;
if ( isempty(layout) || numel(layout)==2 )
  nPlots=sum(pSize); 
  % use given layout
  if ( numel(layout)==2 ) pw=layout(1);          ph=layout(2); layout=[]; 
  % split the dimensions equally over the figure
  else                    pw=ceil(sqrt(nPlots)); ph=ceil(nPlots/pw);
  end;
  layout(:,1)=mod([0 cumsum(pSize(1:end-1))],pw)/pw;
  layout(:,2)=1-1/ph-floor([0 cumsum(pSize(1:end-1))]/pw)/ph;
  layout(:,3)=pSize./pw;
  layout(:,4)=1./ph;
end

% loop over the dimensions making the plots
dispType=opts.dispType;
if ( isempty(dispType) ) dispType='plot'; end;
if ( isstr(dispType) ) dispType={dispType}; end;
plotOpts={};
for dimi=1:nDim;
  dispTypei = dispType{min(end,dimi)};
  comps=P{dimi+1}(:,1:numel(P{1}));
  xs=1:size(comps,1);
  if ( isstruct(di) && numel(di)>=dimi && isfield(di(dimi),'vals') ) xs=di(dimi).vals(1:size(comps,1)); end;
  xlabs=[]; if ( iscell(xs) ) xlabs=xs; xs=1:numel(xs); end;
  if ( numel(opts.plotOpts)>=dimi && iscell(opts.plotOpts{1}) ) plotOpts=opts.plotOpts{dimi}; end; 
  if ( strcmpi(dispTypei,'topoplot') ) % special case, 1 plot per-component
    % build a 2nd layout for the sub-plots
    spw=ceil(sqrt(size(comps,2)*1.5)); sph=ceil(size(comps,2)/spw);
    splayout(:,1)=mod(0:size(comps,2)-1,spw)/spw; 
    tmp=(ones(spw,1)*(0:sph-1))/sph; splayout(:,2)=tmp(end:-1:1+(end-size(comps,2)));
    splayout(:,3)=1./spw;
    splayout(:,4)=1./sph;
    % transform into the sub-box where they should lie
    splayout(:,1)=splayout(:,1)*layout(dimi,3)*.9+layout(dimi,1); % N.B. leave gap for the colorbar
    splayout(:,2)=splayout(:,2)*layout(dimi,4)+layout(dimi,2);
    splayout(:,3)=splayout(:,3)*layout(dimi,3)*.9;
    splayout(:,4)=splayout(:,4)*layout(dimi,4);    % N.B. leave gap for the colorbar
    % make the plots
    if ( ~isempty(opts.electPos) ) 
      electPos=opts.electPos;
    elseif ( isstruct(di) && numel(di)>dimi && isfield(di(dimi),'extra') ) 
      electPos=[di(dimi).extra.pos2d];
    else
      electPos=[];
    end
    for ci=1:size(comps,2);
      hdls(ci)=axes('outerposition',splayout(ci,:));      
      jplot(electPos,comps(:,ci),plotOpts{:},'clim','cent0','colorbar',0); axis equal; hold on;
      if( ~isempty(opts.labels) ) title(opts.labels{min(end,ci)}); end;      
    end
    % Add the colorbar
    pos=[layout(dimi,1)+.9*layout(dimi,3) layout(dimi,2) max(.06,.1*layout(dimi,3)) layout(dimi,4)];
    cb=colorbar('peer',hdls(end),'outerposition',pos);    
  else
    % make the axes
    axes('outerposition',layout(dimi,:));
    switch lower(dispTypei)
     case 'plot';   plot(xs,comps,plotOpts{:});  
     case 'mcplot'; mcplot(xs,comps,plotOpts{:}); ylabel('component');
     case 'imagesc';imagesc(xs,1:size(comps,2),comps,plotOpts{:});
    end
    if ( ~isempty(xlabs) )
       tickIdx=unique(floor(get(gca,'XTick')));
       tickIdx(tickIdx<1)=[]; tickIdx(tickIdx>numel(xs))=[];
       set(gca,'XTick',tickIdx,'XTickLabel',xlabs(tickIdx));              
    end
    if ( isstruct(di) ) 
      lab=di(dimi).name; 
      if ( ~isempty(di(dimi).units) && isstr(di(dimi).units) ) lab=[lab sprintf('(%s)',di(dimi).units)]; end;
    elseif ( iscell(di) ) 
      lab=di{dimi};
    end
    xlabel(lab);
    if ( opts.legend(min(end,dimi)) ) 
      for ci=1:size(comps,2); legtxt{ci}=sprintf('Comp %d (%g)',ci,P{1}(ci)); end; 
      if ( size(comps,2) > 1 ) legend(legtxt); end;
    end
  end
end
return;
function testCase()
P={[1 2 4] randn(10,3) randn(5,3)};
di=mkDimInfo([10 5],{'ch','time'});
clf;parafacPlot(P,di)
P={[1 2 4] randn(10,3) randn(5,3) randn(50,3) randn(40,3)};
di=mkDimInfo([10 5 50 40],{'ch' 'time' 'window' 'epoch'});
clf;parafacPlot(P,di)