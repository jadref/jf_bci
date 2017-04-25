function [z]=jf_plot(z,varargin)
% ERP visualation -- plot the mean of each class
%
% Options:
%   dim -- dim of z to use for the sub-plots
%   subIdx -- {cell ndims(z.X)x1} sub indicies of each dimension to plot
%   eegonly -- [bool] only plot the eeg channels                (1)
%   verb    -- [int] verbosity level for logging
%   saveaspdf -- [bool] or 'str' save a copy of the image to subject directory?  (0)
opts=struct('dim',[],'subIdx',[],'eegonly',1,'verb',0,'saveaspdf',0);
[opts,varargin]=parseOpts(opts,varargin);

if ( isempty(opts.dim) ) opts.dim=n2d(z.di,'ch',0,0); if ( ~opts.dim ) opts.dim=ndims(z.X); end; end;

% sub-set plot if wanted
subIdx=opts.subIdx;
if ( isempty(subIdx) ) 
   idx=repmat({':'},1,ndims(z.X));
   zdi=z.di;
else
   chD=n2d(z.di,'ch',0,0);
   if ( chD~=0 && opts.eegonly && isfield(z.di(chD).extra,'iseeg') ) % && isempty(subIdx{n2d(z.di,'ch')}) )  
      iseeg=[z.di(chD).extra.iseeg];
      if ( any(iseeg) ) 
         if ( isempty(subIdx{chD}) ) subIdx{chD}=iseeg; else subIdx{chD}=intersect(subIdx{chD},find(iseeg)); end;
      end;
   end
   subIdx(ndims(z.X)+1:end)=[];
   [idx,zdi]=subsrefDimInfo(z.di,'dim',1:numel(subIdx),'idx',subIdx);
end

% do the ploting
Xidx=z.X(idx{:});
subPlotD=n2d(zdi,opts.dim);
plotType='plot';
if ( n2d(zdi,'time',0,0)>2 || size(Xidx,3)>50 && size(Xidx,max(1,subPlotD+1)) <50 ) 
  plotType=[plotType 't']; 
end;
image3ddi(Xidx,zdi,n2d(zdi,opts.dim),'ticklabs','sw','disptype',plotType,varargin{:});

if ( ~isempty(opts.saveaspdf) && ~isequal(opts.saveaspdf,0) )
   if ( isfield(z,'rootdir') )
      figDir=fullfile(z.rootdir,z.expt,z.subj,z.session,'figs');
      if( ~exist(figDir,'dir') ) mkdir(figDir); end
      if ( ischar(opts.saveaspdf) ) 
         plotType = opts.saveaspdf; 
      else 
         plotType=sprintf('%3d',floor(rand(1)*1000)); 
      end;
      try; 
       fn=fullfile(z.rootdir,z.expt,z.subj,z.session,'figs',sprintf('%s_%s_%s',z.subj,z.label,plotType));
         fprintf('saving plot to: %s\n',fn);
         saveaspdf(fn); 
      catch; fprintf('save failed!\n'); end   
   end
end

return;
