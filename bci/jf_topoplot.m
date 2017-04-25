function [z]=jf_topoplot(z,varargin);
% topoplot z
%
% Options:
%  dim
%  subIdx
%  coords
%  eegonly
opts=struct('dim','ch','subIdx',[],'coords',[],'eegonly',1,'verb',0);
jplotopts=struct('clim','cent0');
[opts,jplotopts,varargin]=parseOpts({opts,jplotopts},varargin);
    
chD=n2d(z,opts.dim);

% get electrode positions
coords=opts.coords;
if ( isempty(coords) ) 
  if ( isfield(z.di(chD).extra,'pos2d') ) 
    coords=[z.di(chD).extra.pos2d];
  else
    error('no electrode positions specified!');
  end
end

% sub-set to plot if wanted
subIdx={};
if ( ~isempty(opts.subIdx) ) % compute trans on sub-set of the data
  subIdx=subsrefDimInfo(z.di,opts.subIdx{:}); % which subset
end
if ( opts.eegonly && isfield(z.di(chD).extra,'iseeg') ) % && isempty(subIdx{n2d(mu.di,'ch')}) )  
  iseeg=[z.di(chD).extra.iseeg];
  if ( any(iseeg) && any(~iseeg) ) 
    if( ~isempty(subIdx) && isempty(subIdx{chD}) ) 
      subIdx{chD}=iseeg; else subIdx{chD}=intersect(subIdx{chD},find(iseeg)); 
    end;
  end;
end
if ( isempty(subIdx) ) 
  sX=z.X;
else
  sX = z.X(subIdx{:}); 
  coords=coords(:,subIdx{chD}); % sub-set electrodes too
end;

% permute to make plot work
if ( chD~=1 ) sX=permute(sX,[chD 1:chD-1 chD+1:ndims(sX)]); end;

% compute the plot labels
labels=mkLabels(z.di([1:chD-1 chD+1:end]));

% call the function to make the actual plot
jplot(coords,sX,'labels',labels,jplotopts,varargin);
return;


