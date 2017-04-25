function [X,di,fs,summary,opts,info] = readraw_comp3ii(filename, varargin)
opts=struct('label','','fs',[],'single',0,'set','all');
opts=parseOpts(opts,varargin);

info=struct('filename',filename);
v=load(filename);% load data-files

if ( ~isempty(strfind(filename,'Test')) ) % read the true-labels info for test-sets
   [fnpath]=fileparts(filename);
   fns=dir(fnpath); fns=fns(~[fns.isdir]); fns={fns.name};
   ti=strmatch('true_labels',fns);
   if ( ~isempty(ti) ) % if true labels file
      fid=fopen(fullfile(fnpath,fns{ti(1)}));c=fscanf(fid,'%c');fclose(fid); % load the labels
      v.TargetChar=c; % record the target characters
   end
end

fs= 240; % sampling rate for this data
X = permute(v.Signal, [3 2 1]); % [ nCh x nSamp x nLetter ]
[nCh nSamp nLet]=size(X);
samp2ms = 1000/fs;

lettergrid={'A' 'B' 'C' 'D' 'E' 'F';
            'G' 'H' 'I' 'J' 'K' 'L';
            'M' 'N' 'O' 'P' 'Q' 'R';
            'S' 'T' 'U' 'V' 'W' 'X';
            'Y' 'Z' '1' '2' '3' '4';
            '5' '6' '7' '8' '9' '_'};
if ~isfield(v, 'TargetChar'), v.TargetChar = repmat(' ', 1, nLet); end
for li = 1:nLet
	target      = v.TargetChar(li);
   marker      = strmatch(v.TargetChar(li),lettergrid,'exact'); % integer rep of the target
	flashi_samp = find(diff([0 v.Flashing(li, :)], 1, 2)>0); % find the flash locations
   flashi_ms   = flashi_samp * samp2ms;
	stimCode    = v.StimulusCode(li, flashi_samp);           % find the r/c type
   % construct the flip grid
   flipgrid = false([size(lettergrid),numel(flashi_samp)]);
   for fi=1:numel(flashi_samp);
      if ( stimCode(fi)<7 ) flipgrid(:,stimCode(fi),fi) = true; 
      else                  flipgrid(stimCode(fi)-6,:,fi)=true;
      end
   end
   [r,c]       = ind2sub(size(lettergrid),marker);
   flash       = shiftdim(flipgrid(r,c,:),1); % extract from flipgrid
   if ( isfield(v,'StimulusType') ) % compare with from file
      tt=v.StimulusType(li, flashi_samp); 
      if ( ~all(flash(:)==tt(:)) ) error('different answers!'); end;
   end
   extra(li)=struct('marker',marker,'target',target,'flashi_ms',flashi_ms,'stimCode',stimCode,...
                    'flash',flash,'flipgrid',reshape(flipgrid,[],size(flipgrid,ndims(flipgrid))));
end
         
% pick the sub-set of data wanted
cut = [];
switch opts.set
	case 'train', cut = any(isnan(y), 2);
	case 'test', cut = ~any(isnan(y), 2);
	case 'all', cut = [];
	otherwise, error('haeh?')
end
X(cut, :, :) = [];
extra(cut)=[];

% make the dim-info
di = mkDimInfo(size(X),...
               'ch',[],[],...
               'time','ms',(0:size(X,2)-1)*1000./fs,...
               'letter',[],[],[],'uV',[]);
di(3).info.lettergrid   =lettergrid; % letter layout
[di(3).extra]=num2csl(extra,2); % store extra letter info

[fpath,fname,fext]=fileparts(filename);
summary=sprintf('from: %s%s',fname,fext);
summary = sprintf('(%s)', opts.set);
return;