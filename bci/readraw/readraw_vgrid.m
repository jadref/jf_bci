function [X,di,fs,summary,opts,info]=readraw_vgrid(filename,varargin)
opts=struct('label','','fs',[],'single',0);
opts=parseOpts(opts,varargin);

fprintf('reading %s\n', filename);
s=load(filename);% load data-files
info=struct('filename',filename);

if isfield(s, 'period_msec') & ~isfield(s, 'globalprm')
	s.globalprm = prm('$vgrid$event$duration_msec',s.period_msec);
end
if isfield(s, 'globalprm')
	info.conditions = s.globalprm;
end

fn = fieldnames(s);
lfn = lower(char(fn));

ix = find(lfn(:, 1) == 'x');
if isempty(ix), error(sprintf('failed to find variable name beginning ''x'' in file %s', filename)), end
if length(ix) > 1, error(sprintf('multiple variable names beginning ''x'' in file %s', filename)), end
xname = fn{ix};
X = getfield(s,xname);
if ( opts.single && isa(X,'double') ) X=single(X); end;

iy = find(lfn(:, 1) == 'y');
if isempty(iy), error(sprintf('failed to find variable name beginning ''y'' in file %s', filename)), end
if length(iy) > 1, error(sprintf('multiple variable names beginning ''y'' in file %s', filename)), end
yname = fn{iy};
y = getfield(s,yname);

if size(X, 1) ~= size(y, 1), error(sprintf('%s and %s should be the same size in dimension 1', Xname, yname)), end
y = squeeze(y);
%if any(diff([size(x, 1) prod(size(y)) length(y)])), error(sprintf('%s must be a vector with length equal to number of rows in %s', yname, xname)), end

% extract the sampleing rate
is = [];
fn = cellstr(fn);
for i = 1:length(fn)
	n = lower(fn{i});
	if strcmp(n, 'fs')
		is = i;
		break
	elseif length(n) > 7 & ~isempty(findstr(n, 'sampl')) & (~isempty(findstr(n, 'rate')) | ~isempty(findstr(n, 'freq')))
		is = i;
		break
	end
end
if isempty(is), fs = []; else fs = getfield(s,fn{is}); end

iextra = min(find(strcmpi(fn, 'extra')));
if ~isempty(iextra)
	extra = getfield(s,fn{iextra});
	if iscell(extra) & numel(extra) == size(y, 1)
		for i = 1:numel(extra), if isa(extra{i}, 'prm'), extra{i} = report(extra{i}); end, end
		info.trialinfo = struct('prm', extra);
	else
		info.blockinfo = extra;
	end
end

if ( isfield(info,'conditions') && isstr(info.conditions) ) % extract useful info from the prm
   prmFields={ 'duration_ms' '$vgrid$event$duration_msec' % things we want to extract
               'isi_ms'      '$vgrid$event£isi_msec'
               'eventType'   '$vgrid$event$type', 
               'codeTypes'   '$vgrid$event$sequence$alternate$callback'
               'codeOrder'   '$vgrid$event$sequence$alternate$order',
               'lettergrid'  '$vgrid$letters$layout'};
   for pi=1:size(prmFields,1);
      fn    = prmFields{pi,1};
      fmatch= prmFields{pi,2};
      tstrt = strfind(info.conditions,fmatch); % get line
      if ( isempty(tstrt) ) continue; end;
      tstrt = tstrt(1)+numel(fmatch);
      tend  = tstrt+strfind(info.conditions(tstrt:end),'$vgrid')-2; tend=tend(1); % get end
      fv    = info.conditions(tstrt:tend);
      fv    = fv(find(diff(isspace(fv)),1,'first')+1:find(diff(isspace(fv)),1,'last')); % strip spaces
      tmp   = str2num(fv); if ( ~isempty(tmp) ) fv=tmp; end; % parse to if poss
      info  = setfield(info,fn,fv);
   end   
end

% make the dim-info
di = mkDimInfo(size(X),...
               'letter',[],[],...
               'ch',[],[],...
               'time','ms',(1:size(X,3))*1000./fs);
di(end).units='uV';
di(1).info.globalprm=s.globalprm;
di(1).info.lettergrid=info.lettergrid; % letter layout
[di(1).extra.trialinfo]=num2csl(info.trialinfo); % store trial info
[di(1).extra.flash]=num2csl(y,2);
if( isfield(info,'codeOrder') ) [di(1).extra.code] = num2csl(info.codeOrder); end
% extract useful info from the trialinfo
if ( isfield(info.trialinfo,'prm') )
   for i=1:numel(info.trialinfo);
      tinfo=info.trialinfo(i).prm.tree.root;
      [di(1).extra(i).flipgrid]=tinfo.flipgrid.SELF;
      [di(1).extra(i).target]=tinfo.label.letter.SELF;
      if ( isnumeric(di(1).extra(i).target) )
         [di(1).extra(i).target]=sprintf('%d',di(1).extra(i).target);
      end
   end
end
info=rmfield(info,'trialinfo'); % remove what we've stored in di
%for i=1:numel(di(1).extra); di(1).extra(i).prm=s.extra{i}; end;

% ensure correct dimension order
X = permute(X,[2:ndims(X), 1]);
di= di([2:end-1 1 end]);
[fpath,fname,fext]=fileparts(filename);
summary=sprintf('from: %s%s',fname,fext);
return;
%---------------------------------------------------------------------------
function testCase()
