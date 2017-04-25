function [z]=raw2jf(files,varargin)
% read in a (set of) rawfiles and return a jf-data-struct 
%
%  [d]=raw2jf(filenames,varargin)
% Inputs:
%  filenames -- the fully-specified file names to load, *must* be raw files
% Options:
%  readrawfn -- name of the function to call to load in the data-file
%  sessions  -- the session id for each file
%  blocks    -- the block id for each file
%  expt      -- the experiment
%  subj      -- the subject
%  label     -- label for this data set
%  rootdir   -- the root the files come from -- or where they should be saved
%  subsample -- should we sub-sample the data?  Should be a struct with fields:
%          |.fs -- desired output sampling rate
%  spatfilter-- should we spatially filter the data? (e.g. to spat downsample)
%           |   Should be a struct with fields:
%           |.sf     -- the set of spatial filters to use (in columns)
%           |.labels -- labels for the new spat filtered channels
%  capFile   -- file to find the info for the electrode locations.
%  Cnames    -- cell-array of strings which over-ride channel names
%  overridechnms-- [bool] names in capFile override ones from data file? (0)
%  dim       -- dimension to cat multi-file-inputs along ('epoch')
% Output:
%  d         -- a jf data structure containing:
%  |.X       -- the loaded data
%  |.di      -- X's dimension info.  Struct array describing each
%  |  |         dimension of X containing
%  |  |.name -- string name for this dimension
%  |  |.units-- units that dim is measured in
%  |  |.vals -- value in units of the corrospeding element of this
%  |  |         dimension of X
%  |  |.extra-- for each element along this dimension of X a struct
%  |            containing other useful information.
%  |.fs      -- X's sampling rate
%  |.prep    -- structure array containing information about the
%  |    |       procesing that's been applied to get X.  This contains:
%  |    |.method -- the mfile used to process X
%  |    |.opts   -- the options passed to the method
%  |    |.info   -- other useful information produced by method
%  |    |.summary-- short textual description of what method did to X
%  |.expt    -- the experiment + session string
%  |.subj    -- the subject
%  |.label   -- the label for this data set
%  |.summary -- short textual description of what we've done so far
global bciroot; if ( isempty(bciroot) ) bciroot = '~/data/bci/'; end;
opts = struct('expt','eeg/motor/im-tapping',...
              'subj','pd',...
              'label','act',...
              'rootdir',[],...
              'blocks',[],...
              'sessions',[],...
              'subsample',struct('fs',256),...
              'chanIdx',[],...
              'spatfilter',[],...
              'capFile',[],'Cnames',[],'overridechnms',0,...
              'readrawfn',[],...
              'dim','epoch','autoPrune',0);
[opts,varargin]=parseOpts(opts,varargin);
if ( isnumeric(opts.subsample) ) opts.subsample=struct('fs',opts.subsample); end;

% % get the channel names + positions
% if ( ~isempty(opts.capFile) )
%    [Cnames latlong xy xyz]        =readCapInf(opts.capFile);
% end

if ( isempty(files) ) 
   warning('No data files found'); return; 
end

if ( ~iscell(files) ) files={files}; end;

if ( isempty(opts.readrawfn) )
   [fpath,fname,fext]=fileparts(files{1}); fext=fext(2:end);
   switch (lower(fext));
    case 'bdf';         opts.readrawfn='readraw_bdf';
    case {'gdf','edf'}; opts.readrawfn='readraw_xdf';
    case 'dat';         opts.readrawfn='readraw_bci2000';
    case 'mat'; % use the path to identify if possible
     tmp=find(fpath==filesep); 
     if(isempty(tmp) || tmp(end)~=numel(fpath))tmp(end+1)=numel(fpath)+1;end; % get directory parts
     dtype=fpath(tmp(end-1)+1:tmp(end)-1);     
     if ( ~isempty(strfind(dtype,'ft')) || ~isempty(strfind(dtype,'fieldtrip')) )
        opts.readrawfn='readraw_fieldtrip';
     elseif( ~isempty(strfind(dtype,'bciobj')) )
        opts.readrawfn='readraw_bciobj';
     elseif( ~isempty(strfind(dtype,'bs')) || ~isempty(strfind(dtype,'brainstream')) )
        opts.readrawfn='readraw_bs';
     else
        error('dont know the readrawfn to use');
     end
    otherwise; 
     if ( (strcmp(fname,'contents') && strcmp(fext,'txt')) || ...
          (isempty(fext) && (strcmp(fname,'header'))) ) 
       opts.readrawfn=['readraw_ft_buffer_offline'];
     elseif ( exist(['readraw_' fext])==2 ) opts.readrawfn=['readraw_' fext]; 
     else error('dont know the readraw fn to use');
     end
  end
end 
% check that the selected function is available
if( ischar(opts.readrawfn) )
   if ( ~any(exist(opts.readrawfn) == [2 3]) ) % test for executable function
      if ( isempty(strmatch(opts.readrawfn,'readraw_')) ) % check if adding readraw_ helps
         opts.readrawfn=['readraw_' opts.readrawfn];
         if ( ~any(exist(opts.readrawfn) == [2 3]) )
            error('Couldnt find an executable raw reading function');
         end
      end
   end
elseif ( ~isa(opts.readrawfn,'function_handle') )
   error('Couldnt find an executable raw reading function');
end      

for fi=1:numel(files);
   filename=files{fi};
   if ( ischar(filename) )
      if ( ~exist(filename) ) error('Couldnt find file: %s',files{fi}); end;
      [fpath,fname,fext]=fileparts(files{fi});
   
      % extract the data
      fprintf('Reading : %s\n',filename);
      if(isunix && filename(1)=='~')filename=[getenv('HOME') filename(2:end)];end;%do ~ expansion
   end
   [X,di,fs,summary,rr_opts,info]=feval(opts.readrawfn,filename,varargin{:},'subsample',opts.subsample);
   epD=n2d(di,'epoch',0,0);if(epD==0)epD=ndims(X); end;
   if ( isempty(X) ) nEpoch=0; else nEpoch=size(X,epD); end;
   fprintf('\n read %d epochs\n',nEpoch);

   % if didn't find any epochs skip to the next one
   if ( isempty(X) ) continue; end;
   
   % Override channel name with given ones if wanted
   if ( ~isempty(opts.Cnames) )
      tmp=di(1).vals; di(1).vals=opts.Cnames;
      if ( numel(opts.Cnames)~=size(X,1) ) 
         warning('number of cnames (%d) not equal number of channels (%d)',...
                 numel(opts.Cnames),size(X,1)); 
         idx=numel(di(1).vals):size(X,1);
         if( (iscell(di(1).vals) && iscell(tmp)) || (isnumeric(di(1).vals) && isnumeric(tmp)) ) 
            di(1).vals(idx)=tmp(idx);
         elseif ( iscell(di(1).vals) && isnumeric(tmp) ) di(1).vals(idx)=num2csl(tmp(idx));
         else  di(1).vals(idx)=idx;
         end
      end;
   end

   % select a sub-set of channels
   if ( ~isempty(opts.chanIdx) )
      Cnames=di(1).vals;
      chanIdx=opts.chanIdx; 
      if(islogical(chanIdx)) chanIdx=find(chanIdx); 
      elseif( iscell(chanIdx) )
         if ( ischar(chanIdx{1}) && iscell(di(1).vals) && ischar(di(1).vals{1}) )
            mi     =matchstrs(chanIdx,di(1).vals);
            chanIdx=find(mi>0);
         else error('cell arrays of channels should be strings');
         end
      elseif( ~isnumeric(chanIdx) ) 
         error('Unrecognised chanIdx spec');
      end
      X=X(chanIdx,:,:,:,:); 
      di(1).vals=di(1).vals(chanIdx); di(1).extra=di(1).extra(chanIdx);
   end
   
   if ( ~isempty(opts.subsample) && opts.subsample.fs < fs ) % sub-sample
      ofs=fs;
      fprintf('Subsampling: from %gHz -> %gHz ...',fs,opts.subsample.fs);
      [X,idx] = subsample(X,size(X,2)*opts.subsample.fs/fs,2);
      fs= opts.subsample.fs;
      di(2).vals  = di(2).vals(round(idx));
      di(2).extra = di(2).extra(round(idx));
      fprintf('done\n');
   end
   di(2).info.fs = fs;

   if ( ~isempty(opts.spatfilter) ) % spat-filter, i.e. spat-downsample
      dim=1; % channel dimension in X
      X=tprod(X,[1:dim-1 -dim dim+1:ndims(X)],opts.spatfilter.sf,[-dim dim],'n');
      if( ~isfield(opts.spatfilter,'labels')|| isempty(opts.spatfilter.labels))
         for i=1:size(X,dim); opts.spatfilter.labels{i}=sprintf('sf%2d',i);end;
      end;
      di(1).vals = opts.spatfilter.labels;
   end   
         
   if ( ~isempty(opts.blocks) )
      di(3).info.block=opts.blocks(fi);
   end
   if ( ~isempty(opts.sessions) ) 
      di(3).info.session=opts.sessions(fi);
   end
   di(3).info.filename=filename; % record source file name
   
   % setup the prep structure and build the final result
   info.testFn='';
   zprep = jf_addprep([],mfilename,summary,rr_opts,info);
   zz(fi)=struct('X',X,'di',di,'prep',zprep.prep);
   clear X zprep info;
end

if( ~exist('zz','var') ) 
   warning('No data loaded'); z=[]; return;
end
   
% remove any empty entries
rmIdx=[];for i=1:numel(zz); if(isempty(zz(i).X)) rmIdx=[rmIdx;i];end;end;
zz(rmIdx)=[]; % remove empty entries

% cat the entries together
if ( numel(zz)==1 ) 
  z=zz;
else
  z=jf_cat(zz,'dim',opts.dim,'autoPrune',opts.autoPrune);
end
clear zz; % free some ram
if ( ~isempty(opts.capFile) && ischar(opts.capFile) ) % add pos info
   z.di(n2d(z,'ch'))=addPosInfo(z.di(n2d(z,'ch')),opts.capFile,opts.overridechnms);
end


allOpts = mergeStruct(opts,z(1).prep(1).opts);
z.prep(1) = struct('method',mfilename(),'opts',allOpts,...
                   'info',z(1).prep(1).info,'summary',z(1).prep(1).summary,...
                   'timestamp',datestr(now,'yyyymmddHHMM'));
if ( ~isempty(opts.subsample) && exist('ofs','var') ) 
   if ( ~exist('ofs') )
      if ( isfield(z(1).prep(1).info,'ofs') ) ofs=z(1).prep(1).info.ofs; 
      else ofs=1;
      end
   end;
   info=struct('ofs',ofs);
   sssummary=sprintf('%gHz -> %gHz',ofs,opts.subsample.fs);
   z=jf_addprep(z,'subsample',sssummary,opts.subsample,info);
end
z.expt = opts.expt; if(~isempty(opts.expt)&&z.expt(end)==filesep)z.expt(end)=[]; end; % remove last dir part
z.subj = opts.subj;
z.label= opts.label;
session= opts.sessions;
if ( ~isempty(session) && iscell(session) )
   % if they're all in 1 session then use it alone, otherwise cat session names together
   onesess=true;
   for i=2:numel(session);if(~isequal(session{i-1},session{i})) onesess=false; break; end; end;
   if ( onesess ) % all from 1 session
      if ( isempty(session{1}) )       session=[];      
      elseif( ischar(session{1}) )      session=session{1}; 
      elseif ( isnumeric(session{1}) ) session=sprintf('%d',session{1}); 
      end
   else  % from different sessions
      if ( isempty(session{1}) )       session=[];
      elseif ( ischar(session{1}) )     session=[session{1} sprintf('_%s',session{2:end})];
      elseif( isnumeric(session{1}) )  session=[sprintf('%d',session{1}) sprintf('_%d',session{2:end})]
      end
   end
elseif( ~isempty(session) && isnumeric(session) )
   session=[sprintf('%d',session(1)) sprintf('_%d',session(2:end))]
end
z.session=session;
if ( ~isempty(opts.rootdir) ) z.rootdir=opts.rootdir; 
end;
if ( numel(files)==1 ) z.srcfile=files{1}; else z.srcfile=files; end;

% build a summary string
z.summary=sprintf('%s \t %s \t (%s)\n\n',z.expt,z.subj,z.label);
tmp={z.prep.timestamp;z.prep.method;z.prep.summary};
z.summary=[z.summary sprintf('%s\t%s,\t %s\n',tmp{:})];

return;

%----------------------------------------------------------------------------
function []=testCase;
global bciroot; bciroot ={'~/data/bci' '/Volumes/BCI_Data' '/media/JASON_BACKU/data/bci'};
expt       = 'own_experiments/test_experiments/comparative_test';
subjects   = {'rs'};
sessions   = {{'20081111' '20081112'} '' '' '' '' ''};
sessdirregexp='(.*)';
dtype      = 'raw_bci2000';
markerdict = {'LH' 'RH' 'BH' 'FT'};

% these are all per subject/session/condition
blocks = {{{[4 5 7 8 10 11] [6 9 12]} {[6 7 9 10 12 13] [8 11 14]}}};
blkfileregexp='.*R([0-9]*).dat';
markers= {1:4};

subj=subjects{1}; session=sessions{1}{1}; block=blocks{1}{1}{1}; marker=markers{1};
filelst = findFiles(expt,subj,session,block,...
                    'dtype',dtype,'sessdirregexp',sessdirregexp,'blkfileregexp',blkfileregexp);

z=raw2jf({filelst.fname},'expt',expt,'subj',subj,'label','raw2jf',...
         'blocks',[filelst.block],'sessions',{filelst.session},...
         'RecPhaseLimits',[769:770],'trlen_ms',3000);
