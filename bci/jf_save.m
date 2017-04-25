function [z]=jf_save(z,label,overwrite,rootdir,mkdirp,errorp)
% save a jf object in the std location
%  jf_save(z,label,overwrite,rootdir,mkdirp,errorp)
global bciroot; if ( isempty(bciroot) ) bciroot = {'~/data/bci/'}; end;
if ( nargin < 2 || isempty(label) ) label=z.label; end;
if ( nargin < 3 ) overwrite=[]; end;
if ( isnumeric(label) && isempty(overwrite) ) overwrite=label; label=z.label; end;
if ( nargin < 4 ) rootdir=[]; end;
if ( nargin < 5 || isempty(mkdirp) ) if ( overwrite ) mkdirp=1; else mkdirp=0; end; end;
if ( nargin < 6 || isempty(errorp) ) errorp=1; end;
if( isfield(z,'session') ) session=z.session; else session=''; end;
if( isfield(z,'subj') ) subj=z.subj; else subj=''; end;
if( isfield(z,'expt') ) expt=z.expt; else expt=''; end;

if ( isempty(rootdir) ) 
   if ( isfield(z,'rootdir') && exist(z.rootdir,'dir') ) % use load dir by preference
      rootdir=z.rootdir;
   elseif ( isequal(expt(1),'~') || isequal(expt(1),filesep()) )
     rootdir='';
   else
      for ri=1:numel(bciroot); 
         if ( exist(bciroot{ri},'dir') && ...
              ( exist(fullfile(bciroot{ri},expt),'dir') || ...
                exist(fullfile(bciroot{ri},expt,subj),'dir') || ...
                exist(fullfile(bciroot{ri},expt,'subjects',subj),'dir') ) )
            rootdir=bciroot{ri}; break;          
         end; 
       end;
       if ( isempty(rootdir) ) rootdir=bciroot{1}; end;
   end
end

% try and find-or-make the most specific directory for this data
sdir=fullfile(expt,subj);
if ( ~exist(fullfile(rootdir,sdir),'dir') )
   sdir=fullfile(expt,'subjects',subj);
   if ( ~exist(fullfile(rootdir,sdir),'dir') )
      if ( ~mkdirp ) 
         error('Couldnt find a subject directory under: %s',fullfile(rootdir,expt,subj)); 
      else
         sdir=fullfile(expt,subj);
         diri=find(sdir==filesep | sdir=='/');
         if(diri(1)==1);diri=diri(2:end);end;
         if(diri(end)~=numel(sdir))diri=[diri numel(sdir)+1];end;
         for di=1:numel(diri);
            if(~exist(fullfile(rootdir,sdir(1:diri(di)-1)),'dir')) mkdir(fullfile(rootdir,sdir(1:diri(di)-1)));end
         end
         mkdir(fullfile(rootdir,sdir));
      end
   end
end
% now try to add the session info
fdir=fullfile(rootdir,sdir);
if ( ~isempty(session) && exist(fullfile(fdir,session),'dir') )
   fdir=fullfile(fdir,session);
elseif ( ~isempty(session) && mkdirp ) % make session dir
  mkdir(fullfile(fdir,session));
  fdir=fullfile(fdir,session);
end
% now add the jf_prep info
fdir=fullfile(fdir,'jf_prep');
% finally make the directory if its not already there
if ( ~exist(fdir,'dir') )  mkdir(fdir); end

% check for existing files we might overwrite
subjfn=subj;
if ( any(subjfn==filesep) ) subjfn(subjfn==filesep)='_'; end;
fn=fullfile(fdir,sprintf('%s_%s',subjfn,label));
if ( exist([fn '.mat'],'file')~=0 )
  if( iscell(fn) ) fn=cat(2,fn{:}); end;
   if ( isequal(overwrite,1) )
      warning('Existing file: %s *WILL BE OVERWRITTEN*',fn);% automatically OK
      pause(1);
   elseif ( isequal(overwrite,0) ) % automatically not-OK
      error('File: %s already exists\n',fn);
   else
      fprintf('File: %s already exists.\n',fn);
      fprintf('Overwrite file With summary info:\n');
		s=load([fn '.mat'],'summary');fprintf('%s\n',s.summary);
      fprintf('With:\n%s\n',jf_disp(z));
      r=input('Overwrite (Y/N)?','s'); 
      if ( ~isequal(lower(r),'y') ) fprintf('Save aborted\n'); return; end;
   end
end

z.label=label;
z.rootdir=rootdir;
z.subjdir=sdir ; % where we came from & should save to
% build a summary string
z.summary = jf_disp(z);

fprintf('Saving: %s\n',fn);
try;
   if ( ~exist('OCTAVE_VERSION','builtin') ) 
      save([fn '.mat'],'-V7.3','-struct','z'); % don't use compression
   else
      save([fn '.mat'],'-V6','-struct','z'); % don't use compression
   end
  fid=fopen([fn '.txt'],'w');
  fprintf(fid,'%s\n',z.summary);
  fclose(fid);
catch
   %save([fn '.mat'],'-struct','z'); % fall back on full save type
  le=lasterror;fprintf('ERROR Caught:\n %s\n%s\n',le.identifier,le.message);
  if ( errorp )     
    error('save failed!');    
  else
    warning('save failed!');
  end
end
return;
%-------------------------------------------------------------------------------
function testCase()
z=jf_import('eeg/test','test','test',randn(100,100,10),{'ch','time','epoch'},'fs',100)

jf_save(z)

z.session='1';


