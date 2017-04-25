srcdir=fileparts(mfilename('fullpath')); % parent directory, i.e. matfiles directory
% general functions -- if needed
if ( isempty(which('parseOpts')) )
  addpath(fullfile(srcdir,'utils'));
  excludeDirs={'CVS','.svn','.git','__MACOSX','MacOS','private','temp','.bak' 'fieldtrip' 'ft' 'spm' 'biosig' 'Psychtoolbox'};
  addpath(exGenPath(srcdir,excludeDirs));
end
