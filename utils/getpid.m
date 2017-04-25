function [varargout]=getpid(varargin)
% get the current process ID

% The rest of this code is a mex-hiding mechanism which compilies the mex if
% this runs and recursivly calls itself.  
% Based upon code from: http://theoval.sys.uea.ac.uk/matlab
cwd  = pwd; % store the current working directory
name = mfilename('fullpath'); % get the directory where we are
% find out what directory it is defined in
name(name=='\')='/'; % deal with dos'isms
dir=name(1:max(find(name == '/')-1)); % dir is everything before final '/'
try % try changing to that directory
   cd(dir);
catch   % this should never happen, but just in case!
   cd(cwd);
   error(['unable to locate directory containing ''' name '.m''']);
end

try % try recompiling the MEX file
   fprintf(['Compiling ' mfilename ' for first use\n']);
   mex('getpid.c','-output',mfilename);
   fprintf('done\n');
catch
   % this may well happen happen, get back to current working directory!
   cd(cwd);
   error('unable to compile MEX version of ''%s''%s\n%s%s', name, ...
         ', please make sure your', 'MEX compiler is set up correctly', ...
         ' (try ''mex -setup'').');
end

cd(cwd); % change back to the current working directory
rehash;  % refresh the function and file system caches

% recursively invoke MEX version using the same input and output arguments
[varargout{1:nargout}] = feval(mfilename, varargin{:});

% bye bye...

return;
