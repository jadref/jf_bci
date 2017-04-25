function edit(varargin)
%   Hacked: JF-3/4/07 -- uses EDITOR env to find editor if jvm not running!
%     N.B. rename/disable matlab/toolbox/matlab/codetools/edit.m to use
%
%EDIT Edit M-file.
%   EDIT FUN opens the file FUN.M in a text editor.  FUN must be the
%   name of an M-file or a MATLABPATH relative partial pathname (see
%   PARTIALPATH).
%
%   EDIT FILE.EXT opens the specified file.  MAT and MDL files will
%   only be opened if the extension is specified.  P and MEX files
%   are binary and cannot be directly edited.
%
%   EDIT X Y Z ... will attempt to open all specified files in an
%   editor.  Each argument is treated independently.
%
%   EDIT, by itself, opens up a new editor window.
%
%   By default, the MATLAB built-in editor is used.  The user may
%   specify a different editor by modifying the Editor/Debugger
%   Preferences.
%
%   If the specified file does not exist and the user is using the
%   MATLAB built-in editor, an empty file may be opened depending on
%   the Editor/Debugger Preferences.  If the user has specified a
%   different editor, the name of the non-existent file will always
%   be passed to the other editor.
%   


%   Copyright 1984-2005 The MathWorks, Inc.
%   $Revision: 1.1 $  $Date: 2007-04-03 11:58:28 $

if ~iscellstr(varargin)
    error(makeErrID('NotString'), 'The input must be a string.');
end

try
    if (nargin == 0)
        openEditor;
    else
        for i = 1:nargin
            argName = translateUserHomeDirectory(strtrim(varargin{i}));

            checkIsDirectory(argName);
            checkEndsWithBadExtension(argName);

            if (~openIfRunning(argName) && ~openFileIfExists(argName))
                openUsingWhich(argName);
            end
        end 
    end 
catch
    rethrow(lasterror); % rethrow so that we don't display stack trace
end

%--------------------------------------------------------------------------
% Special case for opening invoking 'edit' from inside of a function:
%   function foo
%   edit bar
% In the case above, we should be able to pick up private/bar.m from 
% inside foo.
function opened = openIfRunning(argName)
opened = false;
st = dbstack('-completenames');
% if there are more than two frames on the stack, then edit was called from
% a function
if length(st) > 2 
    dirName = fileparts(st(3).file);
    file = fullfile(dirName, 'private', argName);
    opened = openFileIfExists(file);
    if (~opened && ~hasExtension(file))
        file = [file '.m'];
        opened = openFileIfExists(file);
    end
end

%--------------------------------------------------------------------------
% Helper function that displays an empty file -- taken from the previous edit.m
% Now passes error message to main function for display through error.
function showEmptyFile(file, origArg)
errMessage = '';
errID = '';

% If nothing is found in the MATLAB workspace or directories,
% open a blank buffer only if a simple filename is specified.
% We do this because the directories specified may not exist, and
% it would be too difficult to deal with all the cases.
if isSimpleFile(file)
    checkValidName(file);

    err = javachk('mwt', 'The MATLAB Editor');
    if ~isempty(err)
        % You must have mwt to run the editor on the PC
        if ~isunix
            errMessage = err.message; % cannot cannot call miedit on Windows
            errID = err.identifier;
            % If there is no mwt on Unix, try to run user's default editor
        else
           editor = getenv('EDITOR');
           if ( isempty(editor) ) editor='/usr/bin/emacsclient'; end; % default
       
           % Special case for vi
           if strcmp(editor,'vi') == 1
              editor = 'xterm -e vi';
           end
       
           % On UNIX, we don't want to use quotes in case the user's editor
           % command contains arguments (like "xterm -e vi")
           if nargin == 0
              eval(['!' editor ' &'])
           else
              eval(['!' editor ' "' file '" &'])
           end
           
           % system_dependent('miedit', file);
        end;
    else
        % if we are using the built-in editor and don't show empty buffers
        % then display error message
        if com.mathworks.mde.editor.EditorOptions.getShowNewFilePrompt == false ...
                && com.mathworks.mde.editor.EditorOptions.getNamedBufferOption == com.mathworks.mde.editor.EditorOptions.NAMEDBUFFER_DONTCREATE ...
                && com.mathworks.mde.editor.EditorOptions.getBuiltinEditor ~= 0

            [errMessage, errID] = showFileNotFound(file, origArg, false);
        else
            openEditor(fullfile(pwd,file), true);
        end
    end
else
    [errMessage, errID] = showFileNotFound(file, origArg, false);
end
handleError(errMessage, errID);


%--------------------------------------------------------------------------
% Helper function that calls the java editor.  Taken from the original edit.m.
% Did modify to pass non-existent files to outside editors if
% user has chosen not to use the built-in editor.
% Also now passing out all error messages for proper display through error.
% Note that "emptyBuffer" indicates whether or not MATLAB thinks
% (via information that may possibly be cached) that the file exists.
% It is possible that this is incorrect (for example, if the toolbox
% cache is out-of-date and the file actually no longer is on disc).
function openEditor(file, emptyBuffer)
% OPENEDITOR  Open file in user specified editor

errMessage = '';
errID = '';

% Make sure our environment supports the editor
% Need mwt to get com.mathworks classes (they may depend on mwt).
err = javachk('swing', 'The MATLAB Editor');
if ~isempty(err)
    if isunix
       editor = getenv('EDITOR');
       if ( isempty(editor) ) editor='/usr/bin/emacsclient'; end; % default
       
       % Special case for vi
       if strcmp(editor,'vi') == 1
          editor = 'xterm -e vi';
       end
       
       % On UNIX, we don't want to use quotes in case the user's editor
       % command contains arguments (like "xterm -e vi")
       if nargin == 0
          eval(['!' editor ' &'])
       else
          eval(['!' editor ' "' file '" &'])
       end
    end
else

    % Determine which editor to run.  Assume builtin editor to begin with.
    builtinEd = 1;

    % Get the MATLAB editor preference.
    biEdFlag = system_dependent('getpref', 'EditorBuiltinEditor');
    biEdChoice = system_dependent('getpref', 'EditorOtherEditor');
    if ~isempty(strfind(biEdFlag, 'false')) && length(biEdFlag) > 1
        editor = biEdChoice(2:end); % Trim off the leading 'S'
        if length(deblank(editor)) > 0
            % This flag should be followed by a java class that implements
            % the External Editor Interface
            if ~strncmp(editor, '-eei', 4)
                builtinEd = 0;
            end
        end
    end

    if builtinEd == 1
        % Swing isn't available, so return with error
        if ~isempty(err)
            errMessage = err.message;
            errID = err.identifier;
        else
            % Try to open the Editor
            try
                if nargin==0
                    com.mathworks.mlservices.MLEditorServices.newDocument;
                else
                    if emptyBuffer || fileExists(file)
                        com.mathworks.mlservices.MLEditorServices.openDocument(file);
                    else
                        [errMessage, errID] = showFileNotFound(file, file, true);
                    end
                end % if nargin
            catch
                % Failed. Bail
                errMessage = 'Failed to open editor. Load of Java classes failed.';
                errID = 'JavaErr';
            end
        end
    else
        % User-specified editor
        if ispc
            % On Windows, we need to wrap the editor command in double quotes
            % in case it contains spaces
            if nargin == 0
                eval(['!"' editor '" &'])
            else
                eval(['!"' editor '" "' file '" &'])
            end
        elseif isunix && ~strncmp(computer,'MAC',3)
            % Special case for vi
            if strcmp(editor,'vi') == 1
                editor = 'xterm -e vi';
            end

            % On UNIX, we don't want to use quotes in case the user's editor
            % command contains arguments (like "xterm -e vi")
            if nargin == 0
                eval(['!' editor ' &'])
            else
                eval(['!' editor ' "' file '" &'])
            end
        else
            % Run on Macintosh
            if nargin == 0
                openFileOnMac(editor)
            else
                openFileOnMac(editor, file);
            end
        end
    end
end
handleError(errMessage, errID);

%--------------------------------------------------------------------------
% Helper method to run an external editor from the Mac
function openFileOnMac(applicationName, absPath)

% Put app name in quotes
appInQuotes = ['"' applicationName '"'];

% Is this a .app -style application, or a BSD exectuable?
% If the former, use it to open the file (if any) via the
% BSD OPEN command.
if length(applicationName) > 4 && strcmp(applicationName(end-3:end), ...
        '.app')
    % Make sure that the .app actually exists.
    if exist(applicationName) ~= 7
        error(makeErrID('ExternalEditorNotFound'), ...
            ['Could not find external editor ' applicationName]);
    end
    if nargin == 1 || isempty(absPath)
        unix(['open -a ' appInQuotes]);
    else
        unix(['open -a ' appInQuotes ' "' absPath '"']);
    end
    return;
end

% At this point, it must be BSD a executable (or possibly nonexistent)
% Can we find it?
[status, result] = unix(['which ' appInQuotes ]);

% UNIX found the application
if status == 0
    % Special case for vi and emacs since they need a shell
    if ~isempty(strfind(applicationName,'/vi')) || ...
            strcmp(applicationName, 'vi') == 1
        appInQuotes = ['xterm -e ' appInQuotes];
    elseif ~isempty(strfind(applicationName, '/emacs')) || ...
            strcmp(applicationName, 'emacs') == 1
        appInQuotes = ['xterm -e ' appInQuotes];
    end

    if nargin == 1 || isempty(absPath)
        command = [appInQuotes ' &'];
    else
        command = [appInQuotes ' "' absPath '" &'];
    end

    % We think that we have constructed a viable command.  Execute it,
    % and error if it fails.
    [status, result] = unix(command);
    if status ~= 0
        error(makeErrID('ExternalEditorFailure'), ...
            ['Could not open external editor ' result]);
    end
    return;
else
    % We could not find a BSD executable.  Error.
    error(makeErrID('ExternalEditorNotFound'), ...
        ['Could not find external editor ' result]);
end

%--------------------------------------------------------------------------
% Helper function that trims spaces from a string.  Taken from the original
% edit.m
function s1 = strtrim(s)
%STRTRIM Trim spaces from string.

if isempty(s)
    s1 = s;
else
    % remove leading and trailing blanks (including nulls)
    c = find(s ~= ' ' & s ~= 0);
    s1 = s(min(c):max(c));
end

%----------------------------------------------------------------------------
% Checks if filename is valid by platform.
function checkValidName(file)
% Is this a valid filename?
if ~isunix
    invalid = '/\:*"?<>|';
    a = strtok(file,invalid);

    if ~strcmp(a, file)
        errMessage = sprintf('File ''%s'' contains invalid characters.', file);
        errID = 'BadChars';
        handleError(errMessage, errID);
    end
end

%--------------------------------------------------------------------------
% Helper method that checks if a string specified is a directory.
% If it is a directory, an error message is thrown.
function checkIsDirectory(s)

errMessage = '';
errID = '';

% If argument specified is a simple filename, don't check to
% see if it is a directory (will treat as a filename only).
if isSimpleFile(s)
    return;
end

dir_result = dir(s);

if ~isempty(dir_result)
    dims = size(dir_result);
    if (dims(1) > 1)
        errMessage = sprintf('Can''t edit the directory ''%s.''', s);
        errID = 'BadDir';
    else
        if (dir_result.isdir == 1)
            errMessage = sprintf('Can''t edit the directory ''%s.''', s);
            errID = 'BadDir';
        end
    end
end
handleError(errMessage, errID);

%--------------------------------------------------------------------------
% Helper method that checks if a file exists (exactly as typed).
% Returns true if exists, false otherwise.
function [result, absPathname] = fileExists(argName)

dir_result = dir(argName);

% Default return arguments
result = 0;
absPathname = argName;

if ~isempty(dir_result)
    dims = size(dir_result);
    if (dims(1) == 1)
        if dir_result.isdir == 0
            result = 1;  % File exists
            % If file exists in the current directory, return absolute path
            if (isSimpleFile(argName))
                absPathname = [pwd filesep dir_result.name];
            end
        end
    end
end

%--------------------------------------------------------------------------
% Translates a path like '~/myfile.m' into '/home/username/myfile.m'.
% Will only translate on Unix.
function pathname = translateUserHomeDirectory(pathname)
if isunix && strncmp(pathname, '~/', 2)
    pathname = [deblank(evalc('!echo $HOME')) pathname(2:end)];
end

%--------------------------------------------------------------------------
% Helper method that determines if filename specified has an extension.
% Returns true if filename does have an extension, false otherwise
function result = hasExtension(s)

[pathname,name,ext] = fileparts(s);
if (isempty(ext))
    result = false;
    return;
end
result = true;


%----------------------------------------------------------------------------
% Helper method that returns error message for file not found
%
function [errMessage, errID] = showFileNotFound(file, origArg, rehashToolbox)

if (strcmp(file, origArg))                  % we did not change the original argument
    errMessage = sprintf('File ''%s'' not found.', file);
    errID = 'FileNotFound';
else        % we couldn't find original argument, so we also tried modifying the name
    errMessage = sprintf('Neither ''%s'' nor ''%s'' could be found.', origArg, file);
    errID = 'FilesNotFound';
end

if (rehashToolbox) % reset errMessage to rehash message
    errMessage = sprintf('File ''%s''\nis on your MATLAB path but cannot be found.\nVerify that your toolbox cache is up-to-date.', file);
end

%--------------------------------------------------------------------------
% Helper method that checks if filename specified ends in .mex or .p.
% For mex, actually checks if extension BEGINS with .mex to cover different forms.
% If any of those bad cases are true, throws an error message.
function checkEndsWithBadExtension(s)

errMessage = '';
errID = '';

[pathname,name,ext] = fileparts(s);
ext = lower(ext);
if (strcmp(ext, '.p'))
    errMessage = sprintf('Can''t edit the P-file ''%s''.', s);
    errID = 'PFile';
elseif (strcmp(ext, ['.' mexext]))
    errMessage = sprintf('Can''t edit the MEX-file ''%s''.', s);
    errID = 'MexFile';
end
handleError(errMessage, errID);

%--------------------------------------------------------------------------
function handleError(errMessage, errID)
if (~isempty(errMessage))
    error(makeErrID(errID), '%s', errMessage);
end

%--------------------------------------------------------------------------
% Helper method that checks to see if a file exists
% exactly.  If it does, tries to open file.
function fExists = openFileIfExists(argName)

[fExists, pathName] = fileExists(argName);

if (fExists)
    openEditor(pathName, false);
end

%--------------------------------------------------------------------------
% Helper method that checks for directory seps.
function result = isSimpleFile(file)

result = false;
if isunix
    if isempty(findstr(file, '/'))
        result = true;
    end
else % on windows be more restrictive
    if isempty(findstr(file, '\')) && isempty(findstr(file, '/'))...
            && isempty(findstr(file, ':')) % need to keep : for c: case
        result = true;
    end
end

%--------------------------------------------------------------------------
% Helper method for error messageID display
function realErrID = makeErrID(errIDin)
realErrID = ['MATLABeditor:'  errIDin];

%--------------------------------------------------------------------------
function openUsingWhich(argName)

origName = argName;
opened = false;

if (~hasExtension(argName))
    argName = [argName '.m'];
    opened = openFileIfExists(argName);
end

if (~opened)
    % Now do a which of the new filename (with the extension)
    fName = avoidCodetoolsPrivateDir(which(argName), argName);
   
    if (~isempty(fName))     
        % Since we have a file extension and got a result from which, 
        % we know that the file that exists.
        openEditor(fName, false);
    else
        % Determine if which returns an MDL file
        MdlExistResult = 4;
        if (~strcmp(argName, origName) && exist(which(origName)) == MdlExistResult)
            error(makeErrID('MdlErr'), 'Can''t edit the MDL-file ''%s'' unless you include the ''.mdl'' file extension.', argName);
        end
        % Displaying a non-existent file.
        showEmptyFile(argName, origName);
    end
end

%--------------------------------------------------------------------------
% Avoid picking up files in codetools/private if the user has a file with
% the same name (e.g., takepicture.m).
function result = avoidCodetoolsPrivateDir(whichResult, argName)
result = '';
if (~isempty(whichResult))
    codetoolsPrivateDir = fullfile(fileparts(which('edit.m')), 'private');
    if (~isempty(strmatch(codetoolsPrivateDir, whichResult)) ...
            && isempty(strmatch('private', argName)))
        whichAll = which('-all', argName);
        for i=1:length(whichAll)
            whichAllInd = whichAll{i};
            if ~isempty(strfind(whichAllInd, [filesep 'private' filesep])) && ...
                    ~isempty(strfind(whichAllInd, [filesep '@']))
                result = whichAllInd;
                break;
            end
        end
    else
        result = whichResult;
    end
end
