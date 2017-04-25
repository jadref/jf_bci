%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  freename.m      2002-12-07  %           FIND FREE FILE NAME
%%      (c)         M. Balda    %          updated on 2005-07-15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The function finds first unused name of a file in a required directory.
% The final name of the file consists of a name base appended by n digits
% (possibly with leading zeros) expressing an order of the new file
% possessing the same name base. The function is good for generating names 
% of files containing results of an alternative processing of the same data.
%
% Forms of calls:
% ~~~~~~~~~~~~~~
%   filename = freename(dirname,base);
%       dirname  = name of the directory, where to find a new name
%       base     = base of the new name (without extension)
%   filename = freename(dirname,base,n);
%       n        = number of figures to be appended (3 is default)
%       filename = free name equals a base appended by 'n' figure characters
%
% Examples:
% ~~~~~~~~
%   filename =[freename('Results','meas',2) '.txt'];
%       The name 'meas01.txt' will be sent to filename if there is no file
%       of the same base name 'meas' present in the subdirectory 'Results'.
%       The name 'meas04.txt' is returned provided 'meas03' exists in the
%       subdirectory 'Results'.
%   file = [freename('./', 'data') '.dat'];
%       returns 'data001.dat' if no file file with the base name 'data'
%       exists in the current directory.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function filename = freename(dirname,base,n)
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if exist(dirname, 'dir')                %   Test directory existence
    if nargin<3, n=3; end               %   n = 3 is default
    base(base==' ') = '';               %   delete all spaces in name base
    Dir = dir(dirname);
    D   = strvcat(Dir(3:end).name);
    lb  = length(base);
    m   = size(D,1);
    w   = 0;

    for k = 1:m                         %   Cycle directory items
        j = strfind(D(k,:),base);
        if ~isempty(j)
            j = j+lb;
            w = max(str2double(D(k,j:j+n-1)),w);
        end
    end

    w = w+1;
    ord = sprintf(['%0' num2str(n) 'd'], w);
    filename = [base ord];
else
    warning(['Directory ' dirname ' does not exist'])
    filename = '';
end
