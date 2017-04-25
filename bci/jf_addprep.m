function [z]=jf_addprep(z,method,summary,opts,info)
% add prep record to the input jf-data-struct
%
% [z]=jf_addprep(z,method,summary,opts,info)
% Inputs:
%  d       -- jf data structure, (N.B. use empty d to just get prep struct)
%  method  -- string containing the method name
%  summary -- summary string describing what was done
%  opts    -- options passed to the method
%  info    -- other useful info produced by the method
% Outputs:
%  d.prep    -- structure array containing information about the
%       |       procesing that's been applied to get X.  This contains:
%       |.method -- the mfile used to process X
%       |.opts   -- the options passed to the method
%       |.info   -- other useful information produced by method
%       |.summary-- short textual description of what method did to X
if ( nargin < 3 ) error('Insufficient arguments'); end;
if ( nargin < 4 ) opts=[]; end; if ( iscell(opts) ) opts={opts}; end;
if ( nargin < 5 ) info=[]; end; if ( iscell(info) ) info={info}; end;
if ( ~ischar(summary) ) summary=char(summary); end;
prep = struct('method',method,'opts',opts,'info',info,'summary',summary,...
              'timestamp',datestr(now,'yyyymmddHHMM'));
if ( ~isfield(z,'prep') || isempty(z.prep) ) z.prep=prep; else z.prep(end+1)=prep; end;
sumstr=sprintf('%2d) %s %14s - %s',numel(z.prep),prep.timestamp,prep.method,prep.summary);
if ( isfield(z,'summary') ) z.summary = sprintf('%s\n%s',z.summary,sumstr); 
else z.summary = sumstr; 
end;
return;
%--------------------------------------------------
function testCase()
tt=jf_addprep([],'hello','summary',struct(),[]);
tt=jf_addprep(tt,'hello','summary',struct(),[]);
