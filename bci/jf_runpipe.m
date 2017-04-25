function [z]=jf_runpipe(z,pipeline,varargin)
% apply z2's processing pipleline to z
%
% [z]=jf_runpipe(z,pipeline,varargin)
%
% Inputs:
%   z  - the data to which the processing will be applied
%   pipeline - {str} the list of processing stages to run in left->right order
% Options
%  varargin - options for each of the pipeline stages as:
%   'stagename',struct(opts), 
%   OR
%   'stagename',{'opts'}
%   OR
%   'stagename.optname',optval
%
%  verb      - [int] set the verbosity level
opts=struct('blacklist',{{'raw2jf','jf_cat'}},'steps',[],'verb',1,'pipename',[],'subIdx',[],'verb',0);
for pi=1:numel(pipeline); opts.(pipeline{pi})=[]; end;
opts=parseOpts(opts,varargin);

% run the pipeline
if ( opts.verb>0 && ~isempty(opts.pipename) ) fprintf('\n---\n%s\n---\n',opts.pipename); end;
steps=opts.steps; 
if( isempty(steps) ) steps=1:numel(pipeline); elseif ( islogical(steps) ) steps=find(steps); end;
steps(steps>numel(pipeline) | steps<1)=[]; 
for pi=steps;
   if ( opts.verb > 0 ) fprintf('%d) %s\n',pi,pipeline{pi}); end;
   if ( ~isempty(strmatch(pipeline{pi},opts.blacklist)) ) 
      if (opts.verb>1 ) fprintf('*skipped*'); end;
      continue; 
   end;
   popts=getfield(opts,pipeline{pi}); if( isempty(popts) ) popts={}; elseif (~iscell(popts) ) popts={popts}; end;      
   z = feval(pipeline{pi},z,popts{:});
   if ( opts.verb > 1 ) jf_disp(z); end;
end
if ( ~isempty(opts.pipename) ) z.alg=opts.pipename; end;
return;
%-------------------------------------------------------------
function testcase()
