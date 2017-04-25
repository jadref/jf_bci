function [z]=jf_follow(z,z2,varargin)
% apply z2's processing pipleline to z
%
% [z]=jf_follow(z,z2,varargin)
%
% Inputs:
%   z  - the data to which the processing will be applied
%   z2 - the template data, who's processing is to be followed
% Options
%  blacklist - {str} list of methods to ignore/not follow ({'raw2jf','jf_cat'})
%  steps     - [int] list of z2's prep steps to apply to z, do all steps if empty ([])
%  verb      - [int] set the verbosity level
opts=struct('blacklist',{{'raw2jf','jf_cat'}},'steps',[],'verb',1);
opts=parseOpts(opts,varargin);

% run the steps
steps=opts.steps; 
if( isempty(steps) ) steps=1:numel(z2.prep); elseif ( islogical(steps) ) steps=find(steps); end;
steps(steps>numel(z2.prep) | steps<1)=[]; 
for pi=steps;
   prep=z2.prep(pi);
   if ( opts.verb > 0 ) fprintf('%d) %s\n',pi,prep.method); end;
   if ( isfield(prep.info,'testFn') ) % if we've stored an apply method use it
      if( ~isempty(prep.info.testFn) )
         z = feval(prep.info.testFn{1},z,prep.info.testFn{2:end},'verb',opts.verb);
      end
   else
      if( isempty(strmatch(prep.method,opts.blacklist)) )
         z = feval(prep.method,z,prep.opts,'verb',opts.verb);
      end
   end
end
return;
%-------------------------------------------------------------
function testcase()
