function [events,bgns,ends,offset_samp,opts,bgnsi,endsi]=blkEventEpochFn(events,varargin)
% search through set of events, to first identify the block, then the events within it
%
% [events,bgns,ends,offset_samp,opts,bgnsi,endsi]=eventEpochFn(events,varargin)
% Inputs:
%  events - [struct] vector of events to search.  Events should have the 
%           fields: 
%               'type','value','sample','offset','duration'
%            OR
%               'TYP',  'VAL', 'POS',  'DUR'
%
% Options:
%  blkStart -- {2x1} cell array to match the block start type, and block value type
%  blkEndVal-- {} cell array to match the **value** for end of block
%  blkEnd   -- {2x1} cell array of match the block end type+value
%  matchBlks-- [int] when find multiple blocks, only take this one in file order
%  startSet -- {2x1} cell array of cell arrays to match
%              startSet{1} -- type match possibilities
%              startSet{2} -- value match possibilities
%              e.g. to match type='stim' and value=10 or 11 use: 'startSet',{{'stim'} [10 11]}
%        OR
%              [Nx1 float] set of marker values to match.  Any type with this value matches.
%  endSet   -- {2x1} cell array of cell arrays to match as for startSet
%                N.B. If not given then then is the same as start set
%  offset_ms/samp -- [2x1] start/end of window offsets to impose such that
%               win_start = win_start+offset(1), win_end=win_end+offset(2)
%  trlen_ms/samp  -- [1x1] override endSet and just set the epoch endpoint to this much after the start
%  fs         -- [int] sample rate of the data
% Outputs:
%  type -- [M x 1] id of the epoch (start marker ID)
%  bgns -- [M x 1] indices of the start of the recording phases
%  ends -- [M x 1] indicies of the ends of the recording phases
%  bgnsi-- [M x 1] indicies in to the events structure of the start of the epochs
opts=struct('blkStart',[],'blkEndVal',[],'blkEnd',[],'matchBlks',[]);
[opts,varargin]=parseOpts(opts,varargin);

% first sub-set to the bits between the block start and end
mi=true(size(events));
if ( ~isempty(opts.blkStart) )
  if ( ~isempty(opts.blkEnd) ) % separate match for start/end events
	 blkStrtMi= matchEvents(events,opts.blkStart{:}); % start of all blocks	 
	 blkStrtMi= find(blkStrtMi);
	 blkEndMi = matchEvents(events,opts.blkEnd{:}); % end of all blocks
	 blkEndMi = find(blkEndMi);

  else % start/end share type
	allblkMi = matchEvents(events,opts.blkStart{1}); % start of all blocks
	allblkMi = find(allblkMi);
	if ( isempty(allblkMi) ) warning('didnt match any block starts'); end;
	bsmi  = matchEvents(events(allblkMi),opts.blkStart{:}); % start tgt blocks
	blkStrtMi=allblkMi(bsmi);
	if ( ~isempty(opts.blkEndVal) ) % end have same type as start
      bemi = matchEvents(events(allblkMi),opts.blkStart{1},opts.blkEndVal);
      if ( isempty(bemi) ) warning('didnt match target end value'); end;
      blkEndMi=allblkMi(bemi);
	else
      bemi=find(bsmi)+1;
      blkEndMi=allblkMi(min(end,bemi));
      if( bemi(end)>numel(allblkMi) ) blkEndMi=numel(events); end; % fix last one to go to end of file
	end
  end
  mi(:)=false;
  if ( isempty(blkStrtMi) ) warning('didnt match target blocks');
  else
	 blkI=0;
	 for bi=1:numel(blkStrtMi);
		if ( isempty(blkEndMi) )
		  if ( ~isempty(opts.matchBlks)) % check if only subset blks to be processed
			 blkI=blkI+1; if ( ~any(blkI==opts.matchBlks) ) continue; end;
		  end;
		  mi(blkStrtMi(bi):blkStrtMi(min(end,bi+1)))=true;
		else										  
			bei = find(blkEndMi>blkStrtMi(bi),1); % find the next end
			if ( ~isempty(bei) )
			  bsi = find(blkStrtMi<blkEndMi(bei),1,'last'); % find closest start to this end
			  if ( bsi==bi ) % closest start/end pair
				 if ( ~isempty(opts.matchBlks)) % check if only subset blks to be processed
					blkI=blkI+1; if ( ~any(blkI==opts.matchBlks) ) continue; end;
				 end;
				 mi(blkStrtMi(bi):blkEndMi(bei))=true;
			  else
				 warning('multiple blk-starts in a row without end....');
			  end
			else
			  warning('blk-start without end...');
			end
		 end
	  end
	  if ( bsmi(end)==numel(blkStrtMi) ) mi(blkStrtMi(end):end)=true; end;
	end
end
mi=find(mi);

[tmp,bgns,ends,offset_samp,opts,bgnsi,endsi]=eventEpochFn(events(mi),varargin{:});	
% re-number back to orginal events
bgnsi=mi(bgnsi);
endsi=mi(endsi);
events=events(bgnsi);
return;
