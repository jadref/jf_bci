function [X,di,fs,summary,opts,info]=readraw_fieldtrip(filename,varargin);
% convert fieldtrip struct to a jf_bciobj one
%
% [X,di,fs,summary,opts,info]=readraw_fieldtrip(filename[,options])
%
% Inputs:
%  filename -- the fieldtrip file to read
%              OR
%              [struct] a fieldtrip data structure
% Options:
%  Y              -- [Nx1] set of labels for the trials
%  single         -- [bool] flag store data in single format          (1)
% Outputs
%  X              -- [numel(chanIdx)+1 x nSamp x nEpoch] data matrix
%  di             -- [ndims(X)+1 x 1] dimInfo struct describing the layout of X
%  fs             -- [float] sampling rate of this data
%  summary        -- [str] string describing what we loaded
%  opts           -- the parsed options structure
%  info           -- [struct] containing additional useful info about the loaded data
opts=struct('Y',[],'single',1,'subsample',[],'permDimOrd',0);
opts=parseOpts(opts,varargin);
Y=opts.Y;
info=[];

if( ischar(filename) ) ft=load(filename); 
elseif( isstruct(filename) ) ft=filename;
end;

if ( isstruct(ft) )
   fn=fieldnames(ft);
   if ( numel(fn)==1 ) ft=getfield(ft,fn{1}); 
   else
      datafni=strmatch('data',fn,'exact');
      if ( ~isempty(datafni) ) % use the data field for the structure
         % record the config, first before we loose it
         cfgfni=[];for fi=1:numel(fn); if ( strfind(fn{fi},'cfg') ) cfgfni = fi; break; end; end; 
         if ( ~isempty(cfgfni) ) info=getfield(ft,fn{cfgfni}); end
         ft=getfield(ft,'data');
      end
   end;
end

info.filename=filename;
if ( isfield(ft,'trial') )
   if ( isstruct(ft.trial) ) 
      X=cat(3,ft.trial{:}); names = {'ch','time','epoch'};
   else 
      X=ft.trial; 
   end;
   ft.trial=[]; % free the ram
elseif ( isfield(ft,'powspctrm') )
   X  = ft.powspctrm;   
   ft.powspctrm=[]; % free the ram
end
if ( opts.single && ~isa(X,'single') ) % save some ram 
   X=single(X);
end;
if ( isfield(ft,'dimord') )  %use to re-arrange [ ch x freq x win x epoch]
   names=split('_',ft.dimord);
   perm=[];
   d=strmatch('chan',names); if( ~isempty(d) ) perm=[perm d]; names{d}='ch'; end;
   d=strmatch('freq',names); if( ~isempty(d) ) perm=[perm d]; names{d}='freq'; end;
   d=strmatch('time',names); if( ~isempty(d) ) perm=[perm d]; names{d}='time'; end;
   d=strmatch('rpt',names); if( ~isempty(d) ) perm=[perm d]; names{d}='epoch'; end;
   if ( opts.permDimOrd ) 
      perm =[perm setdiff(1:ndims(X),perm)];   
      X = permute(X,perm);  
      names=names(perm);
   end
elseif ( ndims(X)<=3 )
   names = {'ch' 'time' 'epoch'}
else
   error('Dont understand the input format');
end

% Fill in the dimInfo
di = mkDimInfo(size(X),names);

chD=n2d(di,'ch',0,0);
if ( chD )
   di(chD).name = 'ch';
   di(chD).vals = ft.label;
   if ( size(di(chD).vals,1)>1 ) di(chD).vals=di(chD).vals'; end;
   if ( isfield(ft,'grad') )
      pos3d=repop(ft.grad.tra*ft.grad.pnt,'./',sum(ft.grad.tra,2)); % mean position of each detector
      cis = matchstrs(ft.label,ft.grad.label);
      [di(chD).extra.pos3d]=num2csl(pos3d(cis,:)');
      [di(chD).extra.pos2d]=num2csl(xyz2xy(pos3d(cis,:)'));
      [di(chD).extra.ori  ]=num2csl(ft.grad.ori(cis,:)');
   elseif ( isfield(ft,'elec') && isfield(ft.elec,'pnt') )
      pos3d=ft.elec.pnt;
      cis = matchstrs(ft.label,ft.elec.label);
      [di(chD).extra.pos3d]=num2csl(pos3d(cis,:)');
      [di(chD).extra.pos2d]=num2csl(xyz2xy(pos3d(cis,:)'));      
   end
end

timeD=n2d(di,'time',0,0);
if ( timeD )
   di(timeD).name='time';
   di(timeD).units='ms';
   if ( ~isfield(ft,'time') )
      warning('Couldnt find time info');
      di(timeD).vals=1:size(X,timeDim);
   elseif ( iscell(ft.time) ) di(timeD).vals=ft.time{1}*1000;
   elseif ( isnumeric(ft.time) ) di(timeD).vals=ft.time*1000;
   else
      warning('Couldnt interpert field time');
      di(timeD).vals=1:size(X,timeDim);
   end
   if ( size(di(timeD).vals,1)>1 ) di(timeD).vals=di(timeD).vals'; end;
end

freqD=n2d(di,'freq',0,0);
if ( freqD ) 
   di(freqD).name='freq';
   di(freqD).units='Hz';
   if ( ~isfield(ft,'freq') )
      warning('Couldnt find freq info');
      di(freqD).vals=1:size(X,freqD);
   elseif ( iscell(ft.freq) ) di(freqD).vals=ft.freq{1};
   elseif ( isnumeric(ft.freq) ) di(freqD).vals=ft.freq;
   else
      warning('Couldnt interpert field freq');
      di(freqD).vals=1:size(X,freqD);
   end
   if ( size(di(freqD).vals,1)>1 ) di(freqD).vals=di(freqD).vals'; end;
end

epochD=n2d(di,'epoch',0,0);
if ( epochD ) 
   di(epochD).name='epoch';
end

if ( isfield(ft,'fsample') ) fs = ft.fsample;
elseif ( isfield(ft,'hdr') && isfield(ft.hdr,'Fs') )  fs=ft.hdr.Fs;
else fs=[];
end
if ( timeD ) di(timeD).info.fs=fs; end;

if ( isfield(ft,'cfg') && isfield(ft.cfg,'output') )
   di(end).name = ft.cfg.output; % record the value type
   switch di(end).name;
    case 'pow'; di(end).units = 'rms mV';
    otherwise; di(end).units = 'mV';
  end
end

if ( isfield(ft,'hdr') ) info.hdr = ft.hdr; end;
if ( isfield(ft,'cfg') ) info.cfg = ft.cfg; end;

if ( isempty(Y) )
   if ( isfield(ft,'trialinfo') ) trl=ft.trialinfo;
   elseif ( isfield(ft,'trl') ) trl=ft.trl;
   elseif( isfield(ft,'cfg') && isfield(ft.cfg,'trl') ) trl=ft.cfg.trl; 
   else trl=struct(); 
   end;
   if ( isfield(ft,'event') ) event=ft.event;
   elseif ( isfield(ft,'cfg') && isfield(ft.cfg,'event') ) event=ft.cfg.event;
   else event=struct(); 
   end

   if ( isfield(trl,'eventvalue') )
      Y  = agetfield(trl,'eventvalue');
   elseif ( isnumeric(trl) && isfield(event,'sample') )
      eventsamp=[event.sample];
      for ei=1:size(trl,1); 
         [tmp,trl2event(ei)]=min(abs(eventsamp-(trl(ei,1)-trl(ei,3)))); 
         if ( tmp > 2 ) warning('Hmmm, couldnt match samples info...'); end;
      end;
      di(epochD).extra = event(trl2event); % extra info about the trials
      Y = [event(trl2event).value];
   else
      warning('Couldnt find epoch info');
   end
end
if ( ~isempty(Y) ) [di(epochD).extra(:).marker]=num2csl(Y,2); end;

if ( ischar(filename) )
   summary=['fieldtrip'];
else
   summary=['fieldtrip from struct'];
end
return;