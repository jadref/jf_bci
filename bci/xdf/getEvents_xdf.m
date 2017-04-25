function [status,hdr]=getEvents(filename,varargin)
opts = struct('statusZero_samp',2,'markerCh','Status','isbdf',[]);
[opts,varargin]=parseOpts(opts,varargin);

if ( ischar(filename) ) 
   hdr=openxdf(filename); % get the header info
else
   hdr=filename;
end

nchannels = hdr.NS;
nRec      = hdr.NRec;
reclen    = hdr.SPR;
nsamp     = reclen * nRec; 

if ( isempty(hdr.EVENT) && strmatch('EDF Annotations',hdr.Label) ) % use the event stream
  annotch = strmatch('EDF Annotations',hdr.Label,'exact');
  annot=readxdf(hdr,annotch,[],[]);
  annot=[char(bitand(annot(2:2:end),2^8-1));char(bitand(bitshift(annot(1:2:end),-8),2^8-1))];
  annot=annot(:)';
  [onset_s,dur_s,Desc,type,typeDesc,typeIdx]=parseAnnotations(annot);
  hdr.EVENT.pos = round(onset_s*hdr.SampleRate);
  hdr.EVENT.DUR = dur_s * HDR.SampleRate;
  hdr.EVENT.TYP = type;
  hdr.EVENT.Desc= Desc;
  hdr.EVENT.TYP_Desc = typeDesc;
end  

% Extract the status info so we know how to slice the data.  
if ( ~isempty(hdr.EVENT) && (isfield(hdr.EVENT,'POS') && ~isempty(hdr.EVENT.POS)) )
  status = hdr.EVENT;
else % extract status info from the status channel
  statusch=opts.markerCh;
  if ( isstr(statusch) )
     statusch = strmatch(opts.markerCh,hdr.Label,'exact');
     if ( isempty(statusch) ) 
        statusch = strmatch(upper(opts.markerCh),hdr.Label,'exact');
        if( isempty(statusch) )
           error('Couldnt find a status channel');
        end
     end
  end

  status=readxdf(hdr,statusch,[],[]);
  
  % BODGE! - reprocess biosemi status channel to extract the marker info!
  if ( (~isempty(opts.isbdf) && opts.isbdf) || (isempty(opts.isbdf) && strcmpi(opts.markerCh,'Status')) ) 
     if ( isempty(opts.isbdf) )
        warning('Assuming this is a biosemi gdf-file - so modifying the marker channel info, use isbdf,0 to override.');
     end
     statusinf=int32(status);
     % find indices of negative numbers
     bit24i = find(statusinf < 0 | statusinf>0);
     % make number positive and preserve bits 0-22
     statusinf(bit24i) = bitcmp(abs(statusinf(bit24i))-1);
     % apparently 24 bits reside in 3 higher bytes, shift right 1 byte
     statusinf(bit24i) = bitshift(statusinf(bit24i),-8);
     % re-insert the sign bit on its original location, i.e. bit24
     statusinf(bit24i) = statusinf(bit24i)+(2^(24-1));

     % typecast the data to ensure that the status channel is represented in 32 bits
     statusinf = uint32(statusinf);

     if( isempty(statusinf) ) warning('No data in file: %s',filename); return; end
     % convert to unsigned 24-bit number by: adding 2^23 and masking out bits 25-32
     % N.B. this method loses the value of the 24th bit!
     statusinf=bitand(uint32(statusinf+2^(24-1)),2^24-1); % +24 bit number
     status   =bitand(statusinf,2^16-1);% actual status info in low-order 16 bits
     epoch    =int8(bitget(statusinf,16+1));
     cmrange  =int8(bitget(statusinf,20+1));
     battery  =int8(bitget(statusinf,22+1));
  end
end
