function [dat,hdr]=readxdf(hdr,chanindx,bgnsamp,endsamp)
% [dat,hdr]=readxdf(hdr,chanindx,bgnsamp,endsamp)
if ( nargin < 2 || isempty(chanindx) ) chanindx=1:hdr.NS; end;
if ( nargin < 3 || isempty(bgnsamp) ) 
   if( isfield(hdr,'FILE') && isfield(hdr.FILE,'POS') ) 
      bgnsamp=hdr.FILE.POS;
   else
      bgnsamp=1; 
   end
end;
if ( nargin < 4 || isempty(endsamp) ) 
   endsamp=hdr.Dur * hdr.SampleRate(1)*hdr.NRec; 
end

filename    = hdr.FileName;
try fseek(hdr.FILE.FID,0,0);
   % OK, file already open.
   fid = hdr.FILE.FID;
catch
   % open the file ourselves
   fid = fopen(hdr.FileName,'r','ieee-le');
   hdr.FILE.FID = fid;
   if ( fid < 0 ) 
      error('Could not open %s',filename);
   end
end

% call a sub-function to do the actual reading
hdr.InChanSelect = chanindx;  % restrict to the wanted sub-set of channels
[dat,hdr]=readxdf_sread(hdr,bgnsamp,endsamp);
dat=dat';  % convert to chan x samp format

% Calibrate the data
if length(chanindx)>1
   % using a sparse matrix speeds up the multiplication
   if( numel(hdr.Cal)==hdr.NS ) hdr.Cal=hdr.Cal(:); end;
   calib = hdr.Cal(chanindx);
   for ch=1:size(dat,1); dat(ch,:) = calib(ch) * dat(ch,:); end;
else
   % in case of one channel the calibration would result in a sparse array
   calib = hdr.Cal(chanindx);
   dat   = calib * dat;
end

if( nargout < 2 ) fclose(fid); end;

%%%%%%
%% the rest extracted from biosig toolbox, file: t200/sread.m
function [S,HDR]=readxdf_sread(HDR,bgnsamp,endsamp)
% Outputs:
%  S -- the data
%  hdr -- the updated header info
HDR.FILE.POS = bgnsamp;

nr     = double(min(HDR.NRec*HDR.SPR-bgnsamp, endsamp-bgnsamp));
S      = repmat(NaN,nr,length(HDR.InChanSelect)); 

block1 = floor(bgnsamp/HDR.SPR); % number of the record which contains bgn sample
ix1    = bgnsamp- block1*HDR.SPR;	% starting sample (minus one) within 1st block 
nb     = double(ceil((bgnsamp+nr)/HDR.SPR)-block1); % total number blocks to read
fp     = double(HDR.HeadLen + block1*HDR.AS.bpb); % starting file pointer position
STATUS = fseek(HDR.FILE.FID, fp, 'bof');
count  = 0;
if HDR.NS==0,
elseif all(HDR.GDFTYP==HDR.GDFTYP(1)),
  if (HDR.AS.spb*nb<=2^24), % faster access
    S = [];
    %[HDR.AS.spb, nb,block1,ix1,bgnsamp],
    [s,c] = fread(HDR.FILE.FID,[HDR.AS.spb, nb],gdfdatatype(HDR.GDFTYP(1)));
    for k = 1:length(HDR.InChanSelect),
      K = HDR.InChanSelect(k);
      if (HDR.AS.SPR(K)>0)
        S(:,k) = rs(reshape(s(HDR.AS.bi(K)+1:HDR.AS.bi(K+1),:),HDR.AS.SPR(K)*nb,1),HDR.AS.SPR(K),HDR.SPR);
      else
        S(:,k) = NaN;
      end;
    end;
    S = S(ix1+1:ix1+nr,:); % Huh? why is this +1?
    count = nr;
  else
    S = repmat(NaN,[nr,length(HDR.InChanSelect)]);
    while (count<nr);
      len   = ceil(min([(nr-count)/HDR.SPR,2^22/HDR.AS.spb]));
      [s,c] = fread(HDR.FILE.FID,[HDR.AS.spb, len],gdfdatatype(HDR.GDFTYP(1)));
      s1    = zeros(HDR.SPR*c/HDR.AS.spb,length(HDR.InChanSelect));
      for k = 1:length(HDR.InChanSelect), 
        K = HDR.InChanSelect(k);
        if (HDR.AS.SPR(K)>0)
          tmp = reshape(s(HDR.AS.bi(K)+1:HDR.AS.bi(K+1),:),HDR.AS.SPR(K)*c/HDR.AS.spb,1);
          s1(:,k) = rs(tmp,HDR.AS.SPR(K),HDR.SPR);
        end;
      end;
      ix2   = min(nr-count, size(s1,1)-ix1);
      S(count+1:count+ix2,:) = s1(ix1+1:ix1+ix2,:);
      count = count+ix2;
      ix1   = 0; 
    end;	
  end;
else
  fprintf(2,'SREAD (GDF): different datatypes - this might take some time.\n');
  
  S = repmat(NaN,[nr,length(HDR.InChanSelect)]);
  while (count<nr);
    s = [];
    for k=1:length(HDR.AS.TYP),
      [s0,tmp] = fread(HDR.FILE.FID,[HDR.AS.c(k), 1],gdfdatatype(HDR.AS.TYP(k)));
      s = [s;s0];
    end;
    
    s1    = repmat(NaN,[HDR.SPR,length(HDR.InChanSelect)]);
    for k = 1:length(HDR.InChanSelect), 
      K = HDR.InChanSelect(k);
      if (HDR.AS.SPR(K)>0)
        s1(:,k) = rs(s(HDR.AS.bi(K)+1:HDR.AS.bi(K+1),:),HDR.AS.SPR(K),HDR.SPR);
      end;
    end;
    ix2   = min(nr-count, size(s1,1)-ix1);
    S(count+1:count+ix2,:) = s1(ix1+1:ix1+ix2,:);
    count = count+HDR.SPR;
    ix1   = 0; 
  end;	
end;
if strcmp(HDR.TYPE,'GDF')       % read non-equidistant sampling channels of GDF2.0 format
  if (HDR.VERSION>1.94) %& isfield(HDR.EVENT,'VAL'),
    for k = 1:length(HDR.InChanSelect), 
      ch = HDR.InChanSelect(k);
      if (HDR.AS.SPR(ch)==0),
        ix = find((HDR.EVENT.TYP==hex2dec('7fff')) & (HDR.EVENT.CHN==ch));
        pix= HDR.EVENT.POS(ix)-bgnsamp;
        ix1= find((pix > 0) & (pix <= count));
        S(pix(ix1),k)=HDR.EVENT.DUR(ix(ix1));
      end;
    end;
  end;
end
HDR.FILE.POS = bgnsamp + count;
return
