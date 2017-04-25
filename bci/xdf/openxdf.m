function [HDR,H1,h2] = sopen(arg1,PERMISSION,CHAN,MODE,arg5,arg6)
% SOPEN opens signal files for reading and writing and returns 
%       the header information. Many different data formats are supported.
%
% N.B. RIPPED FROM THE BIOSIG TOOLBOX!!! and renamed 
%  and removed all but the gdf/bdf/edf reading functions
%
%
% Reading of data: 
% 	HDR = sopen(Filename, 'r', [, CHAN [, MODE]]);
% 	[S,HDR] = sread(HDR, NoR, StartPos);
% 	HDR = sclose(HDR);
%
% Writing of data: 
%	HDR = sopen(HDR, 'w');
%   	writing requires a predefined HDR struct. see demo3.m 
%
% 2nd argument (PERMISSION) is one of the following strings 
%	'r'	read header
%	'w'	write header
%       'rz'    on-the-fly decompression of gzipped files (only supported with Octave 2.9.3 or higher). 
%       'wz'    on-the-fly compression to gzipped files (only supported with Octave 2.9.3 or higher). 
%
% CHAN defines a list of selected Channels
%   	Alternative CHAN can be also a Re-Referencing Matrix ReRefMx, 
%       	(i.e. a spatial filter) in form of a matrix or a 
%               filename of a MarketMatrix format  
%   	E.g. the following command returns the difference and 
%   	    the mean of the first two channels. 
%   	HDR = sopen(Filename, 'r', [[1;-1],[.5,5]]);
%   	[S,HDR] = sread(HDR, Duration, Start);
%   	HDR = sclose(HDR);
%
% MODE  'UCAL'  uncalibrated data
%       'OVERFLOWDETECTION:OFF' turns off automated overflow detection
%       'OUTPUT:SINGLE' returned data is of class 'single' [default: 'double']
%       '32bit' for NeuroScan CNT files reading 4-byte integer data
%       Several opteions can be concatenated within MODE. 
%
% HDR contains the Headerinformation and internal data
% S 	returns the signal data 
%
% Several files can be loaded at once with SLOAD
%
% see also: SLOAD, SREAD, SSEEK, STELL, SCLOSE, SWRITE, SEOF


%	$Id: sopen.m 2650 2011-03-09 16:07:08Z schloegl $
%	(C) 1997-2006,2007,2008,2009.2011 by Alois Schloegl <a.schloegl@ieee.org>	
%    	This is part of the BIOSIG-toolbox http://biosig.sf.net/
%
%    BioSig is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BioSig is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BioSig.  If not, see <http://www.gnu.org/licenses/>.


if isnan(str2double('1, 3'));
        fprintf(2,'Warning BIOSIG: incorrect version of STR2DOUBLE.\n');
        fprintf(2,'- Make sure the path to this directory comes before the path to ... /matlab/toolbox/matlab/strfun/\n');
        fprintf(2,'Running the script below should fix the problem. \n\n');
        fprintf(2,'   x = fileparts( which(''sopen'') );\n');
        fprintf(2,'   rmpath(x);\n   addpath(x,''-begin'');\n\n');
end;

global FLAG_NUMBER_OF_OPEN_FIF_FILES;

if ischar(arg1),
        HDR.FileName = arg1;
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
%elseif length(arg1)~=1,
%	HDR = [];
elseif isfield(arg1,'name')
        HDR.FileName = arg1.name;
	HDR.FILE = arg1; 
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
else %if isfield(arg1,'FileName')
        HDR = arg1;
%else
%	HDR = [];
end;

if ~isfield(HDR,'FILE'),
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
end;	
if ~isfield(HDR.FILE,'stdout'),
        HDR.FILE.stdout = 1;
end;	
if ~isfield(HDR.FILE,'stderr'),
        HDR.FILE.stderr = 2;
end;

if nargin<3, CHAN = 0; end; 
if nargin<4, MODE = ''; end;
if nargin<2, 
        HDR.FILE.PERMISSION='r'; 
elseif isempty(PERMISSION),
        HDR.FILE.PERMISSION='r'; 
elseif isnumeric(PERMISSION),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: second argument should be PERMISSION, assume its the channel selection\n');
        CHAN = PERMISSION; 
        HDR.FILE.PERMISSION = 'r'; 
elseif ~any(PERMISSION(1)=='RWrw'),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: PERMISSION must be ''r'' or ''w''. Assume PERMISSION is ''r''\n');
        HDR.FILE.PERMISSION = 'r'; 
else
        HDR.FILE.PERMISSION = PERMISSION; 
end;

LABELS = {}; 
if iscell(CHAN),
	LABELS = CHAN; 
	CHAN = 0; 
	ReRefMx = []; 
elseif ischar(CHAN)
        H2 = sopen(CHAN,'r'); H2=sclose(H2); 
        ReRefMx = H2.Calib;
        CHAN = find(any(CHAN,2));
elseif all(size(CHAN)>1) || any(floor(CHAN)~=CHAN) || any(CHAN<0) || (any(CHAN==0) && (numel(CHAN)>1));
        ReRefMx = CHAN; 
        CHAN = find(any(CHAN,2));
elseif all(CHAN>0) && all(floor(CHAN)==CHAN), 
	if any(diff(CHAN)<=0),
	%	fprintf(HDR.FILE.FID,'Warning SOPEN: CHAN-argument not sorted - header information like Labels might not correspond to data.\n');
	end;	
        ReRefMx = sparse(CHAN,1:length(CHAN),1);
else    
        ReRefMx = [];
end
if isempty(MODE), MODE=' '; end;	% Make sure MODE is not empty -> FINDSTR

% test for type of file 
if any(HDR.FILE.PERMISSION=='r'),
        HDR = getfiletype(HDR);
	if HDR.ErrNum, 
		fprintf(HDR.FILE.stderr,'%s\n',HDR.ErrMsg);
		return;
	end;
elseif any(HDR.FILE.PERMISSION=='w'),
	[pfad,file,FileExt] = fileparts(HDR.FileName);
	HDR.FILE.Name = file;
	HDR.FILE.Path = pfad;
	HDR.FILE.Ext  = FileExt(2:length(FileExt));
        if any(HDR.FILE.PERMISSION=='z')
                HDR.FILE.Ext = [HDR.FILE.Ext,'.gz'];
                HDR.FileName = [HDR.FileName,'.gz'];
                HDR.FILE.PERMISSION = 'wz'; 
        else
                HDR.FILE.PERMISSION = 'w'; 
        end;
	HDR.FILE.OPEN = 0;
        HDR.FILE.FID  = -1;
	HDR.ErrNum  = 0; 
	HDR.ErrMsg = '';
	
	if isfield(HDR,'NS') && (HDR.NS>0), 
	        HDR = physicalunits(HDR); 
	end;         
end;

%% Initialization
if ~isfield(HDR,'NS');
        HDR.NS = NaN; 
end;
if ~isfield(HDR,'SampleRate');
        HDR.SampleRate = NaN; 
end;
if ~isfield(HDR,'PhysDim');
%        HDR.PhysDim = ''; 
end;
if ~isfield(HDR,'T0');
        HDR.T0 = repmat(nan,1,6);
end;
if ~isfield(HDR,'Filter');
        HDR.Filter.Notch    = NaN; 
        HDR.Filter.LowPass  = NaN; 
        HDR.Filter.HighPass = NaN; 
end;
if ~isfield(HDR,'FLAG');
        HDR.FLAG = [];
end;
if ~isfield(HDR.FLAG,'FILT')
        HDR.FLAG.FILT = 0; 	% FLAG if any filter is applied; 
end;
if ~isfield(HDR.FLAG,'TRIGGERED')
        HDR.FLAG.TRIGGERED = 0; % the data is untriggered by default
end;
if ~isfield(HDR.FLAG,'UCAL')
        HDR.FLAG.UCAL = ~isempty(strfind(MODE,'UCAL'));   % FLAG for UN-CALIBRATING
end;
if ~isfield(HDR.FLAG,'OVERFLOWDETECTION')
        HDR.FLAG.OVERFLOWDETECTION = isempty(strfind(upper(MODE),'OVERFLOWDETECTION:OFF'));
end; 
if ~isfield(HDR.FLAG,'FORCEALLCHANNEL')
        HDR.FLAG.FORCEALLCHANNEL = ~isempty(strfind(upper(MODE),'FORCEALLCHANNEL'));
end; 
if ~isfield(HDR.FLAG,'OUTPUT')
	if ~isempty(strfind(upper(MODE),'OUTPUT:SINGLE'));
		HDR.FLAG.OUTPUT = 'single'; 
	else
		HDR.FLAG.OUTPUT = 'double'; 
	end; 
end; 

if ~isfield(HDR,'EVENT');
        HDR.EVENT.TYP = []; 
        HDR.EVENT.POS = []; 
end;
%%%%% Define Valid Data types %%%%%%
%GDFTYPES=[0 1 2 3 4 5 6 7 16 17 255+(1:64) 511+(1:64)];
GDFTYPES=[0 1 2 3 4 5 6 7 16 17 18 255+[1 12 22 24] 511+[1 12 22 24]];

%%%%% Define Size for each data type %%%%%
GDFTYP_BYTE=zeros(1,512+64);
GDFTYP_BYTE(256+(1:64))=(1:64)/8;
GDFTYP_BYTE(512+(1:64))=(1:64)/8;
GDFTYP_BYTE(1:19)=[1 1 1 2 2 4 4 8 8 4 8 0 0 0 0 0 4 8 16]';

if strcmp(HDR.TYPE,'EDF') || strcmp(HDR.TYPE,'GDF') || strcmp(HDR.TYPE,'BDF'),
  H2idx = [16 80 8 8 8 8 8 80 8 32];
  
  HDR.ErrNum = 0; 
  HANDEDNESS = {'unknown','right','left','equal'}; 
  GENDER  = {'X','Male','Female'};
  SCALE13 = {'unknown','no','yes'};
  SCALE14 = {'unknown','no','yes','corrected'};
  
  if any(HDR.FILE.PERMISSION=='r');
    [HDR.FILE.FID]=fopen(HDR.FileName,[HDR.FILE.PERMISSION,'b'],'ieee-le');          
    
    if HDR.FILE.FID<0 
      HDR.ErrNum = [32,HDR.ErrNum];
      return;
    end;

    %%% Read Fixed Header %%%
    [H1,count]=fread(HDR.FILE.FID,[1,192],'uint8');     %
    if count<192,
      HDR.ErrNum = [64,HDR.ErrNum];
      return;
    end;
    
    HDR.VERSION=char(H1(1:8));                     % 8 Byte  Versionsnummer 
    if ~(strcmp(HDR.VERSION,'0       ') || all(abs(HDR.VERSION)==[255,abs('BIOSEMI')]) || strcmp(HDR.VERSION(1:3),'GDF'))
      HDR.ErrNum = [1,HDR.ErrNum];
      if ~strcmp(HDR.VERSION(1:3),'   '); % if not a scoring file, 
                                          %	    return; 
      end;
    end;
    if strcmp(char(H1(1:8)),'0       ') 
      HDR.VERSION = 0; 
    elseif all(abs(H1(1:8))==[255,abs('BIOSEMI')]), 
      HDR.VERSION = -1; 
    elseif all(H1(1:3)==abs('GDF'))
      HDR.VERSION = str2double(char(H1(4:8))); 
    else
      HDR.ErrNum = [1,HDR.ErrNum];
      if ~strcmp(HDR.VERSION(1:3),'   '); % if not a scoring file, 
                                          %	    return; 
      end;
    end;
    
    HDR.Patient.Sex = 0;
    HDR.Patient.Handedness = 0;
    if 0,
		HDR.Patient.Weight = NaN;
		HDR.Patient.Height = NaN;
		HDR.Patient.Impairment.Visual = NaN;
		HDR.Patient.Smoking = NaN;
		HDR.Patient.AlcoholAbuse = NaN;
		HDR.Patient.DrugAbuse = NaN;
		HDR.Patient.Medication = NaN;
    end;
    %if strcmp(HDR.VERSION(1:3),'GDF'),
    if strcmp(HDR.TYPE,'GDF'),
      if (HDR.VERSION >= 1.90)
        HDR.PID = deblank(char(H1(9:84)));                  % 80 Byte local patient identification
        HDR.RID = deblank(char(H1(89:156)));                % 80 Byte local recording identification
        [HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
        HDR.Patient.Name = tmp(2:end); 
        
        HDR.Patient.Medication   = SCALE13{bitand(floor(H1(85)/64),3)+1};
        HDR.Patient.DrugAbuse    = SCALE13{bitand(floor(H1(85)/16),3)+1};
        HDR.Patient.AlcoholAbuse = SCALE13{bitand(floor(H1(85)/4),3)+1};
        HDR.Patient.Smoking      = SCALE13{bitand(H1(85),3)+1};
        tmp = abs(H1(86:87)); tmp(tmp==0) = NaN; tmp(tmp==255) = inf;
        HDR.Patient.Weight = tmp(1);
        HDR.Patient.Height = tmp(2);
        HDR.Patient.Sex = bitand(H1(88),3); %GENDER{bitand(H1(88),3)+1};
        HDR.Patient.Handedness = HANDEDNESS{bitand(floor(H1(88)/4),3)+1};
        HDR.Patient.Impairment.Visual = SCALE14{bitand(floor(H1(88)/16),3)+1};
        if H1(156)>0, 
          HDR.RID = deblank(char(H1(89:156)));
        else
          HDR.RID = deblank(char(H1(89:152)));
          %HDR.REC.LOC.RFC1876  = 256.^[0:3]*reshape(H1(153:168),4,4);
          HDR.REC.LOC.Version   = abs(H1(156));
          HDR.REC.LOC.Size      = dec2hex(H1(155));
          HDR.REC.LOC.HorizPre  = dec2hex(H1(154));
          HDR.REC.LOC.VertPre   = dec2hex(H1(153));
        end;
        HDR.REC.LOC.Latitude  = H1(157:160)*256.^[0:3]'/3600000;
        HDR.REC.LOC.Longitude = H1(161:164)*256.^[0:3]'/3600000;
        HDR.REC.LOC.Altitude  = H1(165:168)*256.^[0:3]'/100;

        tmp = H1(168+[1:16]);
        % little endian fixed point number with 32 bits pre and post comma 
        t1 = tmp(1:8 )*256.^[-4:3]';
        HDR.T0 = datevec(t1);
        t2 = tmp(9:16)*256.^[-4:3]';
        HDR.Patient.Birthday = datevec(t2);
        if (t2 > 1) && (t2 < t1),
          HDR.Patient.Age = floor((t1-t2)/365.25);
        end;
        HDR.REC.Equipment = fread(HDR.FILE.FID,[1,8],'uint8');   
        tmp = fread(HDR.FILE.FID,[1,6],'uint8');
        if (HDR.VERSION < 2.1)	
          HDR.REC.IPaddr = tmp(6:-1:1); 
        end;
        tmp = fread(HDR.FILE.FID,[1,3],'uint16'); 
        tmp(tmp==0)=NaN;
        HDR.Patient.Headsize = tmp;
        tmp = fread(HDR.FILE.FID,[3,2],'float32');
        HDR.ELEC.REF = tmp(:,1)';
        HDR.ELEC.GND = tmp(:,2)';
      else
        HDR.PID = deblank(char(H1(9:88)));                  % 80 Byte local patient identification
        HDR.RID = deblank(char(H1(89:168)));                % 80 Byte local recording identification
        [HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
        HDR.Patient.Name = tmp(2:end); 
        
        tmp = repmat(' ',1,22);
        tmp([1:4,6:7,9:10,12:13,15:16,18:21]) = char(H1(168+[1:16]));
        HDR.T0(1:6)   = str2double(tmp);
        HDR.T0(6)     = HDR.T0(6)/100;
        HDR.reserved1 = fread(HDR.FILE.FID,[1,8*3+20],'uint8');   % 44 Byte reserved
        HDR.REC.Equipment  = HDR.reserved1(1:8);
        HDR.REC.Hospital   = HDR.reserved1(9:16);
        HDR.REC.Technician = HDR.reserved1(17:24);
      end;
      
      %if str2double(HDR.VERSION(4:8))<0.12,
      if (HDR.VERSION < 0.12),
        HDR.HeadLen  = str2double(H1(185:192));    % 8 Byte  Length of Header
      elseif (HDR.VERSION < 1.92)
        HDR.HeadLen  = H1(185:188)*256.^[0:3]';    % 8 Byte  Length of Header
        HDR.reserved = H1(189:192);
      else 
        HDR.HeadLen  = H1(185:186)*256.^[1:2]';    % 8 Byte  Length of Header
      end;
      HDR.H1 = H1; 

      %HDR.NRec = fread(HDR.FILE.FID,1,'int64');     % 8 Byte # of data records
      HDR.NRec = fread(HDR.FILE.FID,1,'int32');      % 8 Byte # of data records
      fread(HDR.FILE.FID,1,'int32');      % 8 Byte # of data records
                                          %if strcmp(HDR.VERSION(4:8),' 0.10')
      if ((abs(HDR.VERSION - 0.10) < 2*eps) || (HDR.VERSION > 2.20)),
        HDR.Dur = fread(HDR.FILE.FID,1,'float64');	% 8 Byte # duration of data record in sec
      else
        tmp  = fread(HDR.FILE.FID,2,'uint32');  % 8 Byte # duration of data record in sec
                                                %tmp1 = warning('off');
        HDR.Dur = tmp(1)./tmp(2);
        %warning(tmp1);
      end;
      tmp = fread(HDR.FILE.FID,2,'uint16');     % 4 Byte # of signals
      HDR.NS = tmp(1);
    else 
      H1(193:256)= fread(HDR.FILE.FID,[1,256-192],'uint8');     %
      H1 = char(H1);
      HDR.PID = deblank(char(H1(9:88)));                  % 80 Byte local patient identification
      HDR.RID = deblank(char(H1(89:168)));                % 80 Byte local recording identification
      [HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
      [tmp1,tmp] = strtok(tmp,' ');
      [tmp1,tmp] = strtok(tmp,' ');
      HDR.Patient.Name = tmp(2:end); 
      
      tmp = find((H1<32) | (H1>126)); 		%%% syntax for Matlab
      if ~isempty(tmp) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                       %H1(tmp)=32; 
        HDR.ErrNum=[1025,HDR.ErrNum];
      end;
      
      tmp = repmat(' ',1,22);
      tmp([3:4,6:7,9:10,12:13,15:16,18:19]) = H1(168+[7:8,4:5,1:2,9:10,12:13,15:16]);
      tmp1 = str2double(tmp);
      if length(tmp1)==6,
        HDR.T0(1:6) = tmp1;
      end;
      
      if any(isnan(HDR.T0)),
        HDR.ErrNum = [1032,HDR.ErrNum];
        
        tmp = H1(168 + [1:16]);
        tmp(tmp=='.' | tmp==':' | tmp=='/' | tmp=='-') = ' ';
        tmp1 = str2double(tmp(1:8));
        if length(tmp1)==3,
          HDR.T0 = tmp1([3,2,1]);
        end;	
        tmp1 = str2double(tmp(9:16));
        if length(tmp1)==3,
          HDR.T0(4:6) = tmp1; 
        end;
        if any(isnan(HDR.T0)),
          HDR.ErrNum = [2,HDR.ErrNum];
        end;
      end;
      
      % Y2K compatibility until year 2084
      if HDR.T0(1) < 85    % for biomedical data recorded in the 1950's and converted to EDF
        HDR.T0(1) = 2000+HDR.T0(1);
      elseif HDR.T0(1) < 100
        HDR.T0(1) = 1900+HDR.T0(1);
        %else % already corrected, do not change
      end;
      
      HDR.HeadLen = str2double(H1(185:192));           % 8 Bytes  Length of Header
      HDR.reserved1=H1(193:236);              % 44 Bytes reserved   
      HDR.NRec    = str2double(H1(237:244));     % 8 Bytes  # of data records
      HDR.Dur     = str2double(H1(245:252));     % 8 Bytes  # duration of data record in sec
      HDR.NS      = str2double(H1(253:256));     % 4 Bytes  # of signals
      HDR.AS.H1 = H1;	                     % for debugging the EDF Header
      
      if strcmp(HDR.reserved1(1:4),'EDF+'),	% EDF+ specific header information 
        [HDR.Patient.Id,   tmp] = strtok(HDR.PID,' ');
        [sex, tmp] = strtok(tmp,' ');
        [bd, tmp] = strtok(tmp,' ');
        [HDR.Patient.Name, tmp] = strtok(tmp,' ');
        if length(sex)>0,
          HDR.Patient.Sex = any(sex(1)=='mM') + any(sex(1)=='Ff')*2;
        else
          HDR.Patient.Sex = 0; % unknown 
        end; 
        if (length(bd)==11),
          HDR.Patient.Birthday = zeros(1,6); 
          bd(bd=='-') = ' '; 
          [n,v,s] = str2double(bd,' ');
          month_of_birth = strmatch(lower(s{2}),{'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'},'exact');
          if ~isempty(month_of_birth)
            v(2) = 0;
          end
          if any(v)
            HDR.Patient.Birthday(:) = NaN;
          else
            HDR.Patient.Birthday(1) = n(3);
            HDR.Patient.Birthday(2) = month_of_birth;
            HDR.Patient.Birthday(3) = n(1);
            HDR.Patient.Birthday(4) = 12;
          end;
        end; 
        
        [chk, tmp] = strtok(HDR.RID,' ');
        if ~strcmp(chk,'Startdate')
          fprintf(HDR.FILE.stderr,'Warning SOPEN: EDF+ header is corrupted.\n');
        end;
        [HDR.Date2, tmp] = strtok(tmp,' ');
        [HDR.RID, tmp] = strtok(tmp,' ');
        [HDR.REC.Technician,  tmp] = strtok(tmp,' ');
        [HDR.REC.Equipment,     tmp] = strtok(tmp,' ');
      end;
    end;
    
    if any(size(HDR.NS)~=1) %%%%% not EDF because filled out with ASCII(0) - should be spaces
      fprintf(HDR.FILE.stderr, 'Warning SOPEN (GDF/EDF/BDF): invalid NS-value in header of %s\n',HDR.FileName);
      HDR.ErrNum=[1040,HDR.ErrNum];
      HDR.NS=1;
    end;
    % Octave assumes HDR.NS is a matrix instead of a scalare. Therefore, we need
    % Otherwise, eye(HDR.NS) will be executed as eye(size(HDR.NS)).
    HDR.NS = HDR.NS(1);     
    
    if isempty(HDR.HeadLen) %%%%% not EDF because filled out with ASCII(0) - should be spaces
      HDR.ErrNum=[1056,HDR.ErrNum];
      HDR.HeadLen=256*(1+HDR.NS);
    end;
    
    if isempty(HDR.NRec) %%%%% not EDF because filled out with ASCII(0) - should be spaces
      HDR.ErrNum=[1027,HDR.ErrNum];
      HDR.NRec = -1;
    end;
    
    if isempty(HDR.Dur) %%%%% not EDF because filled out with ASCII(0) - should be spaces
      HDR.ErrNum=[1088,HDR.ErrNum];
      HDR.Dur=30;
    end;
    
    if  any(HDR.T0>[2084 12 31 24 59 59]) || any(HDR.T0<[1985 1 1 0 0 0])
      HDR.ErrNum = [4, HDR.ErrNum];
    end;

    %%% Read variable Header %%%
    %if ~strcmp(HDR.VERSION(1:3),'GDF'),
    if ~strcmp(HDR.TYPE,'GDF'),
      idx1=cumsum([0 H2idx]);
      idx2=HDR.NS*idx1;
      
      h2=zeros(HDR.NS,256);
      [H2,count]=fread(HDR.FILE.FID,HDR.NS*256,'uint8');
      if count < HDR.NS*256 
        HDR.ErrNum=[8,HDR.ErrNum];
        return; 
      end;
      
      %tmp=find((H2<32) | (H2>126)); % would confirm 
      tmp = find((H2<32) | ((H2>126) & (H2~=255) & (H2~=181)& (H2~=230))); 
      if ~isempty(tmp) %%%%% not EDF because filled out with ASCII(0) - should be spaces
        H2(tmp) = 32; 
        HDR.ErrNum = [1026,HDR.ErrNum];
      end;
      
      for k=1:length(H2idx);
        %disp([k size(H2) idx2(k) idx2(k+1) H2idx(k)]);
        h2(:,idx1(k)+1:idx1(k+1))=reshape(H2(idx2(k)+1:idx2(k+1)),H2idx(k),HDR.NS)';
      end;
      h2=char(h2);

      HDR.Label      =    cellstr(h2(:,idx1(1)+1:idx1(2)));
      HDR.Transducer =    cellstr(h2(:,idx1(2)+1:idx1(3)));
      HDR.PhysDim    =    cellstr(h2(:,idx1(3)+1:idx1(4)));
      HDR.PhysMin    = str2double(cellstr(h2(:,idx1(4)+1:idx1(5))))';
      HDR.PhysMax    = str2double(cellstr(h2(:,idx1(5)+1:idx1(6))))';
      HDR.DigMin     = str2double(cellstr(h2(:,idx1(6)+1:idx1(7))))';
      HDR.DigMax     = str2double(cellstr(h2(:,idx1(7)+1:idx1(8))))';
      HDR.PreFilt    =            h2(:,idx1(8)+1:idx1(9));
      HDR.AS.SPR     = str2double(cellstr(h2(:,idx1(9)+1:idx1(10))));
      %if ~all(abs(HDR.VERSION)==[255,abs('BIOSEMI')]),
      if (HDR.VERSION ~= -1),
        HDR.GDFTYP     = 3*ones(1,HDR.NS);	%	datatype
      else
        HDR.GDFTYP     = (255+24)*ones(1,HDR.NS);	%	datatype
      end;
      
      if isempty(HDR.AS.SPR), 
        fprintf(HDR.FILE.stderr, 'Warning SOPEN (GDF/EDF/BDF): invalid SPR-value in header of %s\n',HDR.FileName);
        HDR.AS.SPR=ones(HDR.NS,1);
        HDR.ErrNum=[1028,HDR.ErrNum];
      end;
    elseif (HDR.NS>0)
      if (ftell(HDR.FILE.FID)~=256),
        error('position error');
      end;	 
      HDR.Label      =  char(fread(HDR.FILE.FID,[16,HDR.NS],'uint8')');		
      HDR.Transducer =  cellstr(char(fread(HDR.FILE.FID,[80,HDR.NS],'uint8')'));	
      
      if (HDR.NS<1),	% hack for a problem with Matlab 7.1.0.183 (R14) Service Pack 3

      elseif (HDR.VERSION < 1.9),
        HDR.PhysDim    =  char(fread(HDR.FILE.FID,[ 8,HDR.NS],'uint8')');
        HDR.PhysMin    =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');	
        HDR.PhysMax    =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');	
        tmp            =       fread(HDR.FILE.FID,[1,2*HDR.NS],'int32');
        HDR.DigMin     =  tmp((1:HDR.NS)*2-1);
        tmp            =       fread(HDR.FILE.FID,[1,2*HDR.NS],'int32');
        HDR.DigMax     =  tmp((1:HDR.NS)*2-1);

        HDR.PreFilt    =  char(fread(HDR.FILE.FID,[80,HDR.NS],'uint8')');	%	
        HDR.AS.SPR     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32')';	%	samples per data record
        HDR.GDFTYP     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32');	%	datatype

      else
        tmp	       =  char(fread(HDR.FILE.FID,[6,HDR.NS],'uint8')');
        % HDR.PhysDim    =  char(fread(HDR.FILE.FID,[6,HDR.NS],'uint8')');
        HDR.PhysDimCode =      fread(HDR.FILE.FID,[1,HDR.NS],'uint16');

        HDR.PhysMin    =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');	
        HDR.PhysMax    =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');	
        HDR.DigMin     =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');
        HDR.DigMax     =       fread(HDR.FILE.FID,[1,HDR.NS],'float64');
        if (HDR.VERSION < 2.22)
          HDR.PreFilt = char(fread(HDR.FILE.FID,[80-12,HDR.NS],'uint8')');	%
        else
          HDR.PreFilt = char(fread(HDR.FILE.FID,[80-12-4,HDR.NS],'uint8')');	%
          HDR.TOffset = fread(HDR.FILE.FID,[1,HDR.NS],'float32');		% 
        end;
        HDR.Filter.LowPass  =  fread(HDR.FILE.FID,[1,HDR.NS],'float32');	% 
        HDR.Filter.HighPass =  fread(HDR.FILE.FID,[1,HDR.NS],'float32');	%
        HDR.Filter.Notch    =  fread(HDR.FILE.FID,[1,HDR.NS],'float32');	%
        HDR.AS.SPR     =       fread(HDR.FILE.FID,[1,HDR.NS],'uint32')';	% samples per data record
        HDR.GDFTYP     =       fread(HDR.FILE.FID,[1,HDR.NS],'uint32');	        % datatype
        HDR.ELEC.XYZ   =       fread(HDR.FILE.FID,[3,HDR.NS],'float32')';	% datatype
        if (HDR.VERSION < 2.19)
          tmp    =       fread(HDR.FILE.FID,[HDR.NS, 1],'uint8');	        % datatype
          tmp(tmp==255) = NaN; 
          HDR.Impedance = 2.^(tmp/8);
          fseek(HDR.FILE.FID, HDR.NS*19, 'cof');	                        % datatype
        else 
          tmp    =       fread(HDR.FILE.FID,[5,HDR.NS],'float32');	% datatype
          ch     =       bitand(HDR.PhysDimCode, hex2dec('ffe0'))==4256;       % channel with voltage data  
          HDR.Impedance(ch) = tmp(1,ch);
          HDR.Impedance(~ch)= NaN;
          ch     =       bitand(HDR.PhysDimCode, hex2dec('ffe0'))==4288;       % channel with impedance data  
          HDR.fZ(ch) = tmp(1,ch);                                         % probe frequency
          HDR.fZ(~ch)= NaN;
        end;
      end;
    end;

    HDR.SPR=1;
    if (HDR.NS>0)
      if ~isfield(HDR,'THRESHOLD')
        HDR.THRESHOLD  = [HDR.DigMin',HDR.DigMax'];       % automated overflow detection 
        if 0 && (HDR.VERSION == 0) && HDR.FLAG.OVERFLOWDETECTION,   % in case of EDF and OVERFLOWDETECTION
          fprintf(2,'WARNING SOPEN(EDF): Physical Max/Min values of EDF data are not necessarily defining the dynamic range.\n'); 
          fprintf(2,'   Hence, OVERFLOWDETECTION might not work correctly. See also EEG2HIST and read \n'); 
          fprintf(2,'   http://dx.doi.org/10.1016/S1388-2457(99)00172-8 (A. Schlögl et al. Quality Control ... Clin. Neurophysiol. 1999, Dec; 110(12): 2165 - 2170).\n'); 
          fprintf(2,'   A copy is available here, too: http://pub.ist.ac.at/~schloegl/publications/neurophys1999_2165.pdf \n'); 
        end;
      end; 
      if any(HDR.PhysMax==HDR.PhysMin), HDR.ErrNum=[1029,HDR.ErrNum]; end;	
      if any(HDR.DigMax ==HDR.DigMin ), HDR.ErrNum=[1030,HDR.ErrNum]; end;	
      HDR.Cal = (HDR.PhysMax-HDR.PhysMin)./(HDR.DigMax-HDR.DigMin);
      HDR.Off = HDR.PhysMin - HDR.Cal .* HDR.DigMin;
      if any(~isfinite(HDR.Cal)),
        fprintf(2,'WARNING SOPEN(GDF/BDF/EDF): Scaling factor is not defined in following channels:\n');
        fprintf(2,'%d,',find(~isfinite(HDR.Cal)));fprintf(2,'\n');
        HDR.Cal(~isfinite(HDR.Cal))=1; 
        HDR.FLAG.UCAL = 1;  
      end;
      
      HDR.AS.SampleRate = HDR.AS.SPR / HDR.Dur;
      if all(CHAN>0)
        chan = CHAN(:)';
      elseif (CHAN==0)
        chan = 1:HDR.NS;
        if strcmp(HDR.TYPE,'EDF')
          if strcmp(HDR.reserved1(1:4),'EDF+')
            tmp = strmatch('EDF Annotations',HDR.Label);
            chan(tmp)=[];
          end; 
        end;
      end;	
      for k=chan,
        if (HDR.AS.SPR(k)>0)
          HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
        end;
      end;
      HDR.SampleRate = HDR.SPR/HDR.Dur;
      
      HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
      HDR.AS.bi  = [0;cumsum(HDR.AS.SPR(:))]; 
      HDR.AS.BPR = ceil(HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)'); 
      if any(HDR.AS.BPR ~= HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)');
        fprintf(2,'\nError SOPEN (GDF/EDF/BDF): block configuration in file %s not supported.\n',HDR.FileName);
      end;
      HDR.AS.SAMECHANTYP = all(HDR.AS.BPR == (HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)')) && ~any(diff(HDR.GDFTYP)); 
      HDR.AS.bpb = sum(ceil(HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)'));	% Bytes per Block
      HDR.Calib  = [HDR.Off; diag(HDR.Cal)];

    else  % (if HDR.NS==0)
      HDR.THRESHOLD = [];
      HDR.AS.SPR = [];
      HDR.Calib  = zeros(1,0); 
      HDR.AS.bpb = 0; 
      HDR.GDFTYP = [];
      HDR.Label  = {};
    end;

    if HDR.VERSION<1.9,
      HDR.Filter.LowPass = repmat(nan,1,HDR.NS);
      HDR.Filter.HighPass = repmat(nan,1,HDR.NS);
      HDR.Filter.Notch = repmat(nan,1,HDR.NS);
      for k=1:HDR.NS,
        tmp = HDR.PreFilt(k,:);
        
        ixh=strfind(tmp,'HP');
        ixl=strfind(tmp,'LP');
        ixn=strfind(tmp,'Notch');
        ix =strfind(lower(tmp),'hz');
        
        [v,c,errmsg]=sscanf(tmp,'%f - %f Hz');
        if (isempty(errmsg) && (c==2)),
          HDR.Filter.LowPass(k) = max(v);
          HDR.Filter.HighPass(k) = min(v);
        else 
          if any(tmp==';')
            [tok1,tmp] = strtok(tmp,';');
            [tok2,tmp] = strtok(tmp,';');
            [tok3,tmp] = strtok(tmp,';');
          else
            [tok1,tmp] = strtok(tmp,' ');
            [tok2,tmp] = strtok(tmp,' ');
            [tok3,tmp] = strtok(tmp,' ');
          end;
          [T1, F1 ] = strtok(tok1,': ');
          [T2, F2 ] = strtok(tok2,': ');
          [T3, F3 ] = strtok(tok3,': ');
          
          [F1 ] = strtok(F1,': ');
          [F2 ] = strtok(F2,': ');
          [F3 ] = strtok(F3,': ');
          
          F1(find(F1==','))='.';
          F2(find(F2==','))='.';
          F3(find(F3==','))='.';
          
          if strcmp(F1,'DC'), F1='0'; end;
          if strcmp(F2,'DC'), F2='0'; end;
          if strcmp(F3,'DC'), F3='0'; end;
          
          tmp = strfind(lower(F1),'hz');
          if ~isempty(tmp), F1=F1(1:tmp-1); end;
          tmp = strfind(lower(F2),'hz');
          if ~isempty(tmp), F2=F2(1:tmp-1); end;
          tmp = strfind(lower(F3),'hz');
          if ~isempty(tmp), F3=F3(1:tmp-1); end;

          tmp = str2double(F1); 
          if isempty(tmp),tmp=NaN; end; 
          if strcmp(T1,'LP'), 
            HDR.Filter.LowPass(k) = tmp;
          elseif strcmp(T1,'HP'), 
            HDR.Filter.HighPass(k)= tmp;
          elseif strcmp(T1,'Notch'), 
            HDR.Filter.Notch(k)   = tmp;
          end;
          tmp = str2double(F2); 
          if isempty(tmp),tmp=NaN; end; 
          if strcmp(T2,'LP'), 
            HDR.Filter.LowPass(k) = tmp;
          elseif strcmp(T2,'HP'), 
            HDR.Filter.HighPass(k)= tmp;
          elseif strcmp(T2,'Notch'), 
            HDR.Filter.Notch(k)   = tmp;
          end;
          tmp = str2double(F3); 
          if isempty(tmp),tmp=NaN; end; 
          if strcmp(T3,'LP'), 
            HDR.Filter.LowPass(k) = tmp;
          elseif strcmp(T3,'HP'), 
            HDR.Filter.HighPass(k)= tmp;
          elseif strcmp(T3,'Notch'), 
            HDR.Filter.Notch(k)   = tmp;
          end;
          %catch
          %        fprintf(2,'Cannot interpret: %s\n',HDR.PreFilt(k,:));
        end;
      end;
    end

    %% GDF Header 3 
    if (HDR.VERSION > 2)
      pos = 256*(HDR.NS+1); 
      fseek(HDR.FILE.FID,pos,'bof');	
      while (pos <= HDR.HeadLen-4)
        %% decode TAG-LENGTH-VALUE structure
        tagval = fread(HDR.FILE.FID,1,'uint32');
        TAG = bitand(tagval, 255);
        LEN = (tagval/256);
        if (pos+4+LEN > HDR.HeadLen)
          fprintf(HDR.FILE.stderr,'ERROR SOPEN(GDF): T-L-V header broken\n'); 
          break; 
        end; 
        switch (TAG)
         case 0		%% last tag-length-value 
          break; 	
         case 1		%% description of user-specified event codes
          VAL= fread(HDR.FILE.FID,[1,LEN],'uint8=>char');
          ix = find(VAL==0);
          N  = find(diff(ix)==1);
          if isempty(N) 
            N = length(ix); 
            ix(N+1)=LEN+1;
          end;	
          for k = 1:N(1),
            HDR.EVENT.CodeDesc{k} = VAL(ix(k)+1:ix(k+1)-1);
          end; 
         case 2		%% bci2000 additional information 
          VAL = fread(HDR.FILE.FID,[1,LEN],'uint8=>char');
          HDR.BCI2000.INFO = VAL;
         case 3		%% Manufacturer Information 
          VAL = fread(HDR.FILE.FID,[1,LEN],'uint8=>char');
          [HDR.Manufacturer.Name,VAL] = strtok(VAL,0); 
          [HDR.Manufacturer.Model,VAL] = strtok(VAL,0); 
          [HDR.Manufacturer.Version,VAL] = strtok(VAL,0); 
          [HDR.Manufacturer.SerialNumber,VAL] = strtok(VAL,0); 
          % case 4	%% OBSOLETE %% Orientation of MEG channels 
          %VAL = fread(HDR.FILE.FID,[HDR.NS,4],'float32');
          %HDR.ELEC.Orientation = VAL(:,1:3);
          %HDR.ELEC.Area = VAL(:,4);
         case 5 		%% IP address
          VAL = fread(HDR.FILE.FID,[1,LEN],'uint8=>uint8');
          HDR.REC.IPaddr = VAL;
         case 6 		%% recording Technician
          VAL = fread(HDR.FILE.FID,[1,LEN],'uint8=>char');
          HDR.REC.Technician = VAL;
         case 7 		%% recording institution/hospital/lab
          VAL = fread(HDR.FILE.FID,[1,LEN],'uint8=>char');
          HDR.REC.Hospital = VAL;
         otherwise 
          fseek(HDR.FILE.FID,LEN,'cof'); 
        end; 
        pos = ftell(HDR.FILE.FID); 
      end; 			
    end; 

    % filesize, position of eventtable, headerlength, etc. 	
    HDR.AS.EVENTTABLEPOS = -1;
    if (HDR.FILE.size == HDR.HeadLen)
      HDR.NRec = 0; 
    elseif HDR.NRec == -1   % unknown record size, determine correct NRec
      HDR.NRec = floor((HDR.FILE.size - HDR.HeadLen) / HDR.AS.bpb);
    end
    if  (HDR.NRec*HDR.AS.bpb) ~= (HDR.FILE.size - HDR.HeadLen);
      %if ~strcmp(HDR.VERSION(1:3),'GDF'),
      if ~strcmp(HDR.TYPE,'GDF'),
        HDR.ErrNum= [16,HDR.ErrNum];
        tmp = HDR.NRec; 
        HDR.NRec = floor((HDR.FILE.size - HDR.HeadLen) / HDR.AS.bpb);
        if tmp~=HDR.NRec,
          fprintf(2,'\nWarning SOPEN (EDF/BDF): filesize (%i) of %s does not fit headerinformation (NRec = %i not %i).\n',HDR.FILE.size,HDR.FileName,tmp,HDR.NRec);
        else
          fprintf(2,'\nWarning: incomplete data block appended (ignored) in file %s.\n',HDR.FileName);
        end
      else
        HDR.AS.EVENTTABLEPOS = HDR.HeadLen + HDR.AS.bpb*HDR.NRec;
      end;
    end; 
    
    % prepare SREAD for different data types 
    n = 0; 
    typ = [-1;HDR.GDFTYP(:)];
    for k = 1:HDR.NS; 
      if (typ(k) == typ(k+1)),
        HDR.AS.c(n)   = HDR.AS.c(n)  + HDR.AS.SPR(k);
        HDR.AS.c2(n)  = HDR.AS.c2(n) + HDR.AS.BPR(k);
      else
        n = n + 1; 
        HDR.AS.c(n)   = HDR.AS.SPR(k);
        HDR.AS.c2(n)  = HDR.AS.BPR(k);
        HDR.AS.TYP(n) = HDR.GDFTYP(k);
      end;
    end;
    
    if 0, 
      
    elseif strcmp(HDR.TYPE,'GDF') && (HDR.AS.EVENTTABLEPOS > 0),  
      status = fseek(HDR.FILE.FID, HDR.AS.EVENTTABLEPOS, 'bof');
      if (HDR.VERSION<1.94),
        [EVENT.Version,c] = fread(HDR.FILE.FID,1,'uint8');
        HDR.EVENT.SampleRate = [1,256,65536]*fread(HDR.FILE.FID,3,'uint8');
        [EVENT.N,c] = fread(HDR.FILE.FID,1,'uint32');
      else %if HDR.VERSION<1.94,
        [EVENT.Version,c] = fread(HDR.FILE.FID,1,'uint8');
        EVENT.N = [1,256,65536]*fread(HDR.FILE.FID,3,'uint8');
        [HDR.EVENT.SampleRate,c] = fread(HDR.FILE.FID,1,'float32');
      end;	

      if ~HDR.EVENT.SampleRate, % ... is not defined in GDF 1.24 or earlier
        HDR.EVENT.SampleRate = HDR.SampleRate; 
      end;
      [HDR.EVENT.POS,c1] = fread(HDR.FILE.FID,[EVENT.N,1],'uint32');
      [HDR.EVENT.TYP,c2] = fread(HDR.FILE.FID,[EVENT.N,1],'uint16');

      if EVENT.Version==1,
        if any([c1,c2]~=EVENT.N)
          fprintf(2,'\nERROR SOPEN (GDF): Eventtable corrupted in file %s\n',HDR.FileName);
        end
        
      elseif EVENT.Version==3,
        [HDR.EVENT.CHN,c3] = fread(HDR.FILE.FID,[EVENT.N,1],'uint16');
        [HDR.EVENT.DUR,c4] = fread(HDR.FILE.FID,[EVENT.N,1],'uint32');
        %[EVENT.N,HDR.FILE.size,HDR.AS.EVENTTABLEPOS+8+EVENT.N*12]
        if any([c1,c2,c3,c4]~=EVENT.N),
          fprintf(2,'\nERROR SOPEN (GDF): Eventtable corrupted in file %s\n',HDR.FileName);
        end;
        
      else
        fprintf(2,'\nWarning SOPEN (GDF): File %s corrupted (Eventtable version %i ).\n',HDR.FileName,EVENT.Version);
      end;
      HDR.AS.endpos = HDR.AS.EVENTTABLEPOS;   % set end of data block, might be important for SSEEK

      % Classlabels according to 
      % http://biosig.cvs.sourceforge.net/*checkout*/biosig/biosig/doc/eventcodes.txt
      % sort event table before extracting HDR.Classlabel and HDR.TRIG
      [HDR.EVENT.POS,ix] = sort(HDR.EVENT.POS);
      HDR.EVENT.TYP = HDR.EVENT.TYP(ix);
      if isfield(HDR.EVENT,'CHN')
        HDR.EVENT.CHN = HDR.EVENT.CHN(ix);
      end;    	    
      if isfield(HDR.EVENT,'DUR')
        HDR.EVENT.DUR = HDR.EVENT.DUR(ix);
      end; 	

      if (length(HDR.EVENT.TYP)>0)
        ix = (HDR.EVENT.TYP>hex2dec('0300')) & (HDR.EVENT.TYP<hex2dec('030d'));
        ix = ix | ((HDR.EVENT.TYP>=hex2dec('0320')) & (HDR.EVENT.TYP<=hex2dec('037f')));
        ix = ix | (HDR.EVENT.TYP==hex2dec('030f')); % unknown/undefined cue
        HDR.Classlabel = mod(HDR.EVENT.TYP(ix),256);
        HDR.Classlabel(HDR.Classlabel==15) = NaN; % unknown/undefined cue
      end;

      % Trigger information and Artifact Selection 
      ix = find(HDR.EVENT.TYP==hex2dec('0300')); 
      HDR.TRIG = HDR.EVENT.POS(ix);
      ArtifactSelection = repmat(logical(0),length(ix),1);
      for k = 1:length(ix),
        ix2 = find(HDR.EVENT.POS(ix(k))==HDR.EVENT.POS);
        if any(HDR.EVENT.TYP(ix2)==hex2dec('03ff'))
          ArtifactSelection(k) = logical(1);                
        end;
      end;
      if any(ArtifactSelection), % define only if necessary
        HDR.ArtifactSelection = ArtifactSelection; 
      end;
      % decode non-equidistant sampling
      ix = find(HDR.EVENT.TYP==hex2dec('7fff'));
      if ~isempty(ix),
        if (HDR.VERSION<1.90), 
          warning('non-equidistant sampling not definded for GDF v1.x')
        end;
        % the following is redundant information %
        HDR.EVENT.VAL = repmat(NaN,size(HDR.EVENT.TYP));
        HDR.EVENT.VAL(ix) = HDR.EVENT.DUR(ix);
      end;

    elseif strcmp(HDR.TYPE,'EDF') && (length(strmatch('EDF Annotations',HDR.Label))==1),
      % EDF+: 
      tmp = strmatch('EDF Annotations',HDR.Label);
      HDR.EDF.Annotations = tmp;
      if 0,isempty(ReRefMx)
        ReRefMx = sparse(1:HDR.NS,1:HDR.NS,1);
        ReRefMx(:,tmp) = [];
      end;	
      
      status = fseek(HDR.FILE.FID,HDR.HeadLen+HDR.AS.bi(HDR.EDF.Annotations)*2,'bof');
      t = fread(HDR.FILE.FID,inf,[int2str(HDR.AS.SPR(HDR.EDF.Annotations)*2),'*uchar=>uchar'],HDR.AS.bpb-HDR.AS.SPR(HDR.EDF.Annotations)*2);
      HDR.EDF.ANNONS = char(t');
      
      N = 0; 
      onset = []; dur=[]; Desc = {};
      [s,t] = strtok(HDR.EDF.ANNONS,0);
      while ~isempty(s)
        N  = N + 1; 
        ix = find(s==20);
        [s1,s2] = strtok(s(1:ix(1)-1),21);
        s1;
        tmp = str2double(s1);
        onset(N,1) = tmp;
        tmp = str2double(s2(2:end));
        if  ~isempty(tmp)
          dur(N,1) = tmp; 	
        else 
          dur(N,1) = 0; 	
        end;
        Desc{N} = char(s(ix(1)+1:end-1));
        [s,t] = strtok(t,0);
        HDR.EVENT.TYP(N,1) = length(Desc{N});
      end;		
      HDR.EVENT.POS = round(onset * HDR.SampleRate);
      HDR.EVENT.DUR = dur * HDR.SampleRate;
      HDR.EVENT.CHN = zeros(N,1); 
      [HDR.EVENT.CodeDesc, CodeIndex, HDR.EVENT.TYP] = unique(Desc(1:N)');


    elseif strcmp(HDR.TYPE,'EDF') && (length(strmatch('ANNOTATION',HDR.Label))==1),
      % EEG from Delta/NihonKohden converted into EDF: 
      tmp = strmatch('ANNOTATION',HDR.Label);
      HDR.EDF.Annotations = tmp;
      FLAG.ANNONS = 1; % valid         
      
      status = fseek(HDR.FILE.FID,HDR.HeadLen+HDR.AS.bi(HDR.EDF.Annotations)*2,'bof');
      t = fread(HDR.FILE.FID,inf,[int2str(HDR.AS.SPR(HDR.EDF.Annotations)*2),'*uchar=>uchar'],HDR.AS.bpb-HDR.AS.SPR(HDR.EDF.Annotations)*2);
      t = reshape(t,HDR.AS.SPR(HDR.EDF.Annotations)*2,HDR.NRec)'; 
      t = t(any(t,2),1:max(find(any(t,1))));
      HDR.EDF.ANNONS = char(t);

      N = 0;
      [t,r] = strtok(char(reshape(t',[1,prod(size(t))])),[0,64]);
      while ~isempty(r),
        [m,r] = strtok(r,[0,64]);
        tb = char([t(1:4),32,t(5:6),32,t(7:8),32,t(9:10),32,t(11:12),32,t(13:end)]);
        [ta, status] = str2double(tb);
        t1 = datenum(ta);
        
        if any(status),
          FLAG.ANNONS = 0; % invalid         
        elseif strcmp(char(m),'TEST');
          t0 = t1;
        elseif ((length(m)==1) && ~isnan(t1-t0)),
          N = N+1;
          HDR.EVENT.POS(N,1) = t1;
          HDR.EVENT.TYP(N,1) = abs(m);
        else
          FLAG.ANNONS = 0; % invalid         
        end;
        [t,r] = strtok(r,[0,64]);
      end;
      HDR.EVENT.POS = round((HDR.EVENT.POS-t0)*(HDR.SampleRate*3600*24))+2;

      if FLAG.ANNONS;
        % decoding was successful
        if isempty(ReRefMx)
          % do not return annotations in signal matrix 
          ReRefMx = sparse(1:HDR.NS,1:HDR.NS,1);
          ReRefMx(:,HDR.EDF.Annotations) = [];
        end;
      end;

    elseif strcmp(HDR.TYPE,'BDF') && any(strmatch('Status',HDR.Label)),
      % BDF: 

      tmp = strmatch('Status',HDR.Label);
      HDR.BDF.Status.Channel = tmp;
      if isempty(ReRefMx)
        ReRefMx = sparse(1:HDR.NS,1:HDR.NS,1);
        ReRefMx(:,tmp) = [];
      end;	

      status = fseek(HDR.FILE.FID,HDR.HeadLen+HDR.AS.bi(HDR.BDF.Status.Channel)*3,'bof');
      %t = fread(HDR.FILE.FID,[3,inf],'uint8',HDR.AS.bpb-HDR.AS.SPR(HDR.BDF.Status.Channel)*3);
      [t,c] = fread(HDR.FILE.FID,inf,[int2str(HDR.AS.SPR(HDR.BDF.Status.Channel)*3),'*uint8'],HDR.AS.bpb-HDR.AS.SPR(HDR.BDF.Status.Channel)*3);
      if (c>HDR.NRec*HDR.SPR*3)
        % a hack to fix a bug in Octave Ver<=3.0.0
        t = t(1:HDR.NRec*HDR.SPR*3);
      end;
      HDR.BDF.ANNONS = reshape(double(t),3,length(t)/3)'*2.^[0;8;16];
      HDR = bdf2biosig_events(HDR); 

    elseif strcmp(HDR.TYPE,'BDF') && ~any(strmatch('Status',HDR.Label)),
      HDR.FLAG.OVERFLOWDETECTION = 0; 
      fprintf(HDR.FILE.stderr,'Warning SOPEN(BDF): File %s does not contain Status Channel - overflowdetection not supported!\n',HDR.FileName);
      
    end;
    
    status = fseek(HDR.FILE.FID, HDR.HeadLen, 'bof');
    HDR.FILE.POS  = 0;
    HDR.FILE.OPEN = 1;
    
    %%% Event file stored in GDF-format
    if ~any([HDR.NS,HDR.NRec,~length(HDR.EVENT.POS)]);
      HDR.TYPE = 'EVENT';
      HDR = sclose(HDR);
    end;	
    
  elseif any(HDR.FILE.PERMISSION=='w');                %%%%%%% ============= WRITE ===========%%%%%%%%%%%%        
    if strcmp(HDR.TYPE,'EDF')
      HDR.VERSION = 0;
    elseif strcmp(HDR.TYPE,'GDF')
      if ~isfield(HDR,'VERSION')
        HDR.VERSION = 2.21;     %% default version 
      elseif (HDR.VERSION < 1.30)
        HDR.VERSION = 1.25;     %% old version 
      elseif (HDR.VERSION < 2.19)
        HDR.VERSION = 2.11;     %% stable version 
      else
        HDR.VERSION = 2.22;     %% experimental 
      end;        
    elseif strcmp(HDR.TYPE,'BDF'),
      HDR.VERSION = -1;
    end;

    if ~isfield(HDR,'RID')
      HDR.RID=char(32+zeros(1,80));
    end;
    if ~isfield(HDR,'T0')
      HDR.T0=zeros(1,6);
      fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.T0 not defined\n');
    elseif any(isnan(HDR.T0))
      HDR.T0(isnan(HDR.T0))=0;
      fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.T0 not completely defined\n');
    end;
    if ~isfield(HDR,'Patient')
      HDR.Patient.Sex = 0; 
      HDR.Patient.Handedness = 0; 
      HDR.Patient.Birthday = zeros(1,6);
      HDR.Patient.Headsize = [NaN, NaN, NaN]; 
      HDR.Patient.Weight = 0;
      HDR.Patient.Height = 0;
    end;
    if ~isfield(HDR.Patient,'Name')
      HDR.Patient.Name = 'X'; 
    end;
    if ~isfield(HDR.Patient,'Id')
      HDR.Patient.Id = 'X'; 
    end;
    if ~isfield(HDR.Patient,'Sex')
      HDR.Patient.Sex = 0; 
    elseif isnumeric(HDR.Patient.Sex)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.Sex,'m') || strcmpi(HDR.Patient.Sex,'male')
      HDR.Patient.Sex = 1; 
    elseif strcmpi(HDR.Patient.Sex,'f') || strcmpi(HDR.Patient.Sex,'female')
      HDR.Patient.Sex = 2; 
    else
      HDR.Patient.Sex = 0; 
    end;
    
    if ~isfield(HDR.Patient,'Handedness')
      HDR.Patient.Handedness = 0; 
    elseif isnumeric(HDR.Patient.Handedness)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.Handedness,'r') || strcmpi(HDR.Patient.Handedness,'right')
      HDR.Patient.Handedness = 1; 
    elseif strcmpi(HDR.Patient.Handedness,'l') || strcmpi(HDR.Patient.Handedness,'left')
      HDR.Patient.Handedness = 2; 
    else	
      HDR.Patient.Handedness = 0; 
    end;
    if ~isfield(HDR.Patient,'Impairment.Visual')
      HDR.Patient.Impairment.Visual = 0;
    elseif isnumeric(HDR.Patient.Impairment.Visual)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.Impairment.Visual,'NO') || strcmpi(HDR.Patient.Impairment.Visual,'NO')
      HDR.Patient.Impairment.Visual = 1;
    elseif strcmpi(HDR.Patient.Impairment.Visual,'Y') || strcmpi(HDR.Patient.Impairment.Visual,'YES')
      HDR.Patient.Impairment.Visual = 2;
    elseif strncmpi(HDR.Patient.Impairment.Visual,'corr',4)
      HDR.Patient.Impairment.Visual = 3;
    elseif isnumeric(HDR.Patient.Impairment.Visual)
    else 
      HDR.Patient.Impairment.Visual = 0;
    end;
    if ~isfield(HDR.Patient,'Smoking')
      HDR.Patient.Smoking = 0;
    elseif isnumeric(HDR.Patient.Smoking)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.Smoking,'NO') || strcmpi(HDR.Patient.Smoking,'NO')
      HDR.Patient.Smoking = 1;
    elseif strcmpi(HDR.Patient.Smoking,'Y') || strcmpi(HDR.Patient.Smoking,'YES')
      HDR.Patient.Smoking = 2;
    elseif isnumeric(HDR.Patient.Smoking)
    else 
      HDR.Patient.Smoking = 0;
    end;
    if ~isfield(HDR.Patient,'AlcoholAbuse')
      HDR.Patient.AlcoholAbuse = 0;
    elseif isnumeric(HDR.Patient.AlcoholAbuse)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.AlcoholAbuse,'NO') || strcmpi(HDR.Patient.AlcoholAbuse,'NO')
      HDR.Patient.AlcoholAbuse = 1;
    elseif strcmpi(HDR.Patient.AlcoholAbuse,'Y') || strcmpi(HDR.Patient.AlcoholAbuse,'YES')
      HDR.Patient.AlcoholAbuse = 2;
    elseif isnumeric(HDR.Patient.AlcoholAbuse)
    else 
      HDR.Patient.AlcoholAbuse = 0;
    end;
    if ~isfield(HDR.Patient,'DrugAbuse')
      HDR.Patient.DrugAbuse = 0;
    elseif isnumeric(HDR.Patient.DrugAbuse)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.DrugAbuse,'NO') || strcmpi(HDR.Patient.DrugAbuse,'NO')
      HDR.Patient.DrugAbuse = 1;
    elseif strcmpi(HDR.Patient.DrugAbuse,'Y') || strcmpi(HDR.Patient.DrugAbuse,'YES')
      HDR.Patient.DrugAbuse = 2;
    elseif isnumeric(HDR.Patient.DrugAbuse)
    else 
      HDR.Patient.DrugAbuse = 0;
    end;
    if ~isfield(HDR.Patient,'Medication')
      HDR.Patient.Medication = 0;
    elseif isnumeric(HDR.Patient.Medication)
      ;       % nothing to be done
    elseif strcmpi(HDR.Patient.Medication,'NO') || strcmpi(HDR.Patient.Medication,'NO')
      HDR.Patient.Medication = 1;
    elseif strcmpi(HDR.Patient.Medication,'Y') || strcmpi(HDR.Patient.Medication,'YES')
      HDR.Patient.Medication = 2;
    else 
      HDR.Patient.Medication = 0;
    end;
    if ~isfield(HDR.Patient,'Weight')
      HDR.Patient.Weight = 0; 
    elseif (HDR.Patient.Weight > 254),
      HDR.Patient.Weight = 255; 
    elseif isnan(HDR.Patient.Weight) || (isnan(HDR.Patient.Weight)<0)
      HDR.Patient.Weight = 0; 
    end;
    if ~isfield(HDR.Patient,'Height')
      HDR.Patient.Height = 0; 
    elseif (HDR.Patient.Height > 254),
      HDR.Patient.Height = 255; 
    elseif isnan(HDR.Patient.Height) || (isnan(HDR.Patient.Height)<0)
      HDR.Patient.Height = 0; 
    end;
    if ~isfield(HDR.Patient,'Birthday') 
      if ~isfield(HDR.Patient,'Age')
        HDR.Patient.Birthday = zeros(1,6);
      elseif isnan(HDR.Patient.Age) 
        HDR.Patient.Birthday = zeros(1,6);
      else
        HDR.Patient.Birthday = datevec(datenum(HDR.T0) + HDR.Patient.Age*365.25);
      end;	
    end;
    if ~isfield(HDR.Patient,'Headsize')
      HDR.Patient.Headsize = [NaN,NaN,NaN]; 
    elseif ~isnumeric(HDR.Patient.Headsize)
      fprintf('Warning SOPEN (GDF)-W: HDR.Patient.Headsize must be numeric.\n');
    elseif (numel(HDR.Patient.Headsize)~=3)
      tmp = [HDR.Patient.Headsize(:);NaN;NaN;NaN]';
      HDR.Patient.Headsize = HDR.Patient.Headsize(1:3); 
    end;
    if ~isfield(HDR,'ELEC')
      HDR.ELEC.XYZ = repmat(0,HDR.NS,3); 
      HDR.ELEC.GND = zeros(1,3); 
      HDR.ELEC.REF = zeros(1,3); 
    end;
    tmp = uint32([hex2dec('00292929'),48*36e5+2^31,15*36e5+2^31,35000]);
    if ~isfield(HDR,'REC')
      HDR.REC.LOC.RFC1876 = tmp; 
    elseif ~isfield(HDR.REC,'LOC');
      HDR.REC.LOC.RFC1876 = tmp; 
    elseif ~isfield(HDR.REC.LOC,'RFC1876')	
      tmp = HDR.REC.LOC;
      HDR.REC.LOC.RFC1876 = [hex2dec('00292929'),tmp.Latitude*36e5,tmp.Longitude*36e5,tmp.Altitude*100];
    end
    if isfield(HDR.REC,'Impedance') && ~isfield(HDR,'Impedance')	
      HDR.Impedance = HDR.REC.Impedance; 
    end
    if ~isfield(HDR,'Impedance')	
      HDR.Impedance = repmat(NaN,HDR.NS,1); 
    end
    HDR.REC.Equipment = [1,abs('BIOSIG ')];
    if ~isfield(HDR.REC,'Lab')
      HDR.REC.Lab = repmat(32,1,8);
    end;
    if ~isfield(HDR.REC,'IPaddr')
      HDR.REC.IPaddr = uint8(zeros(1,6));
    end;
    if ~isfield(HDR.REC,'Technician')
      HDR.REC.Technician = repmat(32,1,8);
    end;
    
    if ~isfield(HDR,'NRec')
      HDR.NRec=-1;
    end;
    if ~isfield(HDR,'Dur')
      if HDR.NS>0,
        fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.Dur not defined\n');
      end;
      HDR.Dur=NaN;
    end;
    if ~isfield(HDR,'NS')
      HDR.ErrMsg = sprintf('Error SOPEN (GDF/EDF/BDF)-W: HDR.NS not defined\n');
      HDR.ErrNum = HDR.ErrNum + 128;
      return;
    end;
    if ~isfield(HDR,'SampleRate')
      HDR.SampleRate = NaN;
    end;
    if ~isnan(HDR.Dur) && ~isnan(HDR.SampleRate) && ~isfield(HDR,'SPR'),
      HDR.SPR = HDR.SampleRate*HDR.Dur; 
    end

    if ~isfield(HDR,'AS')
      HDR.AS.SPR = repmat(NaN,1,HDR.NS);
    end;
    if ~isfield(HDR.AS,'SPR')
      HDR.AS.SPR = repmat(NaN,1,HDR.NS);
    end;
    if isfield(HDR,'SPR')
      HDR.AS.SPR(isnan(HDR.AS.SPR)) = HDR.SPR;
    elseif all(~isnan(HDR.AS.SPR))
      HDR.SPR = 1; 
      for k=1:HDR.NS
        if HDR.AS.SPR(k),
          HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
        end;	 
      end; 
    else 
      warning('either HDR.SPR or HDR.AS.SPR must be defined');
    end;  
    if ~isfield(HDR.AS,'SampleRate')
      HDR.AS.SampleRate = HDR.SampleRate*HDR.AS.SPR/HDR.SPR;
    end;
    
    if ~HDR.NS,
    elseif ~isnan(HDR.Dur) && any(isnan(HDR.AS.SPR)) && ~any(isnan(HDR.AS.SampleRate))
      HDR.AS.SPR = HDR.AS.SampleRate * HDR.Dur;
    elseif ~isnan(HDR.Dur) && ~any(isnan(HDR.AS.SPR)) && any(isnan(HDR.AS.SampleRate))
      HDR.SampleRate = HDR.Dur * HDR.AS.SPR;
    elseif isnan(HDR.Dur) && ~any(isnan(HDR.AS.SPR)) && ~any(isnan(HDR.AS.SampleRate))
      HDR.Dur = HDR.AS.SPR(:) ./ HDR.AS.SampleRate(:);
      if all((HDR.Dur(1)-HDR.Dur)<5*eps)
        HDR.Dur = HDR.Dur(1);
      else
        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF): SPR and SampleRate do not fit\n');
      end;
    elseif ~isnan(HDR.Dur) && ~any(isnan(HDR.AS.SPR)) && ~any(isnan(HDR.AS.SampleRate))
      %% thats ok, 
    else
      fprintf(HDR.FILE.stderr,'ERROR SOPEN (GDF/EDF/BDF): more than 1 of HDR.Dur, HDR.SampleRate, HDR.AS.SPR undefined.\n');
      return; 
    end;
    
    %if (abs(HDR.VERSION(1))==255)  && strcmp(HDR.VERSION(2:8),'BIOSEMI'),
    if (HDR.VERSION == -1),
      HDR.GDFTYP=255+24+zeros(1,HDR.NS);                        
      %elseif strcmp(HDR.VERSION,'0       '),
    elseif HDR.VERSION == 0,
      HDR.GDFTYP=3+zeros(1,HDR.NS);                        
      %elseif strcmp(HDR.VERSION(1:3),'GDF'),
    elseif (HDR.VERSION>0),
      if HDR.NS == 0;
        HDR.GDFTYP = 3;
      elseif ~isfield(HDR,'GDFTYP'),
        HDR.ErrMsg = sprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.GDFTYP not defined\n');
        HDR.ErrNum = HDR.ErrNum + 128;
        % fclose(HDR.FILE.FID); return;
      elseif length(HDR.GDFTYP)==1,
        HDR.GDFTYP = HDR.GDFTYP(ones(HDR.NS,1));
      elseif length(HDR.GDFTYP)>=HDR.NS,
        HDR.GDFTYP = HDR.GDFTYP(1:HDR.NS);
      end;
    else
      fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF): invalid VERSION %s\n ',HDR.VERSION);
      return;
    end;
    [tmp,HDR.THRESHOLD]=gdfdatatype(HDR.GDFTYP);
    
    if (HDR.NS>0),	% header 2
                     % Check all fields of Header2
      Label = repmat(' ',HDR.NS,16); 
      if isfield(HDR,'Label')
        if ischar(HDR.Label)
          sz = min([HDR.NS,16],size(HDR.Label)); 
          Label(1:sz(1),1:sz(2)) = HDR.Label(1:sz(1),1:sz(2));
        elseif iscell(HDR.Label)
          for k=1:min(HDR.NS,length(HDR.Label))
            tmp = [HDR.Label{k},' ']; 
            sz = min(16,length(tmp)); 
            Label(k,1:sz)=tmp(1:sz);
          end; 
        end; 
      else
        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.Label not defined\n');
      end; 

      if ~isfield(HDR,'Transducer')
        HDR.Transducer=repmat({' '},HDR.NS,1); %char(32+zeros(HDR.NS,80));
      elseif ischar(HDR.Transducer) 
        HDR.Transducer = cellstr(HDR.Transducer);
      end; 
      Transducer = char(HDR.Transducer); 
      tmp = min(80,size(Transducer,2));
      Transducer = [Transducer, repmat(' ',size(Transducer,1),80-tmp)];
      Transducer = Transducer(:,1:80);                        
      
      if ~isfield(HDR,'Filter')
        HDR.Filter.LowPass = repmat(NaN,1,HDR.NS); 
        HDR.Filter.HighPass = repmat(NaN,1,HDR.NS); 
        HDR.Filter.Notch = repmat(NaN,1,HDR.NS); 
      else 
        if ~isfield(HDR.Filter,'LowPass')
          HDR.Filter.LowPass = repmat(NaN,1,HDR.NS); 
        elseif (numel(HDR.Filter.LowPass)==1)
          HDR.Filter.LowPass = repmat(HDR.Filter.LowPass,1,HDR.NS); 
        elseif (numel(HDR.Filter.LowPass)~=HDR.NS)
          fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.LowPass has incorrrect number of fields!\n')
        end;
        if ~isfield(HDR.Filter,'HighPass')
          HDR.Filter.HighPass = repmat(NaN,1,HDR.NS); 
        elseif (numel(HDR.Filter.HighPass)==1)
          HDR.Filter.HighPass = repmat(HDR.Filter.HighPass,1,HDR.NS); 
        elseif (numel(HDR.Filter.HighPass)~=HDR.NS)
          fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.HighPass has incorrrect number of fields!\n')
        end;
        if ~isfield(HDR.Filter,'Notch')
          HDR.Filter.Notch = repmat(NaN,1,HDR.NS); 
        elseif (numel(HDR.Filter.Notch)==1)
          HDR.Filter.Notch = repmat(HDR.Filter.Notch,1,HDR.NS); 
        elseif (numel(HDR.Filter.Notch)~=HDR.NS)
          fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.Notch has incorrrect number of fields!\n')
        end;
      end;
      if ~isfield(HDR,'PreFilt')
        HDR.PreFilt = char(32+zeros(HDR.NS,80));
        if isfield(HDR,'Filter'),
          if isfield(HDR.Filter,'LowPass') && isfield(HDR.Filter,'HighPass') && isfield(HDR.Filter,'Notch'),
            if any(length(HDR.Filter.LowPass) == [1,HDR.NS]) && any(length(HDR.Filter.HighPass) == [1,HDR.NS]) && any(length(HDR.Filter.Notch) == [1,HDR.NS])
              PreFilt = {};
              for k = 1:HDR.NS,
                k1 = min(k,length(HDR.Filter.LowPass));
                k2 = min(k,length(HDR.Filter.HighPass));
                k3 = min(k,length(HDR.Filter.Notch));
                PreFilt{k,1} = sprintf('LP: %5.f Hz; HP: %5.2f Hz; Notch: %i',HDR.Filter.LowPass(k1),HDR.Filter.HighPass(k2),HDR.Filter.Notch(k3));
              end;
              HDR.PreFilt = char(PreFilt);
            end;
          end
        end
      elseif size(HDR.PreFilt,1)<HDR.NS,
        HDR.PreFilt = repmat(HDR.PreFilt,HDR.NS,1);
      end;
      tmp = min(80,size(HDR.PreFilt,2));
      HDR.PreFilt = [HDR.PreFilt(1:HDR.NS,1:tmp), char(32+zeros(HDR.NS,80-tmp))];

      if isfield(HDR,'PhysDimCode')
        HDR.PhysDimCode = HDR.PhysDimCode(1:HDR.NS);
      end;	
      PhysDim = char(32+zeros(HDR.NS,8));
      if ~isfield(HDR,'PhysDim')
        HDR.PhysDim=repmat({' '},HDR.NS,1);
        PhysDim = char(32+zeros(HDR.NS,8));
        if HDR.NS>0,
          fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysDim not defined\n');
        end;
      else
        if iscell(HDR.PhysDim),  
          % make column 
          HDR.PhysDim = HDR.PhysDim(:); 
        end;  
        if length(HDR.PhysDim)==0,
          HDR.PhysDim = repmat({' '},HDR.NS,1);
        elseif size(HDR.PhysDim,1)<HDR.NS,
          HDR.PhysDim = repmat(HDR.PhysDim,HDR.NS,1);
        elseif size(HDR.PhysDim,1)>HDR.NS,
          HDR.PhysDim = HDR.PhysDim(1:HDR.NS); 
        end;
        PhysDim = char(HDR.PhysDim); % local copy 
        if size(PhysDim,1)~=HDR.NS,
          PhysDim = char(32+zeros(HDR.NS,8));
        end; 	                        
      end;
      tmp = min(8,size(PhysDim,2));
      PhysDim = [PhysDim(1:HDR.NS,1:tmp), char(32+zeros(HDR.NS,8-tmp))];

      HDR = physicalunits(HDR);
      if ~all(HDR.PhysDimCode>0)
        fprintf(HDR.FILE.stderr,'Warning SOPEN: HDR.PhysDimCode of the following channel(s) is(are) not defined:\n');
        fprintf(HDR.FILE.stderr,'%i ',find(~HDR.PhysDimCode));  
        fprintf(HDR.FILE.stderr,'\n');
      end; 	                        	
      
      if ~isfield(HDR,'PhysMin')
        if HDR.NS>0,
          fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysMin not defined\n');
        end
        HDR.PhysMin=repmat(nan,HDR.NS,1);
      else
        HDR.PhysMin=HDR.PhysMin(1:HDR.NS);
      end;
      if ~isfield(HDR,'TOffset')
        HDR.TOffset=repmat(nan,HDR.NS,1);
      else
        HDR.TOffset=[HDR.TOffset(:);repmat(NaN,HDR.NS,1)];
        HDR.TOffset=HDR.TOffset(1:HDR.NS);
      end;
      if ~isfield(HDR,'PhysMax')
        if HDR.NS>0,
          fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysMax not defined\n');
        end;
        HDR.PhysMax=repmat(nan,HDR.NS,1);
      else
        HDR.PhysMax=HDR.PhysMax(1:HDR.NS);
      end;
      if ~isfield(HDR,'DigMin')
        if HDR.NS>0,
          fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.DigMin not defined\n');
        end
        HDR.DigMin=repmat(nan,HDR.NS,1);
      else
        HDR.DigMin=HDR.DigMin(1:HDR.NS);
      end;
      if ~isfield(HDR,'DigMax')
        if HDR.NS>0,
          fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.DigMax not defined\n');
        end;
        HDR.DigMax=repmat(nan,HDR.NS,1);
      else
        HDR.DigMax=HDR.DigMax(1:HDR.NS);
      end;
      HDR.Cal = (HDR.PhysMax(:)-HDR.PhysMin(:))./(HDR.DigMax(:)-HDR.DigMin(:));
      HDR.Off = HDR.PhysMin(:) - HDR.Cal(:) .* HDR.DigMin(:);

      flag = isfield(HDR,'ELEC');	
      if flag,
        flag = isfield(HDR.ELEC,'XYZ');
      end;		
      if ~flag,				
        HDR.ELEC.XYZ = repmat(NaN,HDR.NS,3); 
        HDR.ELEC.REF = repmat(NaN,1,3); 
        HDR.ELEC.GND = repmat(NaN,1,3); 
      elseif ~isnumeric(HDR.ELEC.XYZ)
        fprintf('Warning SOPEN (GDF)-W: HDR.ELEC.LOC must be numeric.\n');
      elseif any(size(HDR.ELEC.XYZ)==[HDR.NS,3])
        HDR.ELEC.REF = repmat(NaN,1,3); 
        HDR.ELEC.GND = repmat(NaN,1,3); 
      elseif any(size(HDR.ELEC.XYZ)==[HDR.NS+1,3])
        HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
        HDR.ELEC.GND = repmat(NaN,1,3); 
      elseif any(size(HDR.ELEC.XYZ)==[HDR.NS+2,3])
        HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
        HDR.ELEC.GND = HDR.ELEC.XYZ(HDR.NS+2,:); 
      else
        fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.ELEC.LOC not correctly defined\n');
        tmp = [HDR.ELEC.XYZ,repmat(NaN,size(HDR.ELEC.XYZ,1),3)];
        tmp = [tmp;repmat(NaN,HDR.NS+2,size(tmp,2))];
        HDR.ELEC.XYZ = tmp(1:HDR.NS,1:3);
        HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
        HDR.ELEC.GND = HDR.ELEC.XYZ(HDR.NS+2,:); 
      end;
      if ~isfield(HDR,'Impedance')
        HDR.Impedance = repmat(NaN,HDR.NS,1); 
      elseif ~isnumeric(HDR.Impedance)
        fprintf('Warning SOPEN (GDF)-W: HDR.Impedance must be numeric.\n');
      elseif (length(HDR.Impedance)~=HDR.NS)
        sz = size(HDR.Impedance(:));
        tmp = [HDR.Impedance(:),repmat(NaN,sz(1),1);repmat(NaN,HDR.NS,sz(2)+1)];	
        HDR.Impedance = tmp(1:HDR.NS,1); 
      end
      
      ix = find((HDR.DigMax(:)==HDR.DigMin(:)) & (HDR.PhysMax(:)==HDR.PhysMin(:)));
      HDR.PhysMax(ix) = 1; 
      HDR.PhysMin(ix) = 0; 
      HDR.DigMax(ix) = 1; 
      HDR.DigMin(ix) = 0; 
      
      if 0, isfield(HDR.AS,'SampleRate')
        HDR.AS.SPR = HDR.AS.SampleRate(1:HDR.NS)/HDR.SampleRate * HDR.SPR;
        if any(HDR.AS.SPR~=ceil(HDR.AS.SPR)),
          fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.AS.SPR is not integer\n');
        end;         
      elseif ~isfield(HDR.AS,'SPR')
        if HDR.NS>0,
          fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.AS.SPR not defined\n');
        end;
        HDR.AS.SPR = repmat(nan,HDR.NS,1);
        HDR.ErrMsg = sprintf('Error SOPEN (GDF/EDF/BDF)-W: HDR.AS.SPR not defined\n');
        HDR.ErrNum = HDR.ErrNum + 128;
        %fclose(HDR.FILE.FID); return;
      else
        HDR.AS.SPR=reshape(HDR.AS.SPR(1:HDR.NS),HDR.NS,1);
      end;
      
    end;	% header 2

    %%%%%% generate Header 3,  Tag-Length-Value
    TagLenValue = {};
    TagLen = 0;
    if isfield(HDR,'Manufacturer')
      tag = 3;
      if ~isfield(HDR.Manufacturer,'Name') 	HDR.Manufacturer.Name=''; end; 
      if ~isfield(HDR.Manufacturer,'Model') 	HDR.Manufacturer.Model=''; end; 
      if ~isfield(HDR.Manufacturer,'Version') HDR.Manufacturer.Version=''; end;
      if ~isfield(HDR.Manufacturer,'SerialNumber') HDR.Manufacturer.SerialNumber=''; end;  
      TagLenValue{tag} = char([HDR.Manufacturer.Name,0,HDR.Manufacturer.Model,0,HDR.Manufacturer.Version,0,HDR.Manufacturer.SerialNumber]);
      TagLen(tag) = length(TagLenValue{tag}); 
    end;

    if 0, isfield(HDR,'ELEC') && isfield(HDR.ELEC,'Orientation') && all(size(HDR.ELEC.Orientation)==[HDR.NS,3]) 
      %% OBSOLETE 
      tag = 4; 
      TagLenValue{tag} = HDR.ELEC.Orientation;
      TagLen(tag) = 16*HDR.NS; 
    end;

    %%%%%% generate Header 1, first 256 bytes 
    HDR.HeadLen=(HDR.NS+1)*256;
    if any(TagLen>0) && (HDR.VERSION>2)
      HDR.HeadLen = HDR.HeadLen + ceil((sum(TagLen)+4*sum(TagLen>0)+4)/256)*256; 	%% terminating 0-Tag
    end; 
    
    %H1(1:8)=HDR.VERSION; %sprintf('%08i',HDR.VERSION);     % 8 Byte  Versionsnummer 
    if isempty(HDR.Patient.Birthday), bd = 'X';
      HDR.Patient.Birthday = zeros(1,6);
    elseif (~HDR.Patient.Birthday), bd = 'X';
      HDR.Patient.Birthday = zeros(1,6);
    else bd=datestr(datenum(HDR.Patient.Birthday),'dd-mmm-yyyy');
    end;
    if HDR.VERSION == -1,
      H1 = [255,'BIOSEMI',repmat(32,1,248)];
      HDR.PID = [HDR.Patient.Id,' ',GENDER{HDR.Patient.Sex+1}(1),' ',bd,' ',HDR.Patient.Name];
      HDR.RID = ['Startdate ',datestr(HDR.T0,'dd-mmm-yyyy')];
    elseif HDR.VERSION == 0,
      H1 = ['0       ',repmat(32,1,248)]; 
      HDR.PID = [HDR.Patient.Id,' ',GENDER{HDR.Patient.Sex+1}(1),' ',bd,' ',HDR.Patient.Name];
      HDR.RID = ['Startdate ',datestr(datenum(HDR.T0),'dd-mmm-yyyy')];
    elseif HDR.VERSION > 0,
      tmp = sprintf('%5.2f',HDR.VERSION);
      H1 = ['GDF',tmp(1:5),repmat(32,1,248)];
      HDR.PID = [HDR.Patient.Id,' ',HDR.Patient.Name];
      % HDR.RID = 'Hospital_administration_Code Technician_ID [Equipment_ID]'
    else
      fprintf(HDR.FILE.stderr,'Error SOPEN (GDF) WRITE: invalid version number %f\n',HDR.VERSION); 
    end;
    H1( 8+(1:length(HDR.PID))) = HDR.PID;
    H1(88+(1:length(HDR.RID))) = HDR.RID;
    %H1(185:192)=sprintf('%-8i',HDR.HeadLen);
    HDR.AS.SPR = HDR.AS.SPR(1:HDR.NS);
    HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
    HDR.AS.bi  = [0;cumsum(HDR.AS.SPR(:))];
    HDR.AS.BPR = ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)');
    while HDR.NS && any(HDR.AS.BPR(:)  ~= HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP+1)');
      fprintf(2,'\nWarning SOPEN (GDF/EDF/BDF): invalid block configuration in file %s.\n',HDR.FileName);
      HDR.SPR,
      DIV = 2;
      HDR.SPR    = HDR.SPR*DIV;
      HDR.AS.SPR = HDR.AS.SPR*DIV;
      HDR.Dur    = HDR.Dur*DIV; 
      HDR.NRec   = HDR.NRec/DIV; 
      HDR.AS.BPR = ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)');
    end;
    HDR.AS.SAMECHANTYP = all(HDR.AS.BPR == (HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)')) && ~any(diff(HDR.GDFTYP));
    HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
    HDR.AS.bi  = [0;cumsum(HDR.AS.SPR(:))];
    HDR.AS.bpb   = sum(ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)'));	% Bytes per Block
    HDR.FILE.POS  = 0;

    if (HDR.VERSION>=1.9),	% do some header checks
      if datenum([1850,1,1,0,0,0])>datenum(HDR.Patient.Birthday),
        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.Patient.Birthday is not correctly defined.\n');
      end;
    elseif (HDR.VERSION == 0)
      if sum(HDR.AS.bpb)>61440;
        fprintf(HDR.FILE.stderr,'\nWarning SOPEN (EDF): One block exceeds 61440 bytes.\n')
      end;
    end;

    %%%%% Open File 
    if ((HDR.NRec<0) && any(HDR.FILE.PERMISSION=='z')),
      %% due to a limitation zlib
      fprintf(HDR.FILE.stderr,'ERROR SOPEN (GDF/EDF/BDF) "wz": Update of HDR.NRec and writing Eventtable are not possible.\n',HDR.FileName);
      fprintf(HDR.FILE.stderr,'\t Solution(s): (1) define exactly HDR.NRec before calling SOPEN(HDR,"wz"); or (2) write to uncompressed file instead.\n');
      return;
    end;

    if 1,
      [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,[HDR.FILE.PERMISSION,'b'],'ieee-le');          
    elseif ~any(PERMISSION=='+') 
      [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,'w+b','ieee-le');          
    else  % (arg2=='w+')  % may be called only by SDFCLOSE
      if HDR.FILE.OPEN==2 
        [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,'r+b','ieee-le');          
      else
        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W+: Cannot open %s for write access\n',HDR.FileName);
        return;
      end;
    end;

    if HDR.FILE.FID<0 
      %fprintf(HDR.FILE.stderr,'Error EDFOPEN: %s\n',MESSAGE);  
      H1=MESSAGE;H2=[];
      HDR.ErrNum = HDR.ErrNum + 32;
      fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: Could not open %s \n',HDR.FileName);
      return;
    end;
    HDR.FILE.OPEN = 2;

    %if strcmp(HDR.VERSION(1:3),'GDF'),
    if (HDR.VERSION > 0),  % GDF
      if (HDR.VERSION >= 1.90)
        H1(85) = mod(HDR.Patient.Medication,3)*64 + mod(HDR.Patient.DrugAbuse,3)*16 + mod(HDR.Patient.AlcoholAbuse,3)*4  + mod(HDR.Patient.Smoking,3);
        H1(86) = HDR.Patient.Weight; 
        H1(87) = HDR.Patient.Height; 
        H1(88) = bitand(HDR.Patient.Sex,3) + bitand(HDR.Patient.Handedness,3)*4 + bitand(HDR.Patient.Impairment.Visual,3)*16;
        if all(H1(153:156)==32)
          c = fwrite(HDR.FILE.FID,abs(H1(1:152)),'uint8');
          c = fwrite(HDR.FILE.FID,HDR.REC.LOC.RFC1876,'uint32');
        else
          c = fwrite(HDR.FILE.FID,abs(H1(1:156)),'uint8');
          c = fwrite(HDR.FILE.FID,HDR.REC.LOC.RFC1876(2:4),'uint32');
        end;
        tmp = [datenum(HDR.T0), datenum(HDR.Patient.Birthday)];
        tmp = floor([rem(tmp,1)*2^32;tmp]);
        c   = fwrite(HDR.FILE.FID,tmp,'uint32');
        c   = fwrite(HDR.FILE.FID,[HDR.HeadLen/256,0,0,0],'uint16');
        c   = fwrite(HDR.FILE.FID,'b4om2.39','uint8'); % EP_ID=ones(8,1)*32;
        if (HDR.VERSION < 2.1)
          tmp = [HDR.REC.IPaddr, zeros(1,2)];
        else
          tmp = zeros(1,6);
        end;
        c=fwrite(HDR.FILE.FID,tmp(6:-1:1),'uint8'); % IP address, v2.1+: reserved
        c=fwrite(HDR.FILE.FID,HDR.Patient.Headsize(1:3),'uint16'); % circumference, nasion-inion, left-right mastoid in [mm]
        c=fwrite(HDR.FILE.FID,HDR.ELEC.REF(1:3),'float32'); % [X,Y,Z] position of reference electrode
        c=fwrite(HDR.FILE.FID,HDR.ELEC.GND(1:3),'float32'); % [X,Y,Z] position of ground electrode
      else
        Equipment  = [HDR.REC.Equipment, '        '];
        Hospital   = [HDR.REC.Hospital,  '        '];
        Technician = [HDR.REC.Technician,'        '];

        H1(169:184) = sprintf('%04i%02i%02i%02i%02i%02i%02i',floor(HDR.T0),floor(100*rem(HDR.T0(6),1)));
        c=fwrite(HDR.FILE.FID,H1(1:184),'uint8');
        c=fwrite(HDR.FILE.FID,[HDR.HeadLen,0],'int32');
        c=fwrite(HDR.FILE.FID,Equipment(1:8),'uint8'); % EP_ID=ones(8,1)*32;
        c=fwrite(HDR.FILE.FID,Hospital(1:8),'uint8'); % Lab_ID=ones(8,1)*32;
        c=fwrite(HDR.FILE.FID,Technician(1:8),'uint8'); % T_ID=ones(8,1)*32;
        c=fwrite(HDR.FILE.FID,ones(20,1)*32,'uint8'); % 
      end;

      %c=fwrite(HDR.FILE.FID,HDR.NRec,'int64');
      c=fwrite(HDR.FILE.FID,[HDR.NRec,0],'int32');
      if (HDR.VERSION > 2.20)
        fwrite(HDR.FILE.FID,HDR.Dur,'float64');
      else
        [n,d]=rat(HDR.Dur);
        fwrite(HDR.FILE.FID,[n d], 'uint32');
      end; 	
      c=fwrite(HDR.FILE.FID,[HDR.NS,0],'uint16');
    else
      H1(168+(1:16))=sprintf('%02i.%02i.%02i%02i.%02i.%02i',floor(rem(HDR.T0([3 2 1 4 5 6]),100)));
      H1(185:192)=sprintf('%-8i',HDR.HeadLen);
      H1(237:244)=sprintf('%-8i',HDR.NRec);
      if (HDR.Dur==ceil(HDR.Dur))
        tmp = sprintf('%-8i',HDR.Dur);
      else	
        tmp = sprintf('%-8f',HDR.Dur);
      end; 	
      H1(245:252)=tmp(1:8);
      if length(tmp)~=8, 
        tmp = str2double(tmp);
        tmp = (HDR.Dur-tmp)/HDR.Dur; 
        if abs(tmp)>1e-10,
          fprintf(HDR.FILE.stderr,'Warning SOPEN(EDF write): Duration field truncated, error %e (%s instead of %-8f),\n',tmp,H1(245:252),HDR.Dur);
        end; 	
      end;
      H1(253:256)=sprintf('%-4i',HDR.NS);
      H1(abs(H1)==0)=char(32); 

      c=fwrite(HDR.FILE.FID,abs(H1),'uint8');
    end;
    %%%%%% generate Header 2,  NS*256 bytes 
    if HDR.NS>0, 
      %if ~strcmp(HDR.VERSION(1:3),'GDF');
      if ~(HDR.VERSION > 0);
        sPhysMax=char(32+zeros(HDR.NS,8));
        sPhysMin=char(32+zeros(HDR.NS,8));
        for k=1:HDR.NS,
          tmp=sprintf('%-8g',HDR.PhysMin(k));
          lt=length(tmp);
          if lt<9
            sPhysMin(k,1:lt)=tmp;
          else
            if any(upper(tmp)=='E') || any(find(tmp=='.')>8),
              fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: PhysMin(%i) does not fit into header\n', k);
            else
              sPhysMin(k,:)=tmp(1:8);
            end;
          end;
          tmp=sprintf('%-8g',HDR.PhysMax(k));
          lt=length(tmp);
          if lt<9
            sPhysMax(k,1:lt)=tmp;
          else
            if any(upper(tmp)=='E') || any(find(tmp=='.')>8),
              fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: PhysMin(%i) does not fit into header\n', k);
            else
              sPhysMax(k,:)=tmp(1:8);
            end;
          end;
        end;
        c1 = str2double(cellstr(sPhysMax));
        c2 = str2double(cellstr(sPhysMin));
        e = ((HDR.PhysMax(:)-HDR.PhysMin(:))-(c1-c2))./(HDR.PhysMax(:)-HDR.PhysMin(:));
        if any(abs(e)>1e-8)
          fprintf(HDR.FILE.stderr,'Warning SOPEN (EDF-Write): relative scaling error is %e (due to roundoff in PhysMax/Min)\n',max(abs(e)))
        end
        
        idx1=cumsum([0 H2idx]);
        idx2=HDR.NS*idx1;
        h2=char(32*ones(HDR.NS,256));
        size(h2);
        h2(:,idx1(1)+1:idx1(2))=Label;
        h2(:,idx1(2)+1:idx1(3))=Transducer;
        h2(:,idx1(3)+1:idx1(4))=PhysDim;
        %h2(:,idx1(4)+1:idx1(5))=sPhysMin;
        %h2(:,idx1(5)+1:idx1(6))=sPhysMax;
        h2(:,idx1(4)+1:idx1(5))=sPhysMin;
        h2(:,idx1(5)+1:idx1(6))=sPhysMax;
        h2(:,idx1(6)+1:idx1(7))=reshape(sprintf('%-8i',HDR.DigMin)',8,HDR.NS)';
        h2(:,idx1(7)+1:idx1(8))=reshape(sprintf('%-8i',HDR.DigMax)',8,HDR.NS)';
        h2(:,idx1(8)+1:idx1(9))=HDR.PreFilt;
        h2(:,idx1(9)+1:idx1(10))=reshape(sprintf('%-8i',HDR.AS.SPR)',8,HDR.NS)';
        h2(abs(h2)==0)=char(32);
        for k=1:length(H2idx);
          c=fwrite(HDR.FILE.FID,abs(h2(:,idx1(k)+1:idx1(k+1)))','uint8');
        end;
      else
        fwrite(HDR.FILE.FID, abs(Label)','uint8');
        fwrite(HDR.FILE.FID, abs(Transducer)','uint8');
        if (HDR.VERSION < 1.9)
          fwrite(HDR.FILE.FID, abs(PhysDim)','uint8');
          fwrite(HDR.FILE.FID, HDR.PhysMin,'float64');
          fwrite(HDR.FILE.FID, HDR.PhysMax,'float64');

          if 0, exist('OCTAVE_VERSION','builtin'),  % Octave does not support INT64 yet.
            fwrite(HDR.FILE.FID, [HDR.DigMin(:),-(HDR.DigMin(:)<0)]','int32');
            fwrite(HDR.FILE.FID, [HDR.DigMax(:),-(HDR.DigMax(:)<0)]','int32');
          else
            fwrite(HDR.FILE.FID, HDR.DigMin, 'int64');
            fwrite(HDR.FILE.FID, HDR.DigMax, 'int64');
          end;
          fwrite(HDR.FILE.FID, abs(HDR.PreFilt)','uint8');
          fwrite(HDR.FILE.FID, HDR.AS.SPR,'uint32');
          fwrite(HDR.FILE.FID, HDR.GDFTYP,'uint32');
          fwrite(HDR.FILE.FID,32*ones(32,HDR.NS),'uint8');
        else
          fwrite(HDR.FILE.FID, abs(PhysDim(1:HDR.NS,1:6))','uint8');
          fwrite(HDR.FILE.FID, HDR.PhysDimCode(1:HDR.NS),'uint16');
          fwrite(HDR.FILE.FID, HDR.PhysMin(1:HDR.NS),'float64');
          fwrite(HDR.FILE.FID, HDR.PhysMax(1:HDR.NS),'float64');

          fwrite(HDR.FILE.FID, HDR.DigMin(1:HDR.NS), 'float64');
          fwrite(HDR.FILE.FID, HDR.DigMax(1:HDR.NS), 'float64');
          
          if (HDR.VERSION < 2.22)
            fwrite(HDR.FILE.FID, abs(HDR.PreFilt(1:HDR.NS,1:68))','uint8');
          else
            fwrite(HDR.FILE.FID, abs(HDR.PreFilt(1:HDR.NS,1:64))','uint8');
            fwrite(HDR.FILE.FID, HDR.TOffset(1:HDR.NS), 'float32');
          end;
          fwrite(HDR.FILE.FID, HDR.Filter.LowPass(1:HDR.NS),'float32');
          fwrite(HDR.FILE.FID, HDR.Filter.HighPass(1:HDR.NS),'float32');
          fwrite(HDR.FILE.FID, HDR.Filter.Notch(1:HDR.NS),'float32');
          fwrite(HDR.FILE.FID, HDR.AS.SPR(1:HDR.NS),'uint32');
          fwrite(HDR.FILE.FID, HDR.GDFTYP(1:HDR.NS),'uint32');
          fwrite(HDR.FILE.FID, HDR.ELEC.XYZ(1:HDR.NS,:)','float32');
          if (HDR.VERSION < 2.19)
            fwrite(HDR.FILE.FID, max(0,min(255,round(log2(HDR.Impedance(1:HDR.NS))*8)')),'uint8');
            fwrite(HDR.FILE.FID,32*ones(19,HDR.NS),'uint8');
          else 
            tmp = repmat(NaN,5,HDR.NS); 
            ch  = find(bitand(HDR.PhysDimCode, hex2dec('ffe0'))==4256); % channel with voltage data  
            tmp(1,ch) = HDR.Impedance(ch);
            ch  = find(bitand(HDR.PhysDimCode, hex2dec('ffe0'))==4288); % channel with impedance data  
            if isfield(HDR,'fZ')
              tmp(1,ch) = HDR.fZ(ch);                      % probe frequency
            end;
            fwrite(HDR.FILE.FID, tmp, 'float32');
          end         
        end;
      end;
    end;

    if (HDR.VERSION>2)
      %%%%%% GDF2: generate Header 3,  Tag-Length-Value
      for tag=find(TagLen>0)
        fwrite(HDR.FILE.FID, tag+TagLen(tag)*256, 'uint32');
        switch tag 
         case 3 
          fwrite(HDR.FILE.FID, TagLenValue{tag}, 'uint8');
         case 4 	%% OBSOLETE 
                  %  c=fwrite(HDR.FILE.FID, HDR.ELEC.Orientation, 'float32');
                  %  c=c+fwrite(HDR.FILE.FID, HDR.ELEC.Area, 'float32');
          fwrite(HDR.FILE.FID, zeros(4*HDR.NS-c), 'float32');
        end;
      end; 
      if any(TagLen>0) 
        fwrite(HDR.FILE.FID, 0, 'uint32');	%% terminating 0-tag
      end; 
    end; 
    
    tmp = ftell(HDR.FILE.FID);
    if tmp ~= HDR.HeadLen, 
      fwrite(HDR.FILE.FID, zeros(1,HDR.HeadLen-tmp), 'uint8');
    end; 	
    tmp = ftell(HDR.FILE.FID);
    if tmp ~= HDR.HeadLen, 
      fprintf(1,'Warning SOPEN (GDF/EDF/BDF)-WRITE: incorrect header length %i vs. %i bytes\n',tmp, HDR.HeadLen );
      %else   fprintf(1,'SOPEN (GDF/EDF/BDF) in write mode: header info stored correctly\n');
    end;        

  else % if arg2 is not 'r' or 'w'
    fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF): Incorrect 2nd argument. \n');
  end;        
  
  if HDR.ErrNum>0
    fprintf(HDR.FILE.stderr,'ERROR %i SOPEN (GDF/EDF/BDF)\n',HDR.ErrNum);
  end;
  

elseif strcmp(HDR.TYPE,'EVENT') && any(lower(HDR.FILE.PERMISSION)=='w'),
  %%% Save event file in GDF-format
  HDR.TYPE = 'GDF';
  HDR.NS   = 0; 
  HDR.NRec = 0; 
  if any(isnan(HDR.T0))
    HDR.T0 = clock;
  end;
  HDR = sopen(HDR,'w');
  HDR = sclose(HDR);
  HDR.TYPE = 'EVENT';

        
else
        %fprintf(HDR.FILE.stderr,'SOPEN does not support your data format yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        HDR.FILE.FID = -1;	% this indicates that file could not be opened. 
        return;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	General Postprecessing for all formats of Header information 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isfield(HDR,'Patient') && isfield(HDR.Patient,'Weight') && isfield(HDR.Patient,'Height')
	%% Body Mass Index 
       	HDR.Patient.BMI = HDR.Patient.Weight * HDR.Patient.Height^-2 * 1e4;

       	%% Body Surface Area
	% DuBois D, DuBois EF. A formula to estimate the approximate surface area if height and weight be known. Arch Intern Medicine. 1916; 17:863-71.
	% Wang Y, Moss J, Thisted R. Predictors of body surface area. J Clin Anesth. 1992; 4(1):4-10.
       	HDR.Patient.BSA = 0.007184 * HDR.Patient.Weight^0.425 * HDR.Patient.Height^0.725;
end; 

% check consistency
if (HDR.NS>0) && HDR.FLAG.OVERFLOWDETECTION && ~isfield(HDR,'THRESHOLD') && ~strcmp(HDR.TYPE,'EVENT'),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Automated OVERFLOWDETECTION not supported - check yourself for saturation artifacts.\n');
end;

% identify type of signal, complete header information
if HDR.NS>0,
        HDR = physicalunits(HDR); % complete information on PhysDim, and PhysDimCode
        HDR = leadidcodexyz(HDR); % complete information on LeadIdCode and Electrode positions of EEG channels.
        if ~isfield(HDR,'Label')
                HDR.Label = cellstr([repmat('#',HDR.NS,1),int2str([1:HDR.NS]')]);
        elseif isempty(HDR.Label)	
                HDR.Label = cellstr([repmat('#',HDR.NS,1),int2str([1:HDR.NS]')]);
        elseif ischar(HDR.Label)
                HDR.Label = cellstr(HDR.Label); 
        end;
        if ischar(HDR.PhysDim)
                HDR.PhysDim = cellstr(HDR.PhysDim); 
        end; 
        HDR.CHANTYP = repmat(' ',1,HDR.NS);
        tmp = HDR.NS-length(HDR.Label);
        %HDR.Label = [HDR.Label(1:HDR.NS,:);repmat(' ',max(0,tmp),size(HDR.Label,2))];
        Label = char(HDR.Label);
        tmp = reshape(lower([[Label(1:min(HDR.NS,size(Label,1)),:);repmat(' ',max(0,tmp),size(Label,2))],repmat(' ',HDR.NS,1)])',1,HDR.NS*(size(Label,2)+1));
        
        HDR.CHANTYP(ceil([strfind(tmp,'eeg'),strfind(tmp,'meg')]/(size(Label,2)+1))) = 'E'; 
        HDR.CHANTYP(ceil([strfind(tmp,'emg')]/(size(Label,2)+1))) = 'M'; 
        HDR.CHANTYP(ceil([strfind(tmp,'eog')]/(size(Label,2)+1))) = 'O'; 
        HDR.CHANTYP(ceil([strfind(tmp,'ecg'),strfind(tmp,'ekg')]/(size(Label,2)+1))) = 'C'; 
        HDR.CHANTYP(ceil([strfind(tmp,'air'),strfind(tmp,'resp')]/(size(Label,2)+1))) = 'R'; 
        HDR.CHANTYP(ceil([strfind(tmp,'trig')]/(size(Label,2)+1))) = 'T'; 
end;

% add trigger information for triggered data
if HDR.FLAG.TRIGGERED && isempty(HDR.EVENT.POS)
	HDR.EVENT.POS = [0:HDR.NRec-1]'*HDR.SPR+1;
	HDR.EVENT.TYP = repmat(hex2dec('0300'),HDR.NRec,1);
	HDR.EVENT.CHN = repmat(0,HDR.NRec,1);
	HDR.EVENT.DUR = repmat(0,HDR.NRec,1);
end;

% apply channel selections to EVENT table
if any(CHAN) && ~isempty(HDR.EVENT.POS) && isfield(HDR.EVENT,'CHN'),	% only if channels are selected. 
	sel = (HDR.EVENT.CHN(:)==0);	% memory allocation, select all general events
	for k = find(~sel'),		% select channel specific elements
		sel(k) = any(HDR.EVENT.CHN(k)==CHAN);
	end;
	HDR.EVENT.POS = HDR.EVENT.POS(sel);
	HDR.EVENT.TYP = HDR.EVENT.TYP(sel);
	HDR.EVENT.DUR = HDR.EVENT.DUR(sel);	% if EVENT.CHN available, also EVENT.DUR is defined. 
	HDR.EVENT.CHN = HDR.EVENT.CHN(sel);
	% assigning new channel number 
	a = zeros(1,HDR.NS);
	for k = 1:length(CHAN),		% select channel specific elements
		a(CHAN(k)) = k;		% assigning to new channel number. 
	end;
	ix = HDR.EVENT.CHN>0;
	HDR.EVENT.CHN(ix) = a(HDR.EVENT.CHN(ix));	% assigning new channel number
end;	

% complete event information - needed by SVIEWER
if ~isfield(HDR.EVENT,'CHN') && ~isfield(HDR.EVENT,'DUR'),  
	HDR.EVENT.CHN = zeros(size(HDR.EVENT.POS)); 
	HDR.EVENT.DUR = zeros(size(HDR.EVENT.POS)); 

	% convert EVENT.Version 1 to 3, currently used by GDF, BDF and alpha
	flag_remove = zeros(size(HDR.EVENT.TYP));
	types  = unique(HDR.EVENT.TYP);
	for k1 = find(bitand(types(:)',hex2dec('8000')));
		TYP0 = bitand(types(k1),hex2dec('7fff'));
		TYP1 = types(k1);
		ix0  = (HDR.EVENT.TYP==TYP0);
		ix1  = (HDR.EVENT.TYP==TYP1);

	        if sum(ix0)==sum(ix1), 
	                HDR.EVENT.DUR(ix0) = HDR.EVENT.POS(ix1) - HDR.EVENT.POS(ix0);
	                flag_remove = flag_remove | (HDR.EVENT.TYP==TYP1);
                else 
	                fprintf(2,'Warning SOPEN: number of event onset (TYP=%s) and event offset (TYP=%s) differ (%i,%i)\n',dec2hex(double(TYP0)),dec2hex(double(TYP1)),sum(ix0),sum(ix1));
                        %% double(.) operator needed because Matlab6.5 can not fix fix(uint16(..))
	        end;
	end
	if any(HDR.EVENT.DUR<0)
	        fprintf(2,'Warning SOPEN: EVENT ONSET later than EVENT OFFSET\n',dec2hex(TYP0),dec2hex(TYP1));
	        %HDR.EVENT.DUR(:) = 0
	end;
	HDR.EVENT.TYP = HDR.EVENT.TYP(~flag_remove);
	HDR.EVENT.POS = HDR.EVENT.POS(~flag_remove);
	HDR.EVENT.CHN = HDR.EVENT.CHN(~flag_remove);
	HDR.EVENT.DUR = HDR.EVENT.DUR(~flag_remove);
end;	
[tmp,ix] = sort(HDR.EVENT.POS);
HDR.EVENT.TYP=HDR.EVENT.TYP(ix);
HDR.EVENT.POS=HDR.EVENT.POS(ix);
HDR.EVENT.DUR=HDR.EVENT.DUR(ix);
HDR.EVENT.CHN=HDR.EVENT.CHN(ix);

% Calibration matrix
if any(HDR.FILE.PERMISSION=='r') && (HDR.NS>0);
        if isempty(ReRefMx)     % CHAN==0,
                ReRefMx = eye(max(1,HDR.NS));
        end;
        sz = size(ReRefMx);
        if (HDR.NS > 0) && (sz(1) > HDR.NS),
                fprintf(HDR.FILE.stderr,'ERROR SOPEN: to many channels (%i) required, only %i channels available.\n',size(ReRefMx,1),HDR.NS);
                HDR = sclose(HDR);
                return;
        end;
        if ~isfield(HDR,'Calib')
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1); 
        end;
	if ~HDR.FLAG.FORCEALLCHANNEL,
	        HDR.Calib = HDR.Calib*sparse([ReRefMx; zeros(HDR.NS-sz(1),sz(2))]);
	else         
		HDR.ReRefMx = ReRefMx;
	end; 	 
        
        HDR.InChanSelect = find(any(HDR.Calib(2:HDR.NS+1,:),2));
        HDR.Calib = sparse(HDR.Calib([1;1+HDR.InChanSelect(:)],:));
        if strcmp(HDR.TYPE,'native')
                HDR.data = HDR.data(:,HDR.InChanSelect);
        end;
end;

