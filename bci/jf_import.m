function [z]=jf_import(expt,subj,label,X,di,varargin)
% Import data into a jf-data-structure
%
% [z]=jf_import(expt,subj,label,X,di,varargin)
%
% Inputs:
%  expt-- [str] experiment name
%  subj-- [str] subject name
%  label--[str] label describing this data
%  X    --[n-d] data array
%  di   --[struct] dimInfo structure describing X (see mkDimInfo)
%          e.g. di=mkDimInfo(size(X),'ch',[],[],'time','ms',[],'epoch',[],[])
% Options:
%  fs -- [float] sampling rate
%  Y  -- [N x 1] epoch labels or markers, e.g. [-1 1 1] where [-1=left_hand, 1=right_hand]
%  Ydi-- [struct] dimInfo struct describing the labels
%          OR
%        [str] name of the dimension along which Y applies
%  info -- [struct] structure containing other info about the data
%  spType -- [str] type of sub-problem encoding to use
%  capFile -- [str] file to get electrode position info into
%  Cnames -- {str} channel names
%  offset_ms -- [2x1] time in ms for the first sample of the epoch
%  overrideChNames -- [bool] over-ride the channel names with the ones in the capFile (true)
%  session -- what session did this come from
%  summary -- [str] summary of this imported data
infostruct=struct('info',[],'Y',[],'fs',[],'offset_ms',[],'Ydi','epoch','spType','1vR','markerdict',[],'zeroLab',0);
opts=struct('capFile',[],'summary','','session','','Cnames',[],'overrideChNames',1,'verb',0);
[opts,infostruct]=parseOpts({opts,infostruct},varargin);

if ( nargin<5 || isempty(di) ) di={'ch' 'time' 'epoch' 'mV'}; end;
if ( iscell(di) )  di=mkDimInfo(size(X),di); end;

z=struct('X',X,'di',di,'expt',expt,'subj',subj,'label',label);
summary='Import data';
if ( ~isempty(opts.summary) ) summary=sprintf('%s (%s)',summary,opts.summary); end;
z=jf_addprep(z,mfilename,summary,opts,infostruct.info);
z.info = infostruct.info;
z.summary = jf_disp(z);
if ( ~isempty(infostruct.fs) ) % record sampling rate stuff
   if ( strmatch('time',{z.di.name}) ) z.di(n2d(z.di,'time')).info.fs=infostruct.fs; % in dimInfo
   else z.fs=infostruct.fs; % elsewhere
   end;
   % fix the sample times with this fs...
   if ( isequal(z.di(n2d(z,'time')).vals,(1:size(z.X,n2d(z,'time')))) )
      z.di(n2d(z,'time')).vals = z.di(n2d(z,'time')).vals.*1000/infostruct.fs;
   end
   if ( ~isempty(infostruct.offset_ms) )
      z.di(n2d(z,'time')).vals = z.di(n2d(z,'time')).vals+infostruct.offset_ms(1);
   end
end;
if ( ~isempty(infostruct.Y) ) % setup the labels
   Yl=infostruct.Y;  Ydi=infostruct.Ydi;
   if ( size(Yl,1)==1 && numel(Yl)==prod(size(Yl)) ) Yl=Yl'; end; % ensure col vector
   Yl=single(Yl);
   if ( ischar(Ydi) ) Ydi={Ydi}; end;
   if ( iscell(Ydi) ) % construct the Ydi
		[Y,Ydi]=addClassInfo(Yl,'Ydi',Ydi,'spType',infostruct.spType,'markerdict',infostruct.markerdict,'zeroLab',infostruct.zeroLab);
   end
   dim = n2d(z.di,{Ydi(1:end-1).name},0,0); dim(dim==0)=[];
   z.Y   = Y;
   z.Ydi = Ydi;
   [z.di(dim(1)).extra.marker]=num2csl(Yl(:),2); % record marker info   
end;

chD = strmatch('ch',{z.di.name});
if ( ~isempty(opts.Cnames) ) z.di(chD).vals=opts.Cnames; end;
if ( ~isempty(opts.capFile) ) 
   z.di(chD)=addPosInfo(z.di(chD),opts.capFile,opts.overrideChNames,opts.verb);
end
if ( ~isempty(opts.session) ) z.session=opts.session; else z.session=[]; end;
return;

%---------------------------------------------------------------------------
function testCase()
X = randn(10,100,10);
Y = sign(randn(size(X,3),1));
di= mkDimInfo(size(X),'ch',[],[],'time','ms',[],'epoch',[],[]);
z=jf_import('expt','subj','label',X,di,'Y',Y,'fs',256);


