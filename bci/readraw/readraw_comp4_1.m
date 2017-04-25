function [X,di,fs,summary,opts,info]=readraw_comp4_1(filename,varargin)
opts=struct('fs',[],'single',1,'trlen_ms',5000,'offset_ms',[],'gain',.1,'offset',0);
opts=parseOpts(opts,varargin);

fprintf('reading %s\n', filename);
s=load(filename);% load data-files
fprintf('done.\n');

% extract the meta-info
fs    =s.nfo.fs; samp2ms = 1000/fs;  ms2samp = fs/1000;
Cnames=s.nfo.clab;
pos2d =[s.nfo.xpos s.nfo.ypos]';
markerdict=s.nfo.classes;

% extract the data
if ( opts.single ) X=single(s.cnt)*.1;
else               X=double(s.cnt)*.1;
end
% slice the data according to the labels
events=s.mrk;     % [.pos =sample no, .y=label];
bgns=events.pos;  % start of each epoch in samples
ends=bgns+round(opts.trlen_ms*ms2samp); % end after this number ms
y   =events.y;

% offset if wanted
offset_samp=0;
if ( ~isempty(opts.offset_ms) )
   offset_samp = opts.offset_ms(1)*ms2samp;
   if ( numel(opts.offset_ms)<2 ) 
      opts.offset_ms=[-opts.offset_ms opts.offset_ms]; 
   end;
   bgns=min(max(bgns+ceil(opts.offset_ms(1)*ms2samp),1),nsamp);
   ends=min(max(ends+ceil(opts.offset_ms(2)*ms2samp),1),nsamp);
end
trlens=ends-bgns+1;
maxtrlen=max(trlens(:));

% Now use this to slice up X
if ( opts.single ) X = zeros(size(s.cnt,2),maxtrlen,numel(bgns),'single');
else               X = zeros(size(s.cnt,2),maxtrlen,numel(bgns));
end
for tri=1:numel(bgns);
   xtr = double(s.cnt(bgns(tri):ends(tri),:))*opts.gain+opts.offset;
   X(:,:,tri) = xtr';
end

% fill out the dimInfo
times=([1:maxtrlen]+offset_samp)*samp2ms;
di = mkDimInfo(size(X),'ch',[],Cnames,'time','ms',times,'epoch',[],[]);
[di(1).extra.pos2d] =num2csl(pos2d);
[di(3).extra.marker]=num2csl(y);
di(3).info.markerdict=markerdict;
di(end).units='uV';
[ans,fname,fext]=fileparts(filename);
info=struct('filename',filename);
summary=sprintf('%s%s: %depochs',fname,fext,size(X,3));

return;
%------------------------------------------------------------------------------------
function testCase()
expt       = 'eeg/comp4/1';
subjects   = {'a' 'b' 'c' 'd'  'e' 'f'};
sessions   = {{'calib'} {'calib'} {'calib'} {'calib'} {'calib'} {'calib'}};
dtype      = 'raw_comp4_1';
sessfileregexp='.*_(.*)_.*.mat';
fileregexp = '.*.mat';
%markerdict = {'non-tgt' 'tgt'};

% these are all per subject/session/condition
blocks = { {{[]}} {{[]}} {{[]}} {{[]}} {{[]}} };
labels = { {{''}} {{''}} {{''}} {{''}} {{''}} }; % N.B. for this one cond split happens *after* loading
markers= {1:4};

blockprefix='';
spType = '1v1';

si=1;sessi=1;ci=1;

subj=subjects{si}; 
session=sessions{si}{sessi}; 
block=blocks{si}{sessi}{ci}; 
marker=markers{1};
filelst = findFiles(expt,subj,session,block,'dtype',dtype,'fileregexp',fileregexp,'sessfileregexp',sessfileregexp);

[X,di,fs,summary,opts,info]=readraw_comp4_1(filelst.fname);