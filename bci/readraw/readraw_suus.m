function [x,y,fs,trialinfo,summary,opts,info] = readraw_suus(z, files, fileindex, varargin)

opts = {
    'mergedims' 0
}';
error(resolveopts(varargin, opts))

x = [];
y = [];
fs = [];
summary = '';
info=struct('filename',filename);

filename = files{fileindex};
if ~strncmp(lower(fliplr(filename)), 'tam.', 4), return, end  % assuming input is as mat-files

v = load(filename);

% change the order of inputs to meet jez's format (but perserve time order)
% N.B. input is : sample channel epoc rep letter
if ( opts.mergedims ) 
   x = double(permute(v.FEAT,[3 4 5 2 1]));  
   y = double(v.isT);
   %      x will be   : (epoc*rep*letter) channel sample
else 
   x = double(permute(v.FEAT,[5 4 3 2 1]));    
   %      x is now  : letter rep epoc channel sample
   y = double(permute(v.isT,[3 2 1]));
end
fs= v.fs;

% Build the trial info structure
layoutstr=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_'];
info.layoutstr=layoutstr;
info.dims ={'trial','repetition','epoch','electrode','sample'};
if ( opts.mergedims ) % create a trials x dim x samples structure
	for i=1:size(x,1)*size(x,2)*size(x,3); 
   		[flash,rep,tr]=ind2sub([size(x,1) size(x,2) size(x,3)],i);
   		% store the flip grid to
   		flipgrid=false(size(layoutstr));
   		if ( v.StimSeq(i)<7 ) flipgrid(:,v.StimSeq(i))=true; 
   		else                  flipgrid(v.StimSeq(i)-6,:)=true;
   		end   
   		trialinfo(i)=struct('cue',y(i),...
                	   'true_letter',v.TargetChar(tr),'true_letter_index',find(v.TargetChar(tr)==layoutstr),...
                	   'flashno',flash,'repno',rep,'letno',tr,'flipgrid',flipgrid);
	end
	info.stimSeq=v.StimSeq;
	info.orig_dims=info.dims; % store orginal size/dim-names
	info.orig_size=size(x);
	y=y(:);
	x=reshape(x,[size(x,1)*size(x,2)*size(x,3),size(x,4),size(x,5)]);
	info.dims = {'trial','electrode','sample'};
else
	% fill out the trial info when a trial means a whole letter!
	for i=1:size(x,1);
   		% store the flip grid to
   		flipgrid=false([size(layoutstr) size(x,2) size(x,3)]);
		sIdx=(i-1)*(size(x,2)*size(x,3));
		for j=1:(size(x,2)*size(x,3));
   			if ( v.StimSeq(sIdx+j)<7 ) flipgrid(:,v.StimSeq(sIdx+j),j)=true; 
   			else                       flipgrid(v.StimSeq(sIdx+j)-6,:,j)=true;
			end
   		end
   		trialinfo(i)=struct('cue',y(i,:,:),...
                	   'true_letter',v.TargetChar(i),'true_letter_index',find(v.TargetChar(i)==layoutstr),...
                	   'letno',i,'flipgrid',flipgrid);		
	end
end

summary=sprintf('%s suus data',join('-by-',info.dims));
