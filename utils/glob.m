function [ofn]=glob(fn)
% glob expand the input file name
% Warning: doesn't work with spaces in file names!
if ( ischar(fn) ) fn={fn}; end;
if ( isunix ) 
   ofn={};
   for i=1:numel(fn);
		 try;
			[ans nfn]=system(['echo ' fn{i}]); nfn(nfn==10)=[]; nfn(nfn==13)=[];	  
			nlIdx = [0 find(nfn==32) numel(nfn)+1];
			for j=1:numel(nlIdx)-1; %N.B. one pattern -> many files
           tfn{j} = nfn(nlIdx(j)+1:nlIdx(j+1)-1);
			end
			ofn={ofn{:} tfn{:}};
		 catch;
			ofn={ofn{:} fn{i}};
			fprintf('Warning: couldnt glob');
		 end
   end
   %if ( max(size(ofn))==1 ) ofn=ofn{:}; end;
else
   ofn=fn;% do nothing as globbing isn't defined?
end
