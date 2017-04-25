function [str]=printExtRes(st)
% Pretty print algorithm summary results
%
%  [str]=printExtRes(st)
%
if ( any(isnan(st.X(:))) ) st.X(isnan(st.X))=0; end;

str='';
% sort the data-sets into decreasing order of performance
[ans,si]=sort(msum(st.X,[1 3 4 5])./msum(st.X>0,[1 3 4 5]),'descend'); % sort algs
st.X=st.X(:,si,:,:,:); st.di(2).vals=st.di(2).vals(si);
[ans,si]=sort(msum(st.X,[1 2 3 5])./msum(st.X>0,[1 2 3 5]),'descend'); % sort ds
st.X=st.X(:,:,:,si,:); st.di(4).vals=st.di(4).vals(si);
[ans,si]=sort(msum(st.X,[1 2 4 5])./msum(st.X>0,[1 2 4 5]),'descend'); % sort labels
st.X=st.X(:,:,si,:,:); st.di(3).vals=st.di(3).vals(si);

% identify common prefix in dataset and label to remove
% alg
prefix=st.di(2).vals{1}; prefixlen=numel(prefix)+1;
for vi=1:numel(st.di(2).vals)
  lab=st.di(2).vals{vi};
  ci=1; for ci=1:min(numel(prefix),numel(lab)); if(~strcmp(lab(ci),prefix(ci))) break; end; end;
  if(strcmp(lab(ci),prefix(ci))) ci=ci+1; end;
  prefixlen=min(prefixlen,ci);
end
if(prefixlen-1>1) % at least 2char to remove
  for vi=1:numel(st.di(2).vals);st.di(2).vals{vi}=st.di(2).vals{vi}(prefixlen:end);end
end
% label
prefix=st.di(3).vals{1}; prefixlen=numel(prefix)+1;
for vi=1:numel(st.di(3).vals)
  lab=st.di(3).vals{vi};
  ci=1; for ci=1:min(numel(prefix),numel(lab)); if(~strcmp(lab(ci),prefix(ci))) break; end; end;
  if(strcmp(lab(ci),prefix(ci))) ci=ci+1; end;
  prefixlen=min(prefixlen,ci);
end
if(prefixlen>0)
  for vi=1:numel(st.di(3).vals);st.di(3).vals{vi}=st.di(3).vals{vi}(prefixlen:end);end
end
% dataset
prefix=st.di(4).vals{1}; prefixlen=numel(prefix)+1;
for vi=1:numel(st.di(4).vals)
  lab=st.di(4).vals{vi};
  ci=1;for ci=1:min(numel(prefix),numel(lab)); if(~strcmp(lab(ci),prefix(ci))) break; end; end;
  if(strcmp(lab(ci),prefix(ci))) ci=ci+1; end;
  prefixlen=min(prefixlen,ci);
end
if (prefixlen>0)
  for vi=1:numel(st.di(4).vals);st.di(4).vals{vi}=st.di(4).vals{vi}(prefixlen:end);end
end

% print the per-dataset + alg summary
if ( size(st.X,1)>1 || size(st.X,3)>1 ) 
  str=[str sprintf('\n')];
  str=[str sprintf('\n------------------\n %s\n','all')];
  str=[str sprintf('%35s :','Alg')];
  str=[str sprintf('%9s| ',' N ave    ')];
  ci=0;
  for spi=1:size(st.X,1);
     for labi=1:size(st.X,3)
        if (msum(st.X(spi,:,labi,:),1:ndims(st.X))==0) continue; end;        
        if (size(st.X,1)>1) str=[str sprintf('(%s)',st.di(1).vals{spi})]; end;
        for di=1:size(st.X,4);
           if(sum(st.X(spi,:,labi,di))==0) continue; end;
           str=[str sprintf('%2d:%2s+%2s \t',ci,st.di(3).vals{labi},st.di(4).vals{di})];
           ci=ci+1;
        end
     end
  end
  str=[str sprintf('\n')];  
  % results
  bestPerf = st.X(:,1,:,:);
  for ai=1:size(st.X,2);
    nWins = [msum(st.X(:,ai,:,:)>bestPerf+.01,[1 3 4]); msum(st.X(:,ai,:,:)+.01<bestPerf,[1 3 4])];
	 sig='  ' ;
	 if ( sum(nWins)>0 )
       if ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.01) )     sig='**'; % .01
       elseif ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.05) ) sig='* '; % .05
       end; 
    end;

    str=[str sprintf('%35s :',st.di(2).vals{ai})];
    % missing res aware
    str=[str sprintf('%2d %5.3f%2s| ',msum(st.X(:,ai,:,:)>0,[1 3 4]),msum(st.X(:,ai,:,:),[1 3 4])./max(1,msum(st.X(:,ai,:,:)>0,[1 3 4])),sig)];
    
    for spi=1:size(st.X,1);
       for labi=1:size(st.X,3)
          if (msum(st.X(spi,:,labi,:),1:ndims(st.X))==0) continue; end;        
          for di=1:size(st.X,4);
             if(sum(st.X(spi,:,labi,di))==0) continue; end;
             str=[str sprintf('%0.2f\t',st.X(spi,ai,labi,di))];
          end
       end
    end
    str=[str sprintf('\n')];
  end % alg
end % print overall summary

% print a results summary table
for spi=1:size(st.X,1);
  % header line
  str=[str sprintf('\n')];
  str=[str sprintf('\n------------------\n %s:',st.di(1).name)];
  if ( isnumeric(st.di(1).vals) ) str=[str sprintf('%g\n',st.di(1).vals(spi))]; 
  else                            str=[str sprintf('%s\n',st.di(1).vals{spi})]; 
  end
  for labi=1:size(st.X,3);
     if (msum(st.X(spi,:,labi,:),1:ndims(st.X))==0) continue; end;
     if (numel(labi)==1 ) 
        str=[str sprintf('\n') sprintf('\n--------------------\n')];
        str=[str sprintf(' %s : %s\n',st.di(3).name,st.di(3).vals{labi})]; 
     end;
     str=[str sprintf('%35s :','Alg')];
     str=[str sprintf('%9s| ',' N ave    ')];
     for vi=1:numel(st.di(4).vals); str=[str sprintf('%2d:%4s \t',vi,st.di(4).vals{vi})];end;
     str=[str sprintf('\n')];
     % results
     bestPerf = st.X(spi,1,labi,:);
     for ai=1:size(st.X,2);
        nWins = [sum(st.X(spi,ai,labi,:)>bestPerf+.01); sum(st.X(spi,ai,labi,:)+.01<bestPerf)];
		  sig='  ' ;
	     if ( sum(nWins)>0 )
          if ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.01) )     sig='**'; % .01
          elseif ( max(nWins)./sum(nWins)>=.5+binomial_confidence(sum(nWins),.05) ) sig='* '; % .05
          end
        end

        str=[str sprintf('%35s :',st.di(2).vals{ai})];
        % missing res aware
        str=[str sprintf('%2d %5.3f%2s| ',msum(st.X(spi,ai,labi,:)>0,[1 3 4]),msum(st.X(spi,ai,labi,:),[1 3 4])./max(eps,msum(st.X(spi,ai,labi,:)>0,[1 3 4])),sig)];
        str=[str sprintf('%0.2f\t',st.X(spi,ai,labi,:))];
        str=[str sprintf('\n')];
     end % alg
  end % label
end
return
