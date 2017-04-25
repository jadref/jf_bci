function [varargout]=multiCriteriaOpt(obj,objFuzz,mins,verb)
% use the computed measures to pick the 'optimal' fitting parameters
% Use multi-criteria optimisation -> find a non-dominated best trade-off between sse and stability
% search for best point, using multi-objective optimisation
% we want a point which is within a certain distance of the optimal for every criteria
% objFuzz tell's us for each criteria what this distance should be,
% i.e. it is inversely related to that components importance
if ( nargin<4 || isempty(verb) ) verb=1; end;
if ( ndims(obj)>1 ) 
  szObj=size(obj);
  obj=reshape(obj,[prod(szObj(1:end-1)) szObj(end)]);
end
if ( nargin<3 || isempty(mins) )
  mins=min(obj,[],1); %mins(3)=min(obj(:,3)+(obj(:,3)==0));
end
step=1; t=1; bracket=false; optnPts=inf; optPts=[];
for i=1:100;
  pts=true(size(obj));for d=1:size(obj,2); pts(:,d)=obj(:,d)<mins(d)+t*objFuzz(d); end;
  nPts=sum(all(pts,2));
  if ( verb>0 ) 
    fprintf('%2d)\t%5f\t%d\t[%s]\n',i,t,sum(all(pts,2)),sprintf('%d,',find(all(pts,2)))); 
    if ( verb> 1 )
      fprintf('%2d ',1:size(pts,1));fprintf('\n');
      for j=1:size(pts,2); fprintf('%2d ',pts(:,j));fprintf('\n'); end;
    end
  end;
  if ( nPts>0 && nPts<optnPts ) optnPts=nPts; optPts=pts; end;
  if ( nPts==1 ) optPts=pts; break;
  elseif ( ~bracket && nPts==0 ) step=step*1.6; t=t+step; % forward until bracket
  elseif ( ~bracket && nPts>0 )  step=step*.62; t=t-step; bracket=true; 
  elseif ( bracket &&  nPts==0 ) step=step*.62; t=t+step; % golden ratio search
  elseif ( bracket &&  nPts>0  ) step=step*.62; t=t-step;
  end
  if ( step<1e-6 ) break; end;
end
optIdx=find(all(optPts,2));
if ( numel(optIdx)>1 ) % tie-break by 1st objective
  [ans,ti]=min(obj(optIdx,1)); optIdx=optIdx(ti);
end
[is{1:numel(szObj)-1}]=ind2sub(szObj(1:end-1),optIdx);
if ( verb>0 ) 
  fprintf('%d = [%s]\n',optIdx,sprintf('%g,',obj(optIdx,:)));
end
% output in appropriate type
if ( nargout>1 ) varargout=is; else varargout={cat(1,is{:})}; end
return;
%------------------------------------
function testCase();
mins=[min(min(res.obj(:,:,1))) min(min(res.obj(:,:,2))) min(min(res.obj(:,:,3)+(res.obj(:,:,3)==0)))];
multiCriteriaOpt(res.obj,[.5 .05 .6],mins,1);