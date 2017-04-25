function [wedIdx,wedQual]=marriageAlgorithm(pairQual)
% implementation of the Gale-Shapely algorithm for the marriage problem
% to find optimal matching between components
% Inputs:
%  pairQual- [nMen x nWomen] matrix of m->w matching qualities (N.B. nMen<=nWomen)
% Output:
%  wedIdx  - [nMen x 1] vector of weddings containing choosen woman for each man
%  wedQual - [nMen x 1] vector of qualities of each wedding
[nMen,nWomen]=size(pairQual);
transposep=0;if ( nMen > nWomen ) transposep=1; pairQual=pairQual'; tmp=nWomen; nWomen=nMen; nMen=tmp; end;
wedIdx=zeros(nMen,1);
wedQual=zeros(nMen,1);
while ( any(wedIdx==0) )
  mi=find(wedIdx==0,1); % get unmatched man
  pairQualmi = pairQual(mi,:);
  [ans,spQii]=sort(pairQualmi,'descend'); % get ordered list of possible pairings qualities
  for wi=spQii; % look through list of women looking for an available one    
    hubby = find(wedIdx==wi); % has this women already got husband?
    if( isempty(hubby) ) % not already married
      wedIdx(mi)=wi; wedQual(mi)=pairQual(mi,wi);  
      break;
    elseif ( pairQual(mi,wi) > pairQual(hubby,wi) ) % better match
      wedIdx(mi)=wi;   wedQual(mi)=pairQual(mi,wi); 
      wedIdx(hubby)=0; wedQual(hubby)=0; % we replace hubby
      break;
    end
  end
end
if ( transposep ) % invert the index expression
  tmp1=wedIdx; tmp2=wedQual;
  wedIdx=zeros(nWomen,1); wedQual=zeros(nWomen,1);
  for i=1:size(tmp1,1); wedIdx(tmp1(i))=i; wedQual(tmp1(i))=tmp2(i); end;
end
return;
%-----------------------------------
function testCase();
qual=randn(10,10);
qual=randn(8,10);
qual=randn(10,8);
[w,wQ]=marriageAlgorithm(qual);
clf;
subplot(2,1,1);imagesc(qual);
subplot(212);wMx=zeros(size(qual)); for i=1:numel(w); if ( w(i)>0 ) wMx(i,w(i))=qual(i,w(i)); end;end; imagesc(wMx);
set(findobj('type','axes'),'clim',[-1 1]*max(abs(qual(:))));
