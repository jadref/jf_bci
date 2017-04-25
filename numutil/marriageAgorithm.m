function [wed,wedQual]=marriageAlgorithm(pairQual)
% implementation of the Gale-Shapely algorithm for the marriage problem
% to find optimal matching between components
wed=zeros(size(pairQual,1),1);
wedQual=zeros(size(pairQual,1),1);
while ( any(wed==0) )
  mi=find(wed==0,1); % get unmatched man
  pairQualmi = pairQual(mi,:);
  [ans,spQii]=sort(pairQualmi); % get ordered list of possible pairings qualities
  for wi=spQii; % look through list of women looking for an available one    
    hubby = find(wed==wi); % has this women already got husband?
    if( isempty(hubby) ) % not already married
      wed(mi)=wi; wedQual(mi)=pairQual(mi,wi);  
      break;
    elseif ( pairQual(mi,wi) > pairQual(hubby,wi) ) % better match
      wed(mi)=wi;   wedQual(mi)=pairQual(mi,wi); 
      wed(hubby)=0; wedQual(hubby)=0; % we replace hubby
      break;
    end
  end
end
return;
%-----------------------------------
function testCase();
marriageAlgorithm(randn(10,10))