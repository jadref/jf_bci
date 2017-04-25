function []=plotCVLoss(loss,cvLab)
plot(loss');
set(gca,'XTick',[1:size(loss,2)],'XTickLabel',cvLab);
legend('+1','-1');
return;
%--------------------------------------------------------------------------
function testCase()
plotCVLoss(conf2loss(shiftdim(sum(tstconf,1))','perclass'),[1 10 100 1000]);
