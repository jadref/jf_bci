function [str]=dispPrep(prep)
tmp={prep.timestamp;prep.method;prep.summary};
str=sprintf('\n%s %12s - %s',tmp{:});
return;
%--------------------------------------------------------------
function testCase()
