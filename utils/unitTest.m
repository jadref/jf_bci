function [testRes,trueRes,diff]=unitTest(testStr,testRes,trueRes,tol)
% simple function to check the accuracy of a test and report the result
if ( nargin < 4 ) 
   if ( isa(trueRes,'double') && isa(testRes,'double') ) tol=1e-11; 
   elseif ( isa(trueRes,'single') || isa(testRes,'single') ) tol=1e-5; 
   elseif ( isa(trueRes,'integer') ) 
      warning('Integer inputs!'); tol=1;       
   elseif ( isa(trueRes,'logical') ) tol=0;
   end
end
diff=abs(testRes-trueRes)./max(1,abs(testRes+trueRes));
fprintf('%60s = %0.3g ',testStr,max(diff(:)));
if ( max(diff(:)) > tol ) 
   if ( exist('mimage') )
      mimage(squeeze(testRes),squeeze(trueRes),squeeze(diff))
   end
   warning([testStr ': failed!']), 
   fprintf('Type return to continue\b');keyboard;
else
   fprintf('Passed \n');
end
