function [fileregexp]=fileGlob2regexp(filepat)
idx=[1 find(filepat=='.' | filepat=='*') numel(filepat)];
fileregexp=[];
for i=2:numel(idx); 
   fileregexp=[fileregexp filepat(idx(i-1):idx(i)-1)];
   switch ( filepat(idx(i)) )
    case '.'; fileregexp(end+1)='\';
    case '*'; fileregexp(end+1)='.';
   end
end
fileregexp(end+1)=filepat(idx(end));
return;
%---------------------------------------------------------
function testCase()
fileGlob2regexp('*.m')