echo on; deli=0;
for i=1:1000
  rsize=floor(rand * 2500)+1;
  fprintf(' %d, ', rsize) ;
  testMem=ones(rsize,1000); testMem2=randn(rsize,1000);
  testMem=testMem.^55 + 4*testMem2;
  eval(['var' num2str(i) '=testMem;']);
  S=whos;
  if ( sum([S(:).bytes]) > 5e8 ) 
    deli=deli+2;%deli=floor(rand * i)+1;
    fprintf(' Del %d,',deli);
    eval(['clear var' num2str(deli) ';']);    
  end
end
quit
