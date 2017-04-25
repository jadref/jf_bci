# gprint ( [ fname ] , [ type ] )
# This script is used to simply print the current graphic output to
# either the given file name, in the given type OR by default directly
# to the default printer.  
# 
function [] = gprint ( fname, type )
  # default is to produce eps output.
  gset term postscript eps; 
  if ( nargin >= 1)
	 eval(["gset output '",fname,"'"]) ;
	 if ( nargin == 2)
	  eval(["gset term ",type ]);
	  endif
  else 
	 gset output 'graph.eps'; 
#	 gshow output 
  endif
  replot; 
  # redirect to the printer if no file name given.
  if ( nargin == 0 ) 
	 system("lpr graph.eps; rm graph.eps");	 
  endif 
  gset term x11; 
  replot;
endfunction 
