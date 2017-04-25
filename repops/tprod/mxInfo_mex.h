#ifndef mxInfoMexH
#define mxInfoMexH
/*
  matlab specific helper functions 
*/

#include "mex.h"
#ifndef HAVE_OCTAVE  
/* only in true MATLAB do we have to include matrix.h */
#include "matrix.h"
#endif
#include "mxInfo.h"

mxArray* mkmxArrayCopy(const MxInfo info);
/* convert back to mxArray */
mxArray* mkmxArray(const MxInfo info);
MxInfo mkmxInfoMxArray(const mxArray *mat, int nd);

#endif
