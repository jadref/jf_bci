/*

Main code for wrapping matlab's matrices in something else.

$Id: mxInfo.c,v 1.15 2007-09-07 13:39:53 jdrf Exp $



Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)
Permission is granted for anyone to copy, use, or modify this
software and accompanying documents for any uncommercial
purposes, provided this copyright notice is retained, and note is
made of any changes that have been made. This software and
documents are distributed without any warranty, express or
implied


*/
#include <cstdlib>
#include <cstring>
#include "mxInfo.hpp"

/*-------------------------------------------------------------------------*/
/* struct to hold useful info for iterating over a n-d matrix              */
/* e.g. for 3 x 3 x 3 matrix:
   ndim=3, numel=27, sz=[2 2 2], stride=[1 3 9] */
export template<class T> CMxInfo<T>::CMxInfo(const int aNd, const int *aSz, const T *aRp, const T *aIp)
{
	if ( aNd==0 )
	{
		iNd = 1; /* always at least 1 dim */
	}
	else
	{
		iNd = aNd;
	}

	/* use a single malloc call to get all the mem we need */
	//iSz = (unsigned int *)CALLOC(iNd*2+1,sizeof(unsigned int));
	iSz = new unsigned int[iNd*2+1];
	iStride = iSz+iNd;
	iNumel     = 1;
	iStride[0] = 1;
	for (int i = 0; i < iNd; i++)
	{  /* copy dims to get result size */
		 iSz[i]= (i<iNd) ? aSz[i] : 1;/* comp the max idx, 1 pad extras */
		 /* compute the x/y strides for this dim */
		 iStride[i+1] = iStride[i]*iSz[i]; /* nd+1 strides */
		 iNumel  *= iSz[i]; /* total matrix size */
	}
	iRp = aRp;
	iIp = aIp;
}

export template<class T> CMxInfo<T>::CMxInfo(const int aNd)
{
	memset(&this, 0, sizeof(CMxInfo)); // fill with zeros
	// now patch
	if ( aNd==0 )
	{
		iNd = 1; /* always at least 1 dim */
	}
	else
	{
		iNd = aNd;
	}
	// NB we leave iSz as NULL!
/*
	MxInfo minfo;
	nd = (nd==0)?1:nd; // always at least 1 dim
	minfo.nd = nd;
	minfo.numel=0;
	// use a single malloc call to get all the mem we need
	minfo.sz     = (int *)CALLOC(nd*2+1,sizeof(int));
	minfo.stride = minfo.sz+nd;
	minfo.rp = 0;
	minfo.ip = 0;
	minfo.dtype = DOUBLE_DTYPE;
	return minfo;
*/
}

export template<class T> CMxInfo<T>::CMxInfo(const CMxInfo* aInfo)
{
  iNd = aInfo->Nd();
  iNumel=aInfo->Numel();
  /* use a single malloc call to get all the mem we need */
  // iSz = (unsigned int *)CALLOC(iNd*2+1,sizeof(unsigned int));
  iSz = new unsigned int[iNd*2+1];
  iStride = iSz+iNd;
  for(int i=0; i<iNd; i++)
  {
	 iSz[i]=aInfo->Sz()[i];
	 iStride[i]=aInfo->Stride()[i];
  }
  iStride[i]=aInfo->Stride()[i]; /* nd+1 valid strides */
  iRp = aInfo->Rp();
  iIp = aInfo->Ip();
}

export template<class T> CMxInfo<T>::~CMxInfo()
{
	delete[] iSz;
	iSz=0;
	iStride=0;
}


export template<class T> void CMxInfo<T>::printMxInfoSummary(wostream& aStream)
{
	aStream << L"[";
	if ( iNd!=1 )
	{
		for ( i=0; i < iNd-1; i++ )
		{
			aStream << iSz[i] << L"x";
		}
		aStream << iSz[i];
	} else {
		aStream << iSz[i] << L"x1";
	}
	aStream << L"] (";
	printTypeName<T>(aStream);
	if (iIp==0 )
	{
		aStream << L")";
	}
	else
	{
		aStream << L" complex)";
	}
}

export template<class T> void CMxInfo<double>::printTypeName(wostream& aStream)
{
	aStream << L"Template Type \"T\"";
}

// template specialisation
export template<double> void CMxInfo<double>::printTypeName(wostream& aStream)
{
	aStream << L"double";
}

export template<float> void CMxInfo<double>::printTypeName(wostream& aStream)
{
	aStream << L"float";
}


export template<class T> void CMxInfo<T>::printMxInfoData(wostream& aStream){
  int i,j,idx;
  int nCol = (iNd>1 ? iSz[1] : 1);
  for ( i=0; i < nCol; i++)
  {
	 for ( j=0; j < iSz[0]; j++)
	 {
		idx = i*iStride[1] + j*iStride[0];
		aStream << L" " << iRp[idx];
		if ( info.ip!=0 )
		{
			aStream << L" " << iIp[idx];
		}
	 }
  }
}

export template<class T> void CMxInfo<T>::printMxInfo(wostream& aStream)
{
  printMxInfoSummary(aStream);
  aStream << endl;
  printMxInfoData(aStream);
  aStream << endl;
}


/* compute if the input info array is contiguous in memory */
export template<class T> int CMxInfo<T>::isContiguous()
{
  for(int d=0; d<iNd; d++)
  {
	 if ( iSstride[d+1] != iStride[d]*iSz[d] )
	 {
		 break;
	 }
  }
  return iStride[0]==1 && d==iNd;
}

export template<class T> void CMxInfo<T>::copyData(const T* aFrom, T* aTo)
{
	const T* xp=aFrom;
	T* zp=aTo;
	size_t dsz=sizeof(T); // NB this is the NATIVE size on the underlying platform. It may not match Matlab!
	T* zendp = zp+iNumel*dsz;
	if ( isContiguous() )
	{
		// simple linear copy
		// use memmove to perform a block memcpy copy, because memmove copes with overlaps correctly
		zp = memmove(zp, xp, (zendp-zp)*dsz); // returns the destination address - must copy number of BYTES
		//while ( zp < zendp ) *zp++ = *xp++;
	} else {
		int i;
		unsigned int *subs = new unsigned int[iNd]; // calls the default int constructor so will be zero'd
		while ( zp < zendp )
		{
			zp[0] = xp[0]; // copy one thing of type T from current address of xp, to current address of zp
			zp += dsz; // step zp
			for(unsigned int i=0; i < iNd; i++ )
			{
				// if reached the last element of this dim
				xp += iStride[i]*dsz;
				subs[i]++;        // move on to the next element
				if( subs[i] < iSz[i] )
				{
					//move this dim on by one and stop!
					break;
				} else {
				 subs[i] = 0; // reset to the start again!
				 xp -= iStride[i]*iSz[i]*dsz;
				}
			}
		}
		delete[] subs;
	}
}


