#ifndef __MXINFO_H__
#define __MXINFO_H__
/*

  Header file for a simple wrapper for matlabs array types.

  $Id: mxInfo.h,v 1.11 2007-09-07 13:41:17 jdrf Exp $

 Copyright 2006-     by Jason D.R. Farquhar (jdrf@zepler.org)
 Permission is granted for anyone to copy, use, or modify this
 software and accompanying documents for any uncommercial
 purposes, provided this copyright notice is retained, and note is
 made of any changes that have been made. This software and
 documents are distributed without any warranty, express or
 implied



 */
#include <ostream>

using namespace std;

/* enum list for different data types -- MATCHED TO THE MATLAB ONES */
/* DEFINE LIST OF DATA TYPES -- N.B. use defines for pre-processor */
#define  LOGICAL_DTYPE 3
#define  CHAR_DTYPE    4
#define  DOUBLE_DTYPE  6
#define  SINGLE_DTYPE  7
#define  INT8_DTYPE    8
#define  UINT8_DTYPE   9
#define  INT16_DTYPE   10
#define  UINT16_DTYPE  11
#define  INT32_DTYPE   12
#define  UINT32_DTYPE  13
#define  INT64_DTYPE   14
#define  UINT64_DTYPE  15
typedef int MxInfoDTypes;

/*-------------------------------------------------------------------------*/
/* struct to hold useful info for iterating over a n-d matrix              */
/* e.g. for 3 x 3 x 3 matrix:
   ndim=3, numel=27, sz=[2 2 2], stride=[1 3 9] */
template<class T> class CMxInfo
{
public:
	// construction & destruction
	CMxInfo(const int aNd, const int* aSz, const T* aRp, const T* aIp);
	CMxInfo(const int aNd);
	CMxInfo(const CMxInfo* aInfo);
	~CMxInfo();
	// helpers - private?
	int isContiguous();
	void copyData(const T* aFrom, T* aTo);
	// inlined methods
	inline void initmxInfo() { if (iSz) delete[] iSz; iSz=0; };
	inline int stride(int aI) { return (aI<iNd) ? iSz[aI] : 1; };
	inline int sz(int aI) { return (aI<iNd+1) ? iStride[aI] : iStride[iNd]; };
	// IO helpers
	void printMxInfoSummary(wostream& aStream);
	void printMxInfoData(wostream& aStream);
	void printMxInfo(wostream& aStream);
	void printTypeName(wostream& aStream);

	// getters
	inline const unsigned int Nd() const { return iNd; };
	inline const unsigned int Numel() const { return iNumel; };
	inline const unsigned int* Sz() const { return iSz; };
	inline const unsigned int* Stride() const { return iStride; };
	inline const T* Rp() const { return iRp; };
	inline const T* Ip() const { return iIp; };

private:
	unsigned int 	iNd;          /* number of dimensions of the matrix */
	unsigned int 	iNumel;       /* total number elements in the matrix */
	unsigned int*	iSz;         /* size of each matrix dimension */
	unsigned int*	iStride;     /* per dimension stride, up to 1 past nd */
	T*				iRp;      /* real data pointer */
	T*				iIp;      /* imaginary data pointer */
};

#endif
