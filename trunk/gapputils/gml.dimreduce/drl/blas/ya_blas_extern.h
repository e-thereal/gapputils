/***************************************************************************
                               ya_blas_extern.h
                             -------------------
                               W. Michael Brown

  External prototypes for BLAS functionality
                               
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Aug 13 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_BLAS_EXTERN_H
#define YA_BLAS_EXTERN_H

extern "C" {

float sdsdot_(int *, float *, float *, int *, float *, int *);
double dsdot_(int *, float *, int *, float *, int *);
float sdot_(int *, float *, int *, float *, int *);
double ddot_(int *, double *, int *, double *, int *);
void cdotu_sub_(int *, void *, int *, void *, int *, void *);
void cdotc_sub_(int *, void *, int *, void *, int *, void *);
void zdotu_sub_(int *, void *, int *, void *, int *, void *);
void zdotc_sub_(int *, void *, int *, void *, int *, void *);
float snrm2_(int *, float *, int *);
float sasum_(int *, float *, int *);
double dnrm2_(int *, double *, int *);
double dasum_(int *, double *, int *);
float scnrm2_(int *, void *, int *);
float scasum_(int *, void *, int *);
double dznrm2_(int *, void *, int *);
double dzasum_(int *, void *, int *);
int isamax_(int *, float *, int *);
int idamax_(int *, double *, int *);
int icamax_(int *, void   *, int *);
int izamax_(int *, void   *, int *);
void sswap_(int *, float *, int *, float *, int *);
void scopy_(int *, float *, int *, float *, int *);
void saxpy_(int *, float *, float *, int *, float *, int *);
void dswap_(int *, double *, int *, double *, int *);
void dcopy_(int *, double *, int *, double *, int *);
void daxpy_(int *, double *, double *, int *, double *, int *);
void cswap_(int *, void *, int *, void *, int *);
void ccopy_(int *, void *, int *, void *, int *);
void caxpy_(int *, void *, void *, int *, void *, int *);
void zswap_(int *, void *, int *, void *, int *);
void zcopy_(int *, void *, int *, void *, int *);
void zaxpy_(int *, void *, void *, int *, void *, int *);
void srotg_(float *, float *, float *, float *s);
void srotmg_(float *, float *, float *, float *, float *);
void srot_(int *, float *, int *, float *, int *, float *, float *);
void srotm_(int *, float *, int *, float *, int *, float *);
void drotg_(double *, double *, double *, double *);
void drotmg_(double *, double *, double *, double *, double *);
void drot_(int *, double *, int *, double *, int *, double *, double *);
void drotm_(int *, double *, int *, double *, int *, double *);
void sscal_(int *, float *, float *, int *);
void dscal_(int *, double *, double *, int *);
void cscal_(int *, void *, void *, int *);
void zscal_(int *, void *, void *, int *);
void csscal_(int *, float *, void *, int *);
void zdscal_(int *, double *, void *, int *);
void sgemv_(char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void sgbmv_(char *, int *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void strmv_(char *, char *, char *, int *, float *, int *, float *, int *);
void stbmv_(char *, char *, char *, int *, int *, float *, int *, float *, int *);
void stpmv_(char *, char *, char *, int *, float *, float *, int *);
void strsv_(char *, char *, char *, int *, float *, int *, float *, int *);
void stbsv_(char *, char *, char *, int *, int *, float *, int *, float *, int *);
void stpsv_(char *, char *, char *, int *, float *, float *, int *);
void dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dgbmv_(char *, int *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dtrmv_(char *, char *, char *, int *, double *, int *,  double *, int *);
void dtbmv_(char *, char *, char *, int *, int *, double *, int *, double *, int *);
void dtpmv_(char *, char *, char *, int *, double *, double *, int *);
void dtrsv_(char *, char *, char *, int *, double *, int *, double *, int *);
void dtbsv_(char *, char *, char *, int *, int *, double *, int *, double *, int *);
void dtpsv_(char *, char *, char *, int *, double *, double *, int *);
void cgemv_(char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void cgbmv_(char *, int *, int *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void ctrmv_(char *, char *, char *, int *, void *, int *, void *, int *);
void ctbmv_(char *, char *, char *, int *, int *, void *, int *, void *, int *);
void ctpmv_(char *, char *, char *, int *, void *, void *, int *);
void ctrsv_(char *, char *, char *, int *, void *, int *, void *, int *);
void ctbsv_(char *, char *, char *, int *, int *, void *, int *, void *, int *);
void ctpsv_(char *, char *, char *, int *, void *, void *, int *);
void zgemv_(char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zgbmv_(char *, int *, int *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void ztrmv_(char *, char *, char *, int *, void *, int *, void *, int *);
void ztbmv_(char *, char *, char *, int *, int *, void *, int *, void *, int *);
void ztpmv_(char *, char *, char *, int *, void *, void *, int *);
void ztrsv_(char *, char *, char *, int *, void *, int *, void *, int *);
void ztbsv_(char *, char *, char *, int *, int *, void *, int *, void *, int *);
void ztpsv_(char *, char *, char *, int *, void *, void *, int *);
void ssymv_(char *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void ssbmv_(char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void sspmv_(char *, int *, float *, float *, float *, int *, float *, float *, int *);
void sger_(int *, int *, float *, float *, int *, float *, int *, float *, int *);
void ssyr_(char *, int *, float *, float *, int *, float *, int *);
void sspr_(char *, int *, float *, float *, int *, float *);
void ssyr2_(char *, int *, float *, float *, int *, float *, int *, float *, int *);
void sspr2_(char *, int *, float *, float *, int *, float *, int *, float *);
void dsymv_(char *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dsbmv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dspmv_(char *, int *, double *, double *, double *, int *, double *, double *, int *);
void dger_(int *, int *, double *, double *, int *, double *, int *, double *, int *);
void dsyr_(char *, int *, double *, double *, int *, double *, int *);
void dspr_(char *, int *, double *, double *, int *, double *);
void dsyr2_(char *, int *, double *, double *, int *, double *, int *, double *, int *);
void dspr2_(char *, int *, double *, double *, int *, double *, int *, double *);
void chemv_(char *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void chbmv_(char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void chpmv_(char *, int *, void *, void *, void *, int *, void *, void *, int *);
void cgeru_(int *, int *, void *, void *, int *, void *, int *, void *, int *);
void cgerc_(int *, int *, void *, void *, int *, void *, int *, void *, int *);
void cher_(char *, int *, float *, void *, int *, void *, int *);
void chpr_(char *, int *, float *, void *, int *, void *);
void cher2_(char *, int *, void *, void *, int *, void *, int *, void *, int *);
void chpr2_(char *, int *, void *, void *, int *, void *, int *, void *);
void zhemv_(char *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zhbmv_(char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zhpmv_(char *, int *, void *, void *, void *, int *, void *, void *, int *);
void zgeru_(int *, int *, void *, void *, int *, void *, int *, void *, int *);
void zgerc_(int *, int *, void *, void *, int *, void *, int *, void *, int *);
void zher_(char *, int *, double *, void *, int *, void *, int *);
void zhpr_(char *, int *, double *, void *, int *, void *);
void zher2_(char *, int *, void *, void *, int *, void *, int *, void *, int *);
void zhpr2_(char *, int *, void *, void *, int *, void *, int *, void *);
void sgemm_(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void ssymm_(char *, char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void ssyrk_(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *);
void ssyr2k_(char *, char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
void strmm_(char *, char *, char *, char *, int *, int *, float *, float *, int *, float *, int *);
void strsm_(char *, char *, char *, char *, int *, int *, float *, float *, int *, float *, int *);
void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dsymm_(char *, char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dsyrk_(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *);
void dsyr2k_(char *, char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
void dtrmm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *, int *);
void dtrsm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *, int *);
void cgemm_(char *, char *, int *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void csymm_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void csyrk_(char *, char *, int *, int *, void *, void *, int *, void *, void *, int *);
void csyr2k_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void ctrmm_(char *, char *, char *, char *, int *, int *, void *, void *, int *, void *, int *);
void ctrsm_(char *, char *, char *, char *, int *, int *, void *, void *, int *, void *, int *);
void zgemm_(char *, char *, int *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zsymm_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zsyrk_(char *, char *, int *, int *, void *, void *, int *, void *, void *, int *);
void zsyr2k_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void ztrmm_(char *, char *, char *, char *, int *, int *, void *, void *, int *, void *, int *);
void ztrsm_(char *, char *, char *, char *, int *, int *, void *, void *, int *, void *, int *);
void chemm_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void cherk_(char *, char *, int *, int *, float *, void *, int *, float *, void *, int *);
void cher2k_(char *, char *, int *, int *, void *, void *, int *, void *, int *, float *, void *, int *);
void zhemm_(char *, char *, int *, int *, void *, void *, int *, void *, int *, void *, void *, int *);
void zherk_(char *, char *, int *, int *, double *, void *, int *, double *, void *, int *);
void zher2k_(char *, char *, int *, int *, void *, void *, int *, void *, int *, double *, void *, int *);

}

#endif
