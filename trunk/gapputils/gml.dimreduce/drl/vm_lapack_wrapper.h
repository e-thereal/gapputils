/***************************************************************************
                             vm_lapack_wrapper.h
                             -------------------
                               W. Michael Brown

  Wrappers for LAPACK 3.1
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sun Sep 9 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef VM_LAPACK_WRAPPER_H
#define VM_LAPACK_WRAPPER_H

#include "vm_lapack.h"

// ------------------------- SYEV

YALANAMESPACE

#ifdef YA_LAPACK
template <class eltype>
int syev_(char *jobz, char *uplo, integer *n, eltype *a,
          integer *lda, eltype *w, eltype *work, integer *lwork, 
          integer *info);

template <>
int syev_<double>(char *jobz, char *uplo, integer *n, double *a,
                integer *lda, double *w, double *work, integer *lwork, 
                integer *info) { 
  return dsyev_(jobz,uplo,n,a,lda,w,work,lwork,info);
}

template <>
int syev_<float>(char *jobz, char *uplo, integer *n, float *a,
                integer *lda, float *w, float *work, integer *lwork, 
                integer *info) { 
  return ssyev_(jobz,uplo,n,a,lda,w,work,lwork,info);
}

#endif

template <class eltype, class mat_type, class ya_type2>
int syev(char jobz, char uplo, YA_BaseT &a, ya_type2 &w, integer &info) {
  #ifdef YA_LAPACK
  YA_DEBUG_ERROR(YA_TRAIT::VME_BLAS && YATRAIT2::VME_BLAS,
                 "Cannot use lapack with matrix type.");
  YA_DEBUG_ERROR1(a.rows()==a.cols(),"syev requires a square matrix.",a);
  YA_DEBUG_ERROR(a.numel()!=0,"syev cannot be used on empty matrix.");

  w.setup(a.rows());

  info=0;
  integer lda=a.rows();
  integer workl=a.numel();
  YA_VecT work(workl,1);
    
  int ret=syev_<eltype>(&jobz,&uplo,&lda,a.begin(),&lda,w.begin(),
                          work.begin(),&workl,&info);
  
  if (info!=0)
    ya_addwarn(320,9,"eigev","LAPACK error: "+ya_to_ascii(info)+" occured.");
  
  return ret;
  #else
  return 0;
  #endif
}

// ------------------------- SYEVD

#ifdef YA_LAPACK

template<class eltype>
int syevd_(char *jobz, char *uplo, integer *n, eltype *a, integer *lda, 
           eltype *w, eltype *work, integer *lwork, integer *iwork, 
           integer *liwork, integer *info);

template<>
int syevd_<double>(char *jobz, char *uplo, integer *n, double *a, integer *lda, 
                   double *w, double *work, integer *lwork, integer *iwork, 
                   integer *liwork, integer *info) {
  return dsyevd_(jobz,uplo,n,a,lda,w,work,lwork,iwork,liwork,info);
}

template<>
int syevd_<float>(char *jobz, char *uplo, integer *n, float *a, integer *lda, 
                  float *w, float *work, integer *lwork, integer *iwork, 
                  integer *liwork, integer *info) {
  return ssyevd_(jobz,uplo,n,a,lda,w,work,lwork,iwork,liwork,info);
}

#endif

template <class eltype, class mat_type, class ya_type2>
int syevd(char jobz, char uplo, YA_BaseT &a, ya_type2 &w, integer &info) {
  #ifdef YA_LAPACK
  YA_DEBUG_ERROR(YA_TRAIT::VME_BLAS && YATRAIT2::VME_BLAS,
                 "Cannot use lapack with matrix type.");
  YA_DEBUG_ERROR1(a.rows()==a.cols(),"syevd requires a square matrix.",a);
  YA_DEBUG_ERROR(a.numel()!=0,"syevd cannot be used on empty matrix.");
  
  w.setup(a.rows());

  info=0;
  integer lda=a.rows();
  integer workl;
  integer iworkl;
  if (jobz=='V') {
    workl=1+6*a.rows()+2*a.numel();
    iworkl=3+5*a.rows();
  } else {
    workl=1+2*a.rows();
    iworkl=1;
  }
  YA_VecT work(workl,1);
  YA_Cols<long,1> iwork(iworkl,1);
    
  int ret=syevd_<eltype>(&jobz,&uplo,&lda,a.begin(),&lda,w.begin(),
                           work.begin(),&workl,iwork.begin(),&iworkl,&info);
    
  if (info!=0)
    ya_addwarn(320,9,"eig","LAPACK error: "+ya_to_ascii(info)+" occured.");
  
  return ret;
  #else
  return 0;
  #endif
}

// ------------------------- SYEVR

#ifdef YA_LAPACK

template<class eltype>
int syevr_(char *jobz, char *range, char *uplo, integer *n, 
           eltype *a, integer *lda, eltype *vl, eltype *vu, integer *il, 
           integer *iu, eltype *abstol, integer *m, eltype *w, 
           eltype *z__, integer *ldz, integer *isuppz, eltype *work, 
           integer *lwork, integer *iwork, integer *liwork, integer *info);
           
template<>
int syevr_<double>(char *jobz, char *range, char *uplo, integer *n, 
                   double *a, integer *lda, double *vl, double *vu, integer *il, 
                   integer *iu, double *abstol, integer *m, double *w, 
                   double *z__, integer *ldz, integer *isuppz, double *work, 
                   integer *lwork, integer *iwork, integer *liwork, 
                   integer *info) {
  return dsyevr_(jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z__,ldz,isuppz,
                 work,lwork,iwork,liwork,info);
}

template<>
int syevr_<float>(char *jobz, char *range, char *uplo, integer *n, 
                  float *a, integer *lda, float *vl, float *vu, integer *il, 
                  integer *iu, float *abstol, integer *m, float *w, 
                  float *z__, integer *ldz, integer *isuppz, float *work, 
                  integer *lwork, integer *iwork, integer *liwork, 
                  integer *info) {
  return ssyevr_(jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z__,ldz,isuppz,
                 work,lwork,iwork,liwork,info);
}

#endif

template <class eltype, class mat_type, class ya_type2, class ya_type3,
          class ya_type4>
int syevr(char jobz, char range, char uplo, YA_BaseT &a, eltype &vl, 
          eltype &vu, integer &il, integer &iu, eltype abstol, integer &m,
          ya_type2 &w, ya_type3 &z, ya_type4 &isuppz, integer &info) {
  #ifdef YA_LAPACK
  YA_DEBUG_ERROR(YA_TRAIT::VME_BLAS && YATRAIT2::VME_BLAS,
                 "Cannot use lapack with matrix type.");
  YA_DEBUG_ERROR1(a.rows()==a.cols(),"syevr requires a square matrix.",a);
  YA_DEBUG_ERROR(a.numel()!=0,"syevr cannot be used on empty matrix.");
  YA_INDEX2_ERROR(il>0 && il<=a.rows() && iu>0 && iu<=a.rows() && 
                  iu>=il || range=='A',
                  "Eigenvalue range exceeds matrix dimensions.",a,il,iu);
  if (range=='A') {
    if (jobz=='V')
      z.setup(a.rows(),a.rows());
    isuppz.setup(2*a.rows());
  } else if (range=='I') {
    YA_DEBUG_ERROR(iu>=il,"il must be <= iu in syevr call.");
    ya_sizet dim=iu-il+1;
    if (jobz=='V')
      z.setup(a.rows(),dim);
    //isuppz.setup(2*dim);
    isuppz.setup(2*a.rows());
  } else {
    YA_DEBUG_ERROR(0==1,"syevr wrapper not implemented for type 'V'");
  }
  
  info=0;
  long lda=a.rows();
  long workl;
  if (a.rows()<26)
    workl=26*a.rows();
  else
    workl=a.numel();
  YA_VecT work(workl,1);
  long iworkl=10*a.rows();
  YA_Cols<long,1> iwork(iworkl,1);
  long mout;
    
  int ret;
  if (range=='A') {
    w.setup(a.rows());
    ret=syevr_<eltype>(&jobz,&range,&uplo,&lda,a.begin(),&lda,&vl,&vu,
                             &il,&iu,&abstol,&mout,w.begin(),z.begin(),
                             &lda,isuppz.begin(),work.begin(),&workl,
                             iwork.begin(),&iworkl,&info);
  } else {
    YA_VecT w_temp(a.rows());
    ret=syevr_<eltype>(&jobz,&range,&uplo,&lda,a.begin(),&lda,&vl,&vu,
                             &il,&iu,&abstol,&mout,w_temp.begin(),z.begin(),
                             &lda,isuppz.begin(),work.begin(),&workl,
                             iwork.begin(),&iworkl,&info);
    
    w=w_temp(vmcount(iu-il+1));
  }

  if (info!=0)
    ya_addwarn(320,9,"eigev","LAPACK error: "+ya_to_ascii(info)+" occured.");
  
  return ret;
  #else
  return 0;
  #endif
}

// ------------------------- SYEVX

#ifdef YA_LAPACK

template <class eltype>
int syevx_(char *jobz, char *range, char *uplo, integer *n, 
         eltype *a, integer *lda, eltype *vl, eltype *vu, integer *il,
         integer *iu, eltype *abstol, integer *m, eltype *w, 
         eltype *z__, integer *ldz, eltype *work, integer *lwork, 
       integer *iwork, integer *ifail, integer *info);

template <>
int syevx_<double>(char *jobz, char *range, char *uplo, integer *n, 
         double *a, integer *lda, double *vl, double *vu, integer *il,
         integer *iu, double *abstol, integer *m, double *w, 
         double *z__, integer *ldz, double *work, integer *lwork, 
       integer *iwork, integer *ifail, integer *info) {
  return dsyevx_(jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z__,ldz,work,
                 lwork,iwork,ifail,info);
}

template <>
int syevx_<float>(char *jobz, char *range, char *uplo, integer *n, 
         float *a, integer *lda, float *vl, float *vu, integer *il,
         integer *iu, float *abstol, integer *m, float *w, 
         float *z__, integer *ldz, float *work, integer *lwork, 
       integer *iwork, integer *ifail, integer *info) {
  return ssyevx_(jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z__,ldz,work,
                 lwork,iwork,ifail,info);
}

#endif

template <class eltype, class mat_type, class ya_type2, class ya_type3,
          class ya_type4>
int syevx(char jobz, char range, char uplo, YA_BaseT &a, eltype &vl, 
          eltype &vu, integer &il, integer &iu, eltype abstol, integer &m,
          ya_type2 &w, ya_type3 &z, ya_type4 &ifail, integer &info) {
  #ifdef YA_LAPACK
  YA_DEBUG_ERROR(YA_TRAIT::VME_BLAS && YATRAIT2::VME_BLAS,
                 "Cannot use lapack with matrix type.");
  YA_DEBUG_ERROR1(a.rows()==a.cols(),"syevx requires a square matrix.",a);
  YA_DEBUG_ERROR(a.numel()!=0,"syevx cannot be used on empty matrix.");
  YA_DEBUG_ERROR(range!='V',"syevx wrapper not implemented for 'V'");
  YA_INDEX2_ERROR(il>0 && il<=a.rows() && iu>0 && iu<=a.rows() && 
                  iu>=il || range=='A',
                  "Eigenvalue range exceeds matrix dimensions.",a,il,iu);

  ya_sizet dim=iu-il+1;
  if (jobz=='V')
    z.setup(a.rows(),dim);
  
  info=0;
  long lda=a.rows();
  long workl;
  if (a.rows()<8)
    workl=8*a.rows();
  else
    workl=a.numel();
  YA_VecT work(workl,1);
  YA_Cols<long,1> iwork(5*a.rows(),1);

  int ret;
  if (range=='A') {
    w.setup(a.rows());
    ret=syevx_<eltype>(&jobz,&range,&uplo,&lda,a.begin(),&lda,&vl,&vu,
                         &il,&iu,&abstol,&m,w.begin(),z.begin(),&lda,
                         work.begin(),&workl,iwork.begin(),ifail.begin(),
                         &info);
  } else {
    YA_VecT w_temp(a.rows());
    ret=syevx_<eltype>(&jobz,&range,&uplo,&lda,a.begin(),&lda,&vl,&vu,
                         &il,&iu,&abstol,&m,w_temp.begin(),z.begin(),&lda,
                         work.begin(),&workl,iwork.begin(),ifail.begin(),
                         &info);
    w=w_temp(vmcount(dim));
  }

  if (info<0)
    ya_addwarn(320,9,"powereigens",
               "LAPACK error: "+ya_to_ascii(info)+" occured.");
  else if (info>0)
    ya_addwarn(305,9,"powereigens",ya_to_ascii(info)+
                     " eigen values failed to converge.");
  
  return ret;
  #else
  return 0;
  #endif
}

}
#endif
