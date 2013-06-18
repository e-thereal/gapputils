/***************************************************************************
                                  ya_cblas.h
                             -------------------
                               W. Michael Brown

  Level 1, 2, and 3 BLAS functionality using this wrapper to GSL cblas
                               
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

#ifndef YA_CBLAS
#define YA_CBLAS

#include "vm_cblas_wrapper.h"

YALANAMESPACE

// BLAS supports row major (1), doesn't (0)
#define YA_ROWBLAS 1

// Returns CblasNoTrans for column major and CblasTrans for row major
template <int order> struct _ya_cblas_trans;
template<> struct _ya_cblas_trans<1> {
  static inline enum CBLAS_TRANSPOSE ans() { return CblasTrans; }
  template<class ya_type>
  static inline ya_sizet rows(const ya_type &m) { return m.cols(); }
  template<class ya_type>
  static inline ya_sizet cols(const ya_type &m) { return m.rows(); }
};
template <int order> struct _ya_cblas_trans {
  static inline enum CBLAS_TRANSPOSE ans() { return CblasNoTrans; }
  template<class ya_type>
  static inline ya_sizet rows(const ya_type &m) { return m.rows(); }
  template<class ya_type>
  static inline ya_sizet cols(const ya_type &m) { return m.cols(); }
};

// Decide whether or not to use column major or row major
template<int oa, int ob> struct _ya_cblas_order;
template<> struct _ya_cblas_order<1,2> { enum {ans=1}; };
template<> struct _ya_cblas_order<2,1> { enum {ans=1}; };
template<> struct _ya_cblas_order<1,1> { enum {ans=1}; };
template<> struct _ya_cblas_order<0,2> { enum {ans=0}; };
template<> struct _ya_cblas_order<2,0> { enum {ans=0}; };
template<> struct _ya_cblas_order<0,0> { enum {ans=0}; };
template<int oa, int ob> struct _ya_cblas_order { enum { ans=-1 }; };

template <class eltype, class mat_type>
inline eltype
cblas_asum (const YA_BaseT &X)
{
  return _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,1,1,1>::
    cblas_asum(X.numel(), X.c_ptr(), X.stride());
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_axpy (const eltype alpha, const YA_BaseT &X, ya_type2 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_axpy(X.numel(), alpha, X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride());
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_copy (const YA_BaseT &X, ya_type2 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_copy(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride());
}

template <class eltype, class mat_type, class mat_type2>
inline eltype
cblas_dot (const YA_BaseT &X, const YA_BaseT2 &Y)
{
  return _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_dot(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride());
}

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_gbmv (const int KL, const int KU, const eltype alpha, const YA_BaseT &A,
            const YA_BaseT2 &X, const eltype beta, ya_type3 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS_STRIDE,1>::
    cblas_gbmv(CblasColMajor, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), 
               _ya_cblas_trans<YA_TRAIT::VME_ORDER>::rows(A),_ya_cblas_trans<YA_TRAIT::VME_ORDER>::cols(A),
               KL, KU, alpha, A.c_ptr(), 
               _ya_cblas_trans<YA_TRAIT::VME_ORDER>::rows(A), X.c_ptr(), 
               X.stride(),beta, Y.c_ptr(),Y.stride());
}

template<int> struct _cblas_gemm_help;
template<> struct _cblas_gemm_help<1> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const eltype alpha, const YA_BaseT &A, const ya_type2 &B, 
                       const eltype beta, ya_type3 &C)
    { f2(alpha,A.T(),B.T(),beta,C); }
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f2(const eltype alpha, const YA_BaseT &A, const ya_type2 &B, 
                        const eltype beta, ya_type3 &C) {
    _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
      cblas_gemm(CblasRowMajor, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(),
                 _ya_cblas_trans<YATRAIT2::VME_ORDER>::ans(), C.rows(), 
                 C.cols(),A.rows(),alpha, A.c_ptr(),_ya_cblas_trans<YA_TRAIT::VME_ORDER>::cols(A), 
                 B.c_ptr(), _ya_cblas_trans<YA_TRAIT::VME_ORDER>::cols(B), beta, C.c_ptr(), C.rows());
  }
};

template<int> struct _cblas_gemm_help {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const eltype alpha, const YA_BaseT &A, const ya_type2 &B, 
                       const eltype beta, ya_type3 &C) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,1>::
    cblas_gemm(CblasColMajor, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), 
               _ya_cblas_trans<YATRAIT2::VME_ORDER>::ans(), C.rows(), 
               C.cols(), A.cols(), alpha, A.c_ptr(), 
               _ya_cblas_trans<YA_TRAIT::VME_ORDER>::rows(A), B.c_ptr(), 
               _ya_cblas_trans<YATRAIT2::VME_ORDER>::rows(B), beta, 
               C.c_ptr(),C.rows());
  }
};

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_gemm (const eltype alpha, const YA_BaseT &A, const YA_BaseT2 &B, 
            const eltype beta, ya_type3 &C)
{
  _cblas_gemm_help<YATRAIT3::VME_ORDER>::f(alpha,A,B,beta,C);
}

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_gemv (const eltype alpha, const YA_BaseT &A, const YA_BaseT2 &X,
               const eltype beta, ya_type3 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS_STRIDE,1>::
  cblas_gemv(CblasColMajor,_ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(),
             _ya_cblas_trans<YA_TRAIT::VME_ORDER>::rows(A),
             _ya_cblas_trans<YA_TRAIT::VME_ORDER>::cols(A),alpha, 
             A.c_ptr(),
             _ya_cblas_trans<YA_TRAIT::VME_ORDER>::rows(A), 
             X.c_ptr(), X.stride(), beta, Y.c_ptr(), Y.stride());
}

template<int> struct _cblas_ger_help;
template<> struct _cblas_ger_help<1> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const eltype alpha, const YA_BaseT &X, const ya_type2 &Y, 
                       ya_type3 &A) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
    cblas_ger(CblasRowMajor, A.rows(), A.cols(), alpha, X.c_ptr(),
              X.stride(), Y.c_ptr(), Y.stride(), A.c_ptr(), A.rows());
  }
};

template<int> struct _cblas_ger_help {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const eltype alpha, const YA_BaseT &X, const ya_type2 &Y, 
                       ya_type3 &A) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS,1>::
    cblas_ger(CblasColMajor, A.rows(), A.cols(), alpha, X.c_ptr(),
              X.stride(), Y.c_ptr(), Y.stride(), A.c_ptr(), A.rows());
  }
};

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_ger (const eltype alpha, const YA_BaseT &X, const YA_BaseT2 &Y, 
           ya_type3 &A)
{
  _cblas_ger_help<YATRAIT3::VME_ORDER>::f(alpha,X,Y,A);
}

template <class eltype, class mat_type>
inline eltype
cblas_nrm2 (const YA_BaseT &X)
{
  return _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,1,1,1>::cblas_nrm2(X.numel(), X.c_ptr(), X.stride());
}

template <class eltype, class mat_type, class mat_type2>
inline void
cblas_rot (YA_BaseT &X, YA_BaseT2 &Y, const eltype c, const eltype s)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_rot(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride(), c,  s);
}

template <class eltype, class mat_type, class mat_type2, class mat_type3>
inline void
cblas_rotm (YA_BaseT2 &X, YA_BaseT3 &Y, const YA_BaseT &P)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,YA_TRAIT3::VME_BLAS,1>::
    cblas_rotm(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride(), P.c_ptr());
}

template <class eltype, class mat_type>
inline void
cblas_rotmg (eltype &d1, eltype &d2, eltype &b1, const eltype b2, YA_BaseT &P)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,1,1,1>::cblas_rotmg(d1, d2, b1, b2, P);
}

// Wrapper copy
template <class eltype, class mat_type>
inline void
cblas_rotmg (eltype &d1, eltype &d2, eltype &b1, const eltype b2, const YA_WrapperT &P)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,1,1,1>::cblas_rotmg(d1, d2, b1, b2, P);
}

/*
template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_sbmv (const enum CBLAS_UPLO Uplo, const int K, const eltype alpha, 
            const YA_BaseT &A, const YA_BaseT2 &X, const eltype beta, 
            ya_type3 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS_STRIDE,1>::
    cblas_sbmv(CblasColMajor, Uplo, A.rows(), K, alpha, A.c_ptr(), A.rows(), 
               X.c_ptr(), X.stride(), beta, Y.c_ptr(), Y.stride());
}

*/

template <class eltype, class mat_type>
inline void
cblas_scal (const eltype alpha, YA_BaseT &X)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,1,1,1>::
    cblas_scal(X.numel(), alpha, X.c_ptr(), X.stride());
}

// Wrapper copy
template <class eltype, class mat_type>
inline void
cblas_scal (const eltype alpha, const YA_WrapperT &X)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,1,1,1>::
    cblas_scal(X.numel(), alpha, X.c_ptr(), X.stride());
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_spmv (const eltype alpha, const VM_SymMat &Ap, const YA_BaseT &X, 
            const eltype beta, ya_type2 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_spmv(CblasColMajor, CblasUpper, Ap.rows(), alpha, Ap.c_ptr(), 
               X.c_ptr(), X.stride(), beta, Y.c_ptr(), Y.stride());
}

template <class eltype, class mat_type, class mat_type2>
inline void
cblas_spr2 (const eltype alpha, const YA_BaseT &X,
            const YA_BaseT2 &Y, VM_SymMat &Ap)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_spr2(CblasColMajor, CblasUpper, Ap.rows(), alpha, X.c_ptr(), 
               X.stride(), Y.c_ptr(), Y.stride(), Ap.c_ptr());
}

template <class eltype, class mat_type>
inline void
cblas_spr (const eltype alpha, const YA_BaseT &X, VM_SymMat &Ap)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,1,1,1>::
    cblas_spr(CblasColMajor, CblasUpper, Ap.rows(), alpha, 
              X.c_ptr(), X.stride(), Ap.c_ptr());
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_swap (YA_BaseT &X, ya_type2 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_swap(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride());
}

// Wrapper Copy
template <class eltype, class mat_type, class ya_type2>
inline void
cblas_swap (const YA_WrapperT &X, ya_type2 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_swap(X.numel(), X.c_ptr(), X.stride(), Y.c_ptr(), Y.stride());
}

template<int> struct _cblas_symm_help;
template<> struct _cblas_symm_help<1> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO uplo,
                       const eltype alpha,
                       const YA_BaseT &A, const ya_type2 &B, const eltype beta,
                       ya_type3 &C) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
    cblas_symm(CblasRowMajor, Side, uplo, C.rows(), C.cols(), alpha, 
               A.c_ptr(), A.rows(), B.c_ptr(), B.rows(), beta, C.c_ptr(), 
               C.rows());
  }
};

template<> struct _cblas_symm_help<0> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_SIDE Side,  const enum CBLAS_UPLO uplo,
                       const eltype alpha,
                       const YA_BaseT &A, const ya_type2 &B, const eltype beta,
                       ya_type3 &C) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,1>::
    cblas_symm(CblasColMajor, Side, uplo, C.rows(), C.cols(), alpha, 
               A.c_ptr(), A.rows(), B.c_ptr(), B.rows(), beta, C.c_ptr(), 
               C.rows());
  }
};

template<int> struct _cblas_symm_help {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_SIDE Side,  const enum CBLAS_UPLO uplo,
                       const eltype alpha,
                       const YA_BaseT &A, const ya_type2 &B, const eltype beta,
                       ya_type3 &C) {
  _VM_CBLAS_WRAPPER<eltype,0,0,0,0>::
    cblas_symm(CblasColMajor, Side, uplo, C.rows(), C.cols(), alpha, 
               A.begin(), A.rows(), B.begin(), B.rows(), beta, C.begin(), 
               C.rows());
  }
};

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_symm (const enum CBLAS_SIDE Side, const enum CBLAS_UPLO uplo, 
            const eltype alpha, const YA_BaseT &A, 
            const YA_BaseT2 &B, const eltype beta, ya_type3 &C)
{
  _cblas_symm_help<_ya_cblas_order<YA_TRAIT2::VME_ORDER,YATRAIT3::VME_ORDER>::ans>::
    f(Side, uplo, alpha, A, B, beta, C);
}

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_symv (const enum CBLAS_UPLO uplo, const eltype alpha, const YA_BaseT &A,
            const YA_BaseT2 &X, const eltype beta, ya_type3 &Y)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS_STRIDE,1>::
    cblas_symv(CblasColMajor, uplo, A.rows(), alpha, A.c_ptr(), A.rows(), 
               X.c_ptr(), X.stride(), beta, Y.c_ptr(), Y.stride());
}

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_syr2 (const enum CBLAS_UPLO uplo, const eltype alpha, const YA_BaseT &X, 
            const YA_BaseT2 &Y, ya_type3 &A)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS_STRIDE,YATRAIT3::VME_BLAS_STRIDE,1>::
    cblas_syr2(CblasColMajor, uplo, A.rows(), alpha, X.c_ptr(), X.stride(), 
               Y.c_ptr(), Y.stride(), A.c_ptr(), A.rows());
}

template<int> struct _cblas_syr2k_help;
template<> struct _cblas_syr2k_help<1> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
                       const eltype alpha, const YA_BaseT &A,
                       const ya_type2 &B, const eltype beta, ya_type3 &C) {
  ya_sizet K;
  if (Trans==CblasNoTrans)
    K=A.cols();
  else
    K=A.rows();
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
    cblas_syr2k(CblasRowMajor, uplo, Trans, C.rows(),K,alpha,A.c_ptr(),K,B.c_ptr(),beta,C.c_ptr(),C.rows());
  }
};

template<> struct _cblas_syr2k_help<0> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
                       const eltype alpha, const YA_BaseT &A,
                       const ya_type2 &B, const eltype beta, ya_type3 &C) {
  ya_sizet K;
  if (Trans==CblasNoTrans)
    K=A.cols();
  else
    K=A.rows();
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,1>::
    cblas_syr2k(CblasColMajor, uplo, Trans, C.rows(),K,alpha,A.c_ptr(),K,B.c_ptr(),beta,C.c_ptr(),C.rows());
  }
};

template<int> struct _cblas_syr2k_help {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
                       const eltype alpha, const YA_BaseT &A,
                       const ya_type2 &B, const eltype beta, ya_type3 &C) {
  ya_sizet K;
  if (Trans==CblasNoTrans)
    K=A.cols();
  else
    K=A.rows();
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,1>::
    cblas_syr2k(CblasColMajor, uplo, Trans, C.rows(),K,alpha,A.begin(),K,B.begin(),beta,C.c_ptr(),C.rows());
  }
};

template <class eltype, class mat_type, class mat_type2, class ya_type3>
inline void
cblas_syr2k (const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
             const eltype alpha, 
             const YA_BaseT &A, const YA_BaseT2 &B, const eltype beta, ya_type3 &C)
{
  _cblas_syr2k_help<_ya_cblas_order<YA_TRAIT::VME_ORDER,YA_TRAIT2::VME_ORDER>::ans>::
    f(Trans,alpha,A,B,beta,C);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_syr (const enum CBLAS_UPLO uplo, const eltype alpha, const YA_BaseT &X, ya_type2 &A)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS_STRIDE,YATRAIT2::VME_BLAS,1,1>::
    cblas_syr(CblasColMajor, uplo, A.rows(), alpha, X.c_ptr(), 
              X.stride(), A.c_ptr(), A.rows());
}

template<int> struct _cblas_syrk_help;
template<> struct _cblas_syrk_help<1> {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
                       const eltype alpha, const YA_BaseT &A,
                       const eltype beta, ya_type3 &C) {
  ya_sizet K;
  if (Trans==CblasNoTrans)
    K=A.cols();
  else
    K=A.rows();
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
    cblas_syrk(CblasRowMajor, uplo, Trans, C.rows(),K,alpha,A.c_ptr(),K,beta,C.c_ptr(),C.rows());
  }
};

template<int> struct _cblas_syrk_help {
  template<class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void f(const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
                       const eltype alpha, const YA_BaseT &A,
                       const eltype beta, ya_type3 &C) {
  ya_sizet K;
  if (Trans==CblasNoTrans)
    K=A.cols();
  else
    K=A.rows();
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,YATRAIT3::VME_BLAS,YA_ROWBLAS>::
    cblas_syrk(CblasColMajor, uplo, Trans, C.rows(),K,alpha,A.c_ptr(),K,beta,C.c_ptr(),C.rows());
  }
};

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_syrk (const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE Trans, 
            const eltype alpha, const YA_BaseT &A, 
            const eltype beta, ya_type2 &C)
{
  _cblas_syrk_help<YA_TRAIT::VME_ORDER>::f(Trans,alpha,A,beta,C);
}

/*
template <class eltype, class mat_type, class ya_type2>
inline void
cblas_tbmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
             const int N, const int K, const YA_BaseT &A, const int lda,
             ya_type2 &X, const int incX)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,1>::
    cblas_tbmv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_tbsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
             const int N, const int K, const YA_BaseT &A, const int lda,
             ya_type2 &X, const int incX)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,1>::
    cblas_tbsv(CblasColMajor, Uplo, TransA, Diag, N, K, A, lda, X, incX);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_tpmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
             const int N, const YA_BaseT &Ap, ya_type2 &X, const int incX)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,1>::
    cblas_tpmv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_tpsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
             const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
             const int N, const YA_BaseT &Ap, ya_type2 &X, const int incX)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,1>::
    cblas_tpsv(CblasColMajor, Uplo, TransA, Diag, N, Ap, X, incX);
}
*/

template<int> struct _cblas_trmm_help;
template<> struct _cblas_trmm_help<1> {
  template<class eltype, class mat_type, class ya_type2>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO uplo,
                       const enum CBLAS_TRANSPOSE TransA,const enum CBLAS_DIAG Diag, 
                       const eltype alpha, const YA_BaseT &A, ya_type2 &B) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,YA_ROWBLAS>::
    cblas_trmm(CblasRowMajor, Side, uplo, TransA, Diag, B.rows(), B.columns(), alpha, 
               A.c_ptr(), A.rows(), B.c_ptr(), B.rows());
  }
};

template<> struct _cblas_trmm_help<0> {
  template<class eltype, class mat_type, class ya_type2>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO uplo,
                       const enum CBLAS_TRANSPOSE TransA,const enum CBLAS_DIAG Diag, 
                       const eltype alpha, const YA_BaseT &A, ya_type2 &B) {
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS,1,YA_ROWBLAS>::
    cblas_trmm(CblasColMajor, Side, uplo, TransA, Diag, B.rows(), B.columns(), alpha, 
               A.c_ptr(), A.rows(), B.c_ptr(), B.rows());
  }
};

template<int> struct _cblas_trmm_help {
  template<class eltype, class mat_type, class ya_type2>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO uplo,
                       const enum CBLAS_TRANSPOSE TransA,const enum CBLAS_DIAG Diag, 
                       const eltype alpha, const YA_BaseT &A, ya_type2 &B) {
  _VM_CBLAS_WRAPPER<eltype,0,0,0,0>::
    cblas_trmm(CblasColMajor, Side, uplo, TransA, Diag, B.rows(), B.columns(), alpha, 
               A.begin(), A.rows(), B.begin(), B.rows());
  }
};

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_trmm (const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, 
            const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, 
            const eltype alpha, const YA_BaseT &A, ya_type2 &B)
{
  _cblas_trmm_help<_ya_cblas_order<YA_TRAIT::VME_ORDER,YATRAIT2::VME_ORDER>::ans>::
    f(Side, Uplo, TransA, Diag, alpha, A, B);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_trmv (const enum CBLAS_UPLO Uplo, const enum CBLAS_DIAG Diag,
            const YA_BaseT &A, ya_type2 &X)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_trmv(CblasColMajor, Uplo, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), Diag, A.rows(), 
               A.c_ptr(), A.rows(), X.c_ptr(), X.stride());
}

template<int> struct _ya_trsm_help;
template<> struct _ya_trsm_help<1> {
  template<class eltype, class mat_type, class mat_type2>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                       const enum CBLAS_DIAG Diag, const eltype alpha, 
                       const YA_BaseT &A, YA_BaseT2 &B) {
    f2(Side,Uplo,Diag,alpha,A.T(),B);
  }
  template<class eltype, class mat_type, class mat_type2>
  static inline void f2(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_DIAG Diag, const eltype alpha, 
                 const YA_BaseT &A, YA_BaseT2 &B) {
    _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS,1,YA_ROWBLAS>::
      cblas_trsm(CblasRowMajor, Side, Uplo, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), 
                 Diag, B.rows(), B.cols(), alpha, A.c_ptr(), A.cols(), B.c_ptr(), B.rows());
  }
};

template<int> struct _ya_trsm_help;
template<> struct _ya_trsm_help<0> {
  template<class eltype, class mat_type, class mat_type2>
  static inline void f(const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_DIAG Diag, const eltype alpha, 
                const YA_BaseT &A, YA_BaseT2 &B) {
    _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YA_TRAIT2::VME_BLAS,1,1>::
      cblas_trsm(CblasColMajor, Side, Uplo, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), 
                 Diag, B.rows(), B.cols(), alpha, A.c_ptr(), A.rows(), B.c_ptr(), B.rows());
  }
};

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_trsm (const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
            const enum CBLAS_DIAG Diag, const eltype alpha, 
            const YA_BaseT &A, ya_type2 &B)
{
  _ya_trsm_help<YATRAIT2::VME_BLAS>::f(Side,Uplo,Diag,alpha,A,B);
}

template <class eltype, class mat_type, class ya_type2>
inline void
cblas_trsv (const enum CBLAS_UPLO Uplo, const enum CBLAS_DIAG Diag,
            const YA_BaseT &A, ya_type2 &X)
{
  _VM_CBLAS_WRAPPER<eltype,YA_TRAIT::VME_BLAS,YATRAIT2::VME_BLAS_STRIDE,1,1>::
    cblas_trsv(CblasColMajor, Uplo, _ya_cblas_trans<YA_TRAIT::VME_ORDER>::ans(), 
               Diag, X.numel(), A.c_ptr(), A.rows(), X.c_ptr(), X.stride());
}

/*
template <class eltype, class mat_type>
eltype
cblas_zasum (const int N, const void *X, const int incX)
{
#define BASE eltype
#include "source_asum_c.h"
#undef BASE
}

template <class eltype, class mat_type>
eltype
cblas_znrm2 (const int N, const void *X, const int incX)
{
#define BASE eltype
#include "source_nrm2_c.h"
#undef BASE
}
*/

}
#endif
