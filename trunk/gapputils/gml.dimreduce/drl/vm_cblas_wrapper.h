/***************************************************************************
                             vm_cblas_wrapper.h
                             -------------------
                               W. Michael Brown

  Level 1, 2, and 3 BLAS functionality using this wrapper to GSL cblas
                               
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Sep 19 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_CBLAS_WRAPPER
#define VM_CBLAS_WRAPPER

#include "gsl_cblas/gsl_cblas.h"

YALANAMESPACE

template <class eltype,int,int,int,int>
struct _VM_CBLAS_WRAPPER;

// ------------------------------ Use External Blas
#ifdef YA_BLAS

#include "blas/ya_blas_extern.h"

template <>
struct _VM_CBLAS_WRAPPER<double,1,1,1,1> {
  template <class ya_type1, class ya_type2>
  static inline double cblas_dot(const int N, const ya_type1 X, const int incX,
                           const ya_type2 Y, const int incY) {
    return cblas_ddot(N, X, incX, Y, incY);
  }
  template <class ya_type1>
  static inline double cblas_nrm2(const int N, const ya_type1 X, const int incX) {
    return cblas_dnrm2(N, X, incX);
  }
  template <class ya_type1>
  static inline double cblas_asum(const int N, const ya_type1 X, const int incX) {
    return dasum_(const_cast<int *>(&N), const_cast<double *>(X), 
                  const_cast<int *>(&incX));
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_swap(const int N, ya_type1 X, const int incX, 
                         ya_type2 Y, const int incY) {
    cblas_dswap(N, X, incX, Y, incY);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_copy(const int N, const ya_type1 X, const int incX, 
                         ya_type2 Y, const int incY) {
    cblas_dcopy(N, X, incX, Y, incY);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_axpy(const int N, const double alpha, const ya_type1 X,
                  const int incX, ya_type2 Y, const int incY) {
    cblas_daxpy(N, alpha, X, incX, Y, incY);
  }
  template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
  static inline void cblas_rotg(ya_type1 a, ya_type2 b, ya_type3 c, ya_type4 s) {
    cblas_drotg(a, b, c, s);
  }
  template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
  static inline void cblas_rotmg(ya_type1 d1, ya_type2 d2, ya_type3 b1, const double b2, ya_type4 P) {
    cblas_drotmg(d1, d2, b1, b2, P);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_rot(const int N, ya_type1 X, const int incX,
                 ya_type2 Y, const int incY, const double c, const double  s) {
    cblas_drot(N, X, incX, Y, incY, c,  s);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_rotm(const int N, ya_type1 X, const int incX,
                  ya_type2 Y, const int incY, const ya_type3 P) {
    cblas_drotm(N, X, incX, Y, incY, P);
  }
  template <class ya_type1>
  static inline void cblas_scal(const int N, const double alpha, ya_type1 X, const int incX) {
    cblas_dscal(N, alpha, X, incX);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_gemv(const enum CBLAS_ORDER order,
                  const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                  const double alpha, const ya_type1 A, const int lda,
                  const ya_type2 X, const int incX, const double beta,
                  ya_type3 Y, const int incY) {
    cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_dgbmv(const enum CBLAS_ORDER order,
                  const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                  const int KL, const int KU, const double alpha,
                  const ya_type1 A, const int lda, const ya_type2 X,
                  const int incX, const double beta, ya_type3 Y, const int incY) {
    cblas_dgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, 
                incY);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const ya_type1 A, const int lda, 
                  ya_type2 X, const int incX) {
    cblas_dtrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const int K, const ya_type1 A, const int lda, 
                  ya_type2 X, const int incX) {
    cblas_dtbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const ya_type1 Ap, ya_type2 X, const int incX) {
    cblas_dtpmv(order, Uplo, TransA, Diag, N, Ap, X, incX);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const ya_type1 A, const int lda, ya_type2 X,
                  const int incX) {
    cblas_dtrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const int K, const ya_type1 A, const int lda,
                  ya_type2 X, const int incX) {
    cblas_dtbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                  const int N, const ya_type1 Ap, ya_type2 X, const int incX) {
    cblas_dtpsv(order, Uplo, TransA, Diag, N, Ap, X, incX);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const int N, const double alpha, const ya_type1 A,
                  const int lda, const ya_type2 X, const int incX,
                  const double beta, ya_type3 Y, const int incY) {
    cblas_dsymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_sbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const int N, const int K, const double alpha, const ya_type1 A,
                  const int lda, const ya_type2 X, const int incX,
                  const double beta, ya_type3 Y, const int incY) {
    cblas_dsbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_spmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const int N, const double alpha, const ya_type1 Ap,
                  const ya_type2 X, const int incX,
                  const double beta, ya_type3 Y, const int incY) {
    cblas_dspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
                 const double alpha, const ya_type1 X, const int incX,
                 const ya_type2 Y, const int incY, ya_type3 A, const int lda) {
    cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const ya_type1 X,
                 const int incX, ya_type2 A, const int lda) {
    cblas_dsyr(order, Uplo, N, alpha, X, incX, A, lda);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_spr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const ya_type1 X,
                 const int incX, ya_type2 Ap) {
    cblas_dspr(order, Uplo, N, alpha, X, incX, Ap);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const int N, const double alpha, const ya_type1 X,
                  const int incX, const ya_type2 Y, const int incY, ya_type3 A,
                  const int lda) {
    cblas_dsyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_spr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                  const int N, const double alpha, const ya_type1 X,
                  const int incX, const ya_type2 Y, const int incY, ya_type3 A) {
    cblas_dspr2(order, Uplo, N, alpha, X, incX, Y, incY, A);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const double alpha, const ya_type1 A,
                  const int lda, const ya_type2 B, const int ldb,
                  const double beta, ya_type3 C, const int ldc) {
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, 
               ldc);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                  const enum CBLAS_UPLO Uplo, const int M, const int N,
                  const double alpha, const ya_type1 A, const int lda,
                  const ya_type2 B, const int ldb, const double beta,
                  ya_type3 C, const int ldc) {
    cblas_dsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const double alpha, const ya_type1 A, const int lda,
                  const double beta, ya_type2 C, const int ldc) {
    cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
  }
  template <class ya_type1, class ya_type2, class ya_type3>
  static inline void cblas_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                   const double alpha, const ya_type1 A, const int lda,
                   const ya_type2 B, const int ldb, const double beta,
                   ya_type3 C, const int ldc) {
    cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_trmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                  const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_DIAG Diag, const int M, const int N,
                  const double alpha, const ya_type1 A, const int lda,
                  ya_type2 B, const int ldb) {
    cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
  }
  template <class ya_type1, class ya_type2>
  static inline void cblas_trsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                  const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_DIAG Diag, const int M, const int N,
                  const double alpha, const ya_type1 A, const int lda,
                  ya_type2 B, const int ldb) {
    cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
  }
};

#endif

// ------------------------------ Use Internal BLAS

#define GSL_MIN(a,b) ((a) < (b) ? (a) : (b))
#define INDEX int
#define OFFSET(N, incX) ((incX) > 0 ?  0 : ((N) - 1) * (-(incX)))
#define BLAS_ERROR(x)  std::cerr << "BLAS_ERROR: " << x << std::endl; exit(1);

#define CONJUGATE(x) ((x) == CblasConjTrans)
#define TRANSPOSE(x) ((x) == CblasTrans || (x) == CblasConjTrans)
#define UPPER(x) ((x) == CblasUpper)
#define LOWER(x) ((x) == CblasLower)

/* Handling of packed complex types... */

#define REAL(a,i) (a)[2*(i)]
#define IMAG(a,i) (a)[2*(i)+1]

#define REAL0(a) (a)[0]
#define IMAG0(a) (a)[1]

#define CONST_REAL(a,i) (a)[2*(i)]
#define CONST_IMAG(a,i) (a)[2*(i)+1]

#define CONST_REAL0(a) (a)[0]
#define CONST_IMAG0(a) (a)[1]


#define GB(KU,KL,lda,i,j) ((KU+1+(i-j))*lda + j)

#define TRCOUNT(N,i) ((((i)+1)*(2*(N)-(i)))/2)

/* #define TBUP(N,i,j) */
/* #define TBLO(N,i,j) */

#define TPUP(N,i,j) (TRCOUNT(N,(i)-1)+(j)-(i))
#define TPLO(N,i,j) (((i)*((i)+1))/2 + (j))

template <class eltype,int,int,int,int>
struct _VM_CBLAS_WRAPPER {
  #define BASE eltype

  template <class ya_type1>
  static eltype
  cblas_asum (const int N, const ya_type1 X, const int incX)
  {
  #include "gsl_cblas/source_asum_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_axpy (const int N, const eltype alpha, const ya_type1 X, const int incX,
               ya_type2 Y, const int incY)
  {
  #include "gsl_cblas/source_axpy_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_copy (const int N, const ya_type1 X, const int incX, ya_type2 Y,
               const int incY)
  {
  #include "gsl_cblas/source_copy_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static eltype
  cblas_dot (const int N, const ya_type1 X, const int incX, const ya_type2 Y,
              const int incY)
  {
  #define INIT_VAL  0.0
  #define ACC_TYPE  eltype
  #include "gsl_cblas/source_dot_r.h"
  #undef ACC_TYPE
  #undef INIT_VAL
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_gbmv (const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
               const int M, const int N, const int KL, const int KU,
               const eltype alpha, const ya_type1 A, const int lda,
               const ya_type2 X, const int incX, const eltype beta, 
               ya_type3 Y, const int incY)
  {
  #include "gsl_cblas/source_gbmv_r.h"
  }

  template <class vm_type, class ya_type2, class ya_type3>
  static void
  cblas_gemm (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const eltype alpha, const vm_type A, const int lda,
              const ya_type2 B, const int ldb, const eltype beta, ya_type3 C,
              const int ldc)
  {
    #define YALA_GEMM_R_FIX
    if (Order == CblasRowMajor) {
      vm_type F=A;
      ya_type2 G=B;
      #include "gsl_cblas/source_gemm_r.h"
    } else {
      ya_type2 F=B;
      vm_type G=A;
      #include "gsl_cblas/source_gemm_r.h"
    }
    #undef YALA_GEMM_R_FIX
  }

  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_gemv (const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
               const int M, const int N, const eltype alpha, const ya_type1 A,
               const int lda, const ya_type2 X, const int incX,
               const eltype beta, ya_type3 Y, const int incY)
  {
  #include "gsl_cblas/source_gemv_r.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_ger (const enum CBLAS_ORDER order, const int M, const int N,
              const eltype alpha, const ya_type1 X, const int incX,
              const ya_type2 Y, const int incY, ya_type3 A, const int lda)
  {
  #include "gsl_cblas/source_ger.h"
  }
  
  template <class ya_type1>
  static eltype
  cblas_nrm2 (const int N, const ya_type1 X, const int incX)
  {
  #include "gsl_cblas/source_nrm2_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_rot (const int N, ya_type1 X, const int incX, ya_type2 Y, const int incY,
              const eltype c, const eltype s)
  {
  #include "gsl_cblas/source_rot.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
  static void
  cblas_rotg (ya_type1 a, ya_type2 b, ya_type3 c, ya_type4 s)
  {
  #include "gsl_cblas/source_rotg.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_rotm (const int N, ya_type1 X, const int incX, ya_type2 Y,
               const int incY, const ya_type3 P)
  {
  #include "gsl_cblas/source_rotm.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
  static void
  cblas_rotmg (ya_type1 d1, ya_type2 d2, ya_type3 b1, const eltype b2, ya_type4 P)
  {
  #include "gsl_cblas/source_rotmg.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_sbmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const int K, const eltype alpha, const ya_type1 A,
               const int lda, const ya_type2 X, const int incX,
               const eltype beta, ya_type3 Y, const int incY)
  {
  #include "gsl_cblas/source_sbmv.h"
  }
  
  template <class ya_type1>
  static void
  cblas_scal (const int N, const eltype alpha, ya_type1 X, const int incX)
  {
  #include "gsl_cblas/source_scal_r.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_spmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const eltype alpha, const ya_type1 Ap,
               const ya_type2 X, const int incX, const eltype beta, ya_type3 Y,
               const int incY)
  {
  #include "gsl_cblas/source_spmv.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_spr2 (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const eltype alpha, const ya_type1 X, const int incX,
               const ya_type2 Y, const int incY, ya_type3 Ap)
  {
  #include "gsl_cblas/source_spr2.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_spr (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
              const int N, const eltype alpha, const ya_type1 X, const int incX,
              ya_type2 Ap)
  {
  #include "gsl_cblas/source_spr.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_swap (const int N, ya_type1 X, const int incX, ya_type2 Y,
               const int incY)
  {
  #include "gsl_cblas/source_swap_r.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_symm (const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
               const enum CBLAS_UPLO Uplo, const int M, const int N,
               const eltype alpha, const ya_type1 A, const int lda,
               const ya_type2 B, const int ldb, const eltype beta, ya_type3 C,
               const int ldc)
  {
  #include "gsl_cblas/source_symm_r.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_symv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const eltype alpha, const ya_type1 A, const int lda,
               const ya_type2 X, const int incX, const eltype beta, ya_type3 Y,
               const int incY)
  {
  #include "gsl_cblas/source_symv.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_syr2 (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const eltype alpha, const ya_type1 X, const int incX,
               const ya_type2 Y, const int incY, ya_type3 A, const int lda)
  {
  #include "gsl_cblas/source_syr2.h"
  }
  
  template <class ya_type1, class ya_type2, class ya_type3>
  static void
  cblas_syr2k (const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                const eltype alpha, const ya_type1 A, const int lda,
                const ya_type2 B, const int ldb, const eltype beta, ya_type3 C,
                const int ldc)
  {
  #include "gsl_cblas/source_syr2k_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_syr (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
              const int N, const eltype alpha, const ya_type1 X, const int incX,
              ya_type2 A, const int lda)
  {
  #include "gsl_cblas/source_syr.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_syrk (const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
               const eltype alpha, const ya_type1 A, const int lda,
               const eltype beta, ya_type2 C, const int ldc)
  {
  #include "gsl_cblas/source_syrk_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_tbmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const int K, const ya_type1 A, const int lda,
               ya_type2 X, const int incX)
  {
  #include "gsl_cblas/source_tbmv_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_tbsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const int K, const ya_type1 A, const int lda,
               ya_type2 X, const int incX)
  {
  #include "gsl_cblas/source_tbsv_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_tpmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const ya_type1 Ap, ya_type2 X, const int incX)
  {
  #include "gsl_cblas/source_tpmv_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_tpsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const ya_type1 Ap, ya_type2 X, const int incX)
  {
  #include "gsl_cblas/source_tpsv_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_trmm (const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
               const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_DIAG Diag, const int M, const int N,
               const eltype alpha, const ya_type1 A, const int lda, ya_type2 B,
               const int ldb)
  {
  #include "gsl_cblas/source_trmm_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_trmv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const ya_type1 A, const int lda, ya_type2 X,
               const int incX)
  {
  #include "gsl_cblas/source_trmv_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_trsm (const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
               const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_DIAG Diag, const int M, const int N,
               const eltype alpha, const ya_type1 A, const int lda, ya_type2 B,
               const int ldb)
  {
  #include "gsl_cblas/source_trsm_r.h"
  }
  
  template <class ya_type1, class ya_type2>
  static void
  cblas_trsv (const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const ya_type1 A, const int lda, ya_type2 X,
               const int incX)
  {
  #include "gsl_cblas/source_trsv_r.h"
  }
  
  /*
  template <class ya_type1>
  eltype
  cblas_zasum (const int N, const void *X, const int incX)
  {
  #include "source_asum_c.h"
  }
  
  template <class ya_type1>
  eltype
  cblas_znrm2 (const int N, const void *X, const int incX)
  {
  #include "source_nrm2_c.h"
  }
  */
  
  #undef BASE
};
  
#undef GSL_MIN
#undef INDEX
#undef OFFSET
#undef BLAS_ERROR
#undef CONJUGATE
#undef TRANSPOSE
#undef UPPER
#undef LOWER
#undef REAL
#undef IMAG
#undef REAL0
#undef IMAG0
#undef CONST_REAL
#undef CONST_IMAG
#undef CONST_REAL0
#undef CONST_IMAG0
#undef GB
#undef TRCOUNT

}

#endif
