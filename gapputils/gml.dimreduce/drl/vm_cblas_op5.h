/***************************************************************************
                                vm_cblas_op5.h
                             -------------------
                               W. Michael Brown

  // Calculation of cblas operations c=alpha*a*b+beta*c
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Sep 21 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_CBLAS_OP5_H
#define VM_CBLAS_OP5_H

YALANAMESPACE

// This file chooses the correct CBLAS call using templates based
// on the matrix types

// --- Helper functions for cblas_op3
template<int blas_type,int ans_type>
struct _vm_cblas_op3 {
  template <class eltype, class mat_type, class ya_type2>
  static inline void op3(const eltype alpha,const YA_BaseT &x,ya_type2 &y) {
    YA_DEBUG_ERROR(1==0,"YALA++ Developer Error");
  }
};
// --- Both are Blas types
template<>
struct _vm_cblas_op3<1,1> {
  template <class eltype, class mat_type, class ya_type2>
  static inline void op3(const eltype alpha,const YA_BaseT &x,ya_type2 &y) {
    cblas_axpy(alpha,x,y);    
  }
};
// --- X is a Blas type, y is not
template<>
struct _vm_cblas_op3<1,0> {
  template <class eltype, class mat_type, class ya_type2>
  static inline void op3(const eltype alpha,const YA_BaseT &x,ya_type2 &y) {
    YA_MatT ans(y);
    _vm_cblas_op3<1,1>::op3(alpha,YA_MatT(x),ans);
    y=ans;
  }
};
// --- X is not a blas type
template<int ans_type>
struct _vm_cblas_op3<0,ans_type> {
  template <class eltype, class mat_type, class ya_type2>
  static inline void op3(const eltype alpha,const YA_BaseT &x,ya_type2 &y) {
    _vm_cblas_op3<1,ans_type>::op3(alpha,YA_MatT(x),y);
  }
};
// --- Computes y=alpha*x+y
template <class eltype, class mat_type, class mat_type2>
void cblas_op3(const eltype alpha, const YA_BaseT &x, YA_BaseT2 &y) {
  YA_DEBUG_ERROR(x.numel()==y.numel(),
                 "Cannot use cblas_op3 with different length objects.");
  _vm_cblas_op3<YA_TRAIT::VME_BLAS_STRIDE,
                YA_TRAIT2::VME_BLAS_STRIDE>::op3(alpha,x,y);
}
// --- Computes y=alpha*x+y (const copy)
template <class eltype, class mat_type, class mat_type2>
void cblas_op3(const eltype alpha, const YA_BaseT &x, const YA_BaseT2 &y) {
  YA_DEBUG_ERROR(x.numel()==y.numel(),
                 "Cannot use cblas_op3 with different length objects.");
  _vm_cblas_op3<YA_TRAIT::VME_BLAS_STRIDE,
                YA_TRAIT2::VME_BLAS_STRIDE>::op3(alpha,x,y);
}

// ------------------------------ cblas_op5

// --- Helper functions for cblas_op5
template<int blas_type,int blas_type2,int ans_type,int symm1,int symm2,
         int col1, int col2, int row2>
struct _vm_cblas_op5 {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
           const eltype beta, ya_type3 &c) {
    YA_DEBUG_ERROR(1==0,"YALA++ Developer Error");
  }
};
  
// General case
template<int col1, int row2>
struct _vm_cblas_op5<1,1,1,0,0,col1,0,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    cblas_gemm(alpha,a,b,beta,c);
  }
}; 

// A is a column vector, b is a row vector
template<>
struct _vm_cblas_op5<1,1,1,0,0,1,0,1> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    if (beta==static_cast<eltype>(0)) {
      zero(c);
      cblas_ger(alpha,a,b,c);
    } else {
      c*=beta;
      cblas_ger(alpha,a,b,c);
    }
  }
}; 

// B is a column vector
template<>
struct _vm_cblas_op5<1,1,1,0,0,0,1,0> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    cblas_gemv(alpha,a,b,beta,c);
  }
}; 

// B is symmetric
template<int col1>
struct _vm_cblas_op5<1,1,1,0,1,col1,0,0> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    cblas_symm(CblasRight,CblasUpper,alpha,a,b,beta,c);
  }
}; 

// A is symmetric, B is not a column-vector
template<int row2>
struct _vm_cblas_op5<1,1,1,1,0,0,0,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    cblas_symm(CblasLeft,CblasUpper,alpha,a,b,beta,c);
  }
}; 

// Symmetric A, Column-vector B
template<>
struct _vm_cblas_op5<1,1,1,1,0,0,1,0> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    cblas_symv(CblasUpper,alpha,a,b,beta,c);
  }
}; 

// C is not a BLAS type
template<int symm1, int symm2, int col1, int col2, int row2>
struct _vm_cblas_op5<1,1,0,symm1,symm2,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    YA_MatT ctemp(c);
    _vm_cblas_op5<1,1,1,symm1,symm2,col1,col2,row2>::op5
                             (alpha,a,b,beta,ctemp);
    c=ctemp;
  }
}; 

// B is not a BLAS type and B is symmetric, A is also
template<int ans_type,int col1, int col2, int row2>
struct _vm_cblas_op5<1,0,ans_type,1,1,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    _vm_cblas_op5<1,1,ans_type,1,0,col1,col2,row2>::op5
                             (alpha,a,YA_MatT(b),beta,c);
  }
};

// B is not a BLAS type and B is symmetric, A is not
template<int ans_type,int col1, int col2, int row2>
struct _vm_cblas_op5<1,0,ans_type,0,1,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    YA_MatT symb;
    copy_lower_triangle(symb,b);
    _vm_cblas_op5<1,1,ans_type,0,1,col1,col2,row2>::op5
                             (alpha,a,symb,beta,c);
  }
};

// B is not a BLAS type and B is not symmetric
template<int ans_type,int symm1,int col1, int col2, int row2>
struct _vm_cblas_op5<1,0,ans_type,symm1,0,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    _vm_cblas_op5<1,1,ans_type,symm1,0,col1,col2,row2>::op5
                             (alpha,a,YA_MatT(b),beta,c);
  }
};

// A is not a BLAS type and A is symmetric
template<int blas_type2, int ans_type,int symm2,int col1, int col2, int row2>
struct _vm_cblas_op5<0,blas_type2,ans_type,1,symm2,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    YA_MatT syma;
    copy_lower_triangle(syma,a);
    _vm_cblas_op5<1,blas_type2,ans_type,1,0,col1,col2,row2>::op5
                             (alpha,syma,b,beta,c);
  }
};

// A is not a BLAS type and A is not symmetric
template<int blas_type2, int ans_type,int symm2,int col1, int col2, int row2>
struct _vm_cblas_op5<0,blas_type2,ans_type,0,symm2,col1,col2,row2> {
  template <class eltype, class mat_type, class ya_type2, class ya_type3>
  static inline void op5(const eltype alpha,const YA_BaseT &a, const ya_type2 &b, 
                  const eltype beta, ya_type3 &c) {
    _vm_cblas_op5<1,blas_type2,ans_type,0,symm2,col1,col2,row2>::op5
                             (alpha,YA_MatT(a),b,beta,c);
  }
};

// ---- Cblas operation c=alpha*a*b+beta*c
template<class eltype, class mat_type, class mat_type2, class mat_type3>
inline void cblas_op5(const eltype alpha,const YA_BaseT &a, const YA_BaseT2 &b, 
                      const eltype beta, YA_BaseT3 &c) {
  YA_DEBUG_ERROR(c.rows()==a.rows() && c.cols()==b.cols(),
                 "Dimensions for matrix multiplication do not match.");
  _vm_cblas_op5<YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,
                YA_TRAIT3::VME_BLAS_STRIDE,YA_TRAIT::VME_SYMMETRIC,
                YA_TRAIT2::VME_SYMMETRIC,YA_TRAIT::VME_COL_VECTOR,
                YA_TRAIT2::VME_COL_VECTOR,YA_TRAIT2::VME_ROW_VECTOR>::op5(alpha,
                       a,b,beta,c);
}
// ---- Cblas operation c=alpha*a*b+beta*c (const copy)
template<class eltype, class mat_type, class mat_type2, class mat_type3>
inline void cblas_op5(const eltype alpha,const YA_BaseT &a, const YA_BaseT2 &b, 
                      const eltype beta, const YA_WrapperT3 &c) {
  YA_DEBUG_ERROR(c.rows()==a.rows() && c.cols()==b.cols(),
                 "Dimensions for matrix multiplication do not match.");
  _vm_cblas_op5<YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,
                YA_TRAIT3::VME_BLAS_STRIDE,YA_TRAIT::VME_SYMMETRIC,
                YA_TRAIT2::VME_SYMMETRIC,YA_TRAIT::VME_COL_VECTOR,
                YA_TRAIT2::VME_COL_VECTOR,YA_TRAIT2::VME_ROW_VECTOR>::op5(alpha,
                       a,b,beta,c);
}

}
#endif
