/***************************************************************************
                              vm_matrix_expr.h
                             -------------------
                               W. Michael Brown

  Binary Matrix Expression Templates for Vector/Matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Feb 16 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_MATRIX_EXPR_H
#define VM_MATRIX_EXPR_H

YALANAMESPACE

// ------------------------- Expression Templates
// ------------------------- Operation Functors
template<class eltype, class ya_type1> class VMExprOpMax {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i) {
  if (one.rows()==1)
    return maximum(one);
  eltype result=one.at2(0,i);
  ya_sizet oR=one.rows();
  for (ya_sizet k=1; k<oR; k++)
    if (one.at2(k,i)>result)
      result=one.at2(k,i);
  return result; }
  static inline ya_sizet len(const ya_type1 &one) {
    if (one.rows()==1) return 1; else return one.cols(); }
};
template<class eltype, class ya_type1> class VMExprOpSum {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i) {
  if (one.rows()==1)
    return total(one);
  eltype result=one.at2(0,i);
  ya_sizet oR=one.rows();
  for (ya_sizet k=1; k<oR; k++)
    result+=one.at2(k,i);
  return result; }
  static inline ya_sizet len(const ya_type1 &one) {
    if (one.rows()==1) return 1; else return one.cols(); }
};
template<class eltype, class ya_type1> class VMExprOpSumR {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i) {
  if (one.cols()==1)
    return total(one);
  eltype result=one.at2(i,0);
  ya_sizet oC=one.cols();
  for (ya_sizet k=1; k<oC; k++)
    result+=one.at2(i,k);
  return result; }
  static inline ya_sizet len(const ya_type1 &one) {
    if (one.cols()==1) return 1; else return one.rows(); }
};
template<class eltype, class ya_type1> class VMExprOpProd {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i) {
  if (one.rows()==1)
    return product(one);
  eltype result=one.at2(0,i);
  ya_sizet oR=one.rows();
  for (ya_sizet k=1; k<oR; k++)
    result*=one.at2(k,i);
  return result; }
  static inline ya_sizet len(const ya_type1 &one) {
    if (one.rows()==1) return 1; else return one.cols(); }
};
template<class eltype, class ya_type1> class VMExprOpDMat2 {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i, const ya_sizet j) {
  if (i==j)
    return 0;
  return dist2(one(i,":"),one(j,":")); }
  static inline ya_sizet rows(const ya_type1 &one) { return one.rows(); }
  static inline ya_sizet columns(const ya_type1 &one) { return one.rows(); }
  static inline ya_sizet length(const ya_type1 &one) { return one.rows()*one.rows(); }
};
template<class eltype, class ya_type1> class VMExprOpDMat {
 public: static inline eltype op(const ya_type1 &one, const ya_sizet i, const ya_sizet j) {
  if (i==j)
    return 0;
  return sqrt(dist2(one(i,":"),one(j,":"))); }
  static inline ya_sizet rows(const ya_type1 &one) { return one.rows(); }
  static inline ya_sizet columns(const ya_type1 &one) { return one.rows(); }
  static inline ya_sizet length(const ya_type1 &one) { return one.rows()*one.rows(); }
};
template<class eltype, class ya_type1, class ya_type2> class VMExprOpMatMul {
 public: static inline eltype op(const ya_type1 &one, const ya_type2 &two,
                            const ya_sizet i, const ya_sizet j) {
  if (YATRAIT1::VME_DIAGONAL)
    return one.at2(i,i)*two.at2(i,j);
  else if (YATRAIT2::VME_DIAGONAL)
    return one.at2(i,j)*two.at2(j,j);
  else {
    eltype result=0;
    ya_sizet oC=one.cols();
    for (ya_sizet k=0; k<oC; k++)
      result+=one.at2(i,k)*two.at2(k,j);
    return result;
  }
} };
template<class ya_type1, class ya_type2> class VMOpMatMulSize {
 public:
   static inline ya_sizet rows(const ya_type1 &one, const ya_type2 &two) { return one.rows(); }
   static inline ya_sizet columns(const ya_type1 &one, const ya_type2 &two) { return two.cols(); }
   static inline ya_sizet length(const ya_type1 &one, const ya_type2 &two) { return one.rows()*two.cols(); }
};



// ------------------------- Expression Template Implementations
template<class eltype, class mat_type, class op, class stor1>
class VM_Unary_Row : public YA_Common<eltype, VM_UnaryRow> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Unary_Row>::cref cref;
  
  VM_Unary_Row(const mat_type &l) : one(l) {}

  inline cref at(const ya_sizet i) const
    { return op::op(one,i); }
  inline cref at(const ya_sizet i)
    { return op::op(one,i); }
  using YA_Common<eltype, VM_UnaryRow>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one,j);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one,j);
  }
  inline ya_sizet rows() const { return 1; }
  inline ya_sizet cols() const { return op::len(one); }
  inline ya_sizet numel() const { return op::len(one); }

 protected:
  stor1 one;
};

template<class eltype, class mat_type, class op, class stor1>
class VM_Unary_Col : public YA_Common<eltype, VM_UnaryCol> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Unary_Col>::cref cref;

  VM_Unary_Col(const mat_type &l) : one(l) {}

  inline cref at(const ya_sizet i) const
    { return op::op(one,i); }
  inline cref at(const ya_sizet i)
    { return op::op(one,i); }
  using YA_Common<eltype, VM_UnaryCol>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one,j);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one,j);
  }
  inline ya_sizet rows() const { return op::len(one); }
  inline ya_sizet cols() const { return 1; }
  inline ya_sizet numel() const { return op::len(one); }

 protected:
  stor1 one;
};

template<class eltype, class mat_type, class op, class stor1>
class VM_Unary_Mat : public YA_Common<eltype, VM_UnaryMat> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Unary_Mat>::cref cref;

  VM_Unary_Mat(const mat_type &l) : one(l) {}

  inline cref at(const ya_sizet i) const
    { div_t index=div(i,rows());
      return op::op(one,index.rem,index.quot); }
  inline cref at(const ya_sizet i)
    { div_t index=div(i,rows());
      return op::op(one,index.rem,index.quot); }
  using YA_Common<eltype, VM_UnaryMat>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one,i,j);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one,i,j);
  }
  inline ya_sizet rows() const { return op::rows(one); }
  inline ya_sizet cols() const { return op::columns(one); }
  inline ya_sizet numel() const { return op::length(one); }
 protected:
  stor1 one;
};

template<class eltype, class mat_type, class mat_type2, class op, class size, class stor1, class stor2>
class VM_Matrix_Expr : public YA_Common<eltype, VM_MatrixBin> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Matrix_Expr>::cref cref;

  VM_Matrix_Expr(const mat_type &l, const mat_type2 &r) : one(l), two(r) {}

  inline cref at(const ya_sizet i) const
    { div_t index=div(i,rows());
      return op::op(one,two,index.rem,index.quot); }
  inline cref at(const ya_sizet i)
    { div_t index=div(i,rows());
      return op::op(one,two,index.rem,index.quot); }
  using YA_Common<eltype, VM_MatrixBin>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one,two,i,j);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one,two,i,j);
  }
  inline ya_sizet rows() const { return size::rows(one,two); }
  inline ya_sizet cols() const { return size::columns(one,two); }
  inline ya_sizet numel() const { return size::length(one,two); }
 protected:
  stor1 one;
  stor2 two;
};

template<class eltype, class mat_type>
class VM_Size_Mat : public YA_Common<eltype, VM_Size_Mat<eltype, mat_type> > {
 public:

  /// Return types for indexing into this matrix
  typedef ya_sizet cref;

  VM_Size_Mat(const mat_type &l) : one(l) {}

  inline cref at(const ya_sizet i) const
    { if (i==0) return one.rows(); return one.cols(); }
  inline cref at(const ya_sizet i)
    { if (i==0) return one.rows(); return one.cols(); }
  using YA_Common<eltype, VM_Size_Mat>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return at(j);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return at(j);
  }
  inline ya_sizet rows() const { return 1; }
  inline ya_sizet cols() const { return 2; }
  inline ya_sizet numel() const { return 2; }

 protected:
  const mat_type& one;
};

#define VM_EMax VM_Unary_Row<eltype,mat_type,VMExprOpMax<eltype,mat_type> >
#define VM_ESum VM_Unary_Row<eltype,mat_type,VMExprOpSum<eltype,mat_type> >
#define VM_ESumR VM_Unary_Col<eltype,mat_type,VMExprOpSumR<eltype,mat_type> >
#define VM_EProd VM_Unary_Row<eltype,mat_type,VMExprOpProd<eltype,mat_type> >
#define VM_EDMat2 VM_Unary_Mat<eltype,mat_type,VMExprOpDMat2<eltype,mat_type> >
#define VM_EDMat VM_Unary_Mat<eltype,mat_type,VMExprOpDMat<eltype,mat_type> >
#define VM_EMatMul VM_Matrix_Expr<eltype,mat_type,mat_type2,VMExprOpMatMul<eltype,mat_type,mat_type2>,VMOpMatMulSize<mat_type,mat_type2> >

// Helper functions for multiplication
template<int blas_type,int blas_type2,int col_type2,class eltype,
         class mat_type, class mat_type2>
struct _vm_opmatmul {
  typedef const VM_EMatMul rettype;
  static inline VM_EMatMul op(const YA_BaseT &m, const YA_BaseT2 &t) {
    return VM_EMatMul(m.stv(),t.stv());
  }
};

#ifdef YA_BLAS
// t is not a column vector, but use blas
template<class eltype, class mat_type, class mat_type2>
struct _vm_opmatmul<1,1,0,eltype,mat_type,mat_type2> {
  typedef YA_MatT rettype;
  static inline YA_MatT op(const YA_BaseT &m, const YA_BaseT2 &t) {
    YA_MatT ans(m.rows(),t.cols());
    cblas_op5(static_cast<eltype>(1),m,t.stv(),
              static_cast<eltype>(0),ans);  
    return ans;
  }
};
// t is a column vector, use blas
template<class eltype, class mat_type, class mat_type2>
struct _vm_opmatmul<1,1,1,eltype,mat_type,mat_type2> {
  typedef YA_VecT rettype;
  static inline YA_VecT op(const YA_BaseT &m, const YA_BaseT2 &t) {
    YA_VecT ans(m.rows(),1);
    cblas_op5(static_cast<eltype>(1),m,t.stv(),
              static_cast<eltype>(0),ans);
    return ans;
  }
};
#endif

template<class eltype, class mat_type, class mat_type2>
inline typename _vm_opmatmul<YA_TRAIT::VME_BLAS_STRIDE,YA_TRAIT2::VME_BLAS_STRIDE,
                    mat_type2::VM_COL_VECTOR,eltype,mat_type,
                    mat_type2>::rettype 
vm_mult(const YA_BaseT &m, const YA_BaseT2 &t) {
   YA_DEBUG_ERROR2(m.cols()==t.rows(),
                   "Dimensions do not match for matrix multiplication",m,t);
   return _vm_opmatmul<YA_TRAIT::VME_USE_BLAS,YA_TRAIT2::VME_USE_BLAS,
                       YA_TRAIT2::VME_COL_VECTOR,eltype,mat_type,
                       mat_type2>::op(m,t);
}

}

using namespace yala;

// ------------------------- Expression Template Definitions
// ---- Max
template<class eltype, class mat_type>
inline const VM_EMax max(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()>0,"Cannot use max on empty matrix");
  return VM_EMax(m.stv());
}
// ---- Sum
template<class eltype, class mat_type>
inline const VM_ESum sum(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()>0,"Cannot use sum on empty matrix");
  return VM_ESum(m.stv());
}
// ---- Row Sums
template<class eltype, class mat_type>
inline const VM_ESumR row_sum(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()>0,"Cannot use sum on empty matrix");
  return VM_ESumR(m.stv());
}
// ---- Product
template<class eltype, class mat_type>
inline const VM_EProd prod(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()>0,"Cannot use prod on empty matrix");
  return VM_EProd(m.stv());
}
// ---- Size
template<class eltype, class mat_type>
inline const VM_Size_Mat<eltype,mat_type> size(const YA_BaseT &m) {
   return VM_Size_Mat<eltype,mat_type>(m.stv());
}
// ---- Compute the symmetric distance^2 matrix
/** Assumes that each row represents a point (each column a dimension) **/
template<class eltype, class mat_type>
inline const VM_EDMat2 dist2_mat(const YA_BaseT &m) {
  return VM_EDMat2(m.stv());
}
// ---- Compute the symmetric distance matrix
/** Assumes that each row represents a point (each column a dimension) **/
template<class eltype, class mat_type>
inline const VM_EDMat dist_mat(const YA_BaseT &m) {
  return VM_EDMat(m.stv());
}

// -------------- Matrix multiplication
#ifdef YA_BLAS_MULT
template<class eltype, class mat_type, class mat_type2>
inline YA_MatT operator* (const YA_BaseT &one, const YA_BaseT2 &two) {
   YA_DEBUG_ERROR2(one.cols()==two.rows(),
                   "Dimensions do not match for matrix multiplication",one,two);
  if (YA_TRAIT::VME_BLAS_STRIDE && YA_TRAIT2::VME_BLAS_STRIDE) {
    YA_MatT ans(one.rows(),two.cols());
    cblas_op5(static_cast<eltype>(1),one,two,
              static_cast<eltype>(0),ans);
    return ans;
  } else {
    return YA_MatT(VM_EMatMul(one.stv(),two.stv()));
  }
}
#else
template<class eltype, class mat_type, class mat_type2>
inline const VM_EMatMul operator* (const YA_BaseT &one, const YA_BaseT2 &two) {
  YA_DEBUG_ERROR2(one.cols()==two.rows(),
                  "Dimensions do not match for matrix multiplication",one,two);
  return VM_EMatMul(one.stv(),two.stv());
}
#endif

#endif
