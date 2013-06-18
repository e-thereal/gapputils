/***************************************************************************
                               vm_unary_expr.h
                             -------------------
                               W. Michael Brown

  Expression Templates for Vector/Matrix
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Feb 12 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_UNARY_EXPR_H
#define VM_UNARY_EXPR_H

YALANAMESPACE

// ------------------------- Expression Templates

// ------------------------- Operation Functors
template<class eltype> class VMExprOpMin {
  public: static inline eltype op(const eltype i) {return -i;}
};
template<class eltype> class VMExprOpNot {
  public: static inline eltype op(const eltype i) {if (i==0) return 1; return 0;}
};
template<class eltype> class VMExprOpBCom {
  public: static inline eltype op(const eltype i) {return ~i;}
};
template<class eltype> class VMExprOpSq {
  public: static inline eltype op(const eltype i) {return i*i;}
};
template<class eltype> class VMExprOpAbs {
  public: static inline eltype op(const eltype i) {return fabs(i);}
};
template<class eltype> class VMExprOpACos {
  public: static inline eltype op(const eltype i) {return acos(i);}
};
template<class eltype> class VMExprOpACosH {
  public: static inline eltype op(const eltype i) {return acosh(i);}
};
template<class eltype> class VMExprOpASin {
  public: static inline eltype op(const eltype i) {return asin(i);}
};
template<class eltype> class VMExprOpASinH {
  public: static inline eltype op(const eltype i) {return asinh(i);}
};
template<class eltype> class VMExprOpATan {
  public: static inline eltype op(const eltype i) {return atan(i);}
};
template<class eltype> class VMExprOpATanH {
  public: static inline eltype op(const eltype i) {return atanh(i);}
};
template<class eltype> class VMExprOpCeil {
  public: static inline eltype op(const eltype i) {return ceil(i);}
};
template<class eltype> class VMExprOpCos {
  public: static inline eltype op(const eltype i) {return cos(i);}
};
template<class eltype> class VMExprOpCosH {
  public: static inline eltype op(const eltype i) {return cosh(i);}
};
template<class eltype> class VMExprOpErf {
  public: static inline eltype op(const eltype i) {return erf(i);}
};
template<class eltype> class VMExprOpErfC {
  public: static inline eltype op(const eltype i) {return erfc(i);}
};
template<class eltype> class VMExprOpExp {
  public: static inline eltype op(const eltype i) {return exp(i);}
};
template<class eltype> class VMExprOpExp2 {
  public: static inline eltype op(const eltype i) {return exp2(i);}
};
template<class eltype> class VMExprOpExpM1 {
  public: static inline eltype op(const eltype i) {return expm1(i);}
};
template<class eltype> class VMExprOpFinite {
  public: static inline eltype op(const eltype i)
    {if (i==VM_INF || i==VM_MINUS_INF) return 0; return 1;}
};
template<class eltype> class VMExprOpFloor {
  public: static inline eltype op(const eltype i) {return floor(i);}
};
template<class eltype> class VMExprOpIsINF {
  public: static inline eltype op(const eltype i)
    {if (i==VM_INF || i==VM_MINUS_INF) return 1; return 0; }
};
template<class eltype> class VMExprOpIsNaN {
  public: static inline eltype op(const eltype i)
    {if (i==VM_NAN) return 1; return 0; }
};
template<class eltype> class VMExprOpLGM {
  public: static inline eltype op(const eltype i) { return lgamma(i); }
};
template<class eltype> class VMExprOpLog {
  public: static inline eltype op(const eltype i) { return log(i); }
};
template<class eltype> class VMExprOpLog10 {
  public: static inline eltype op(const eltype i) { return log10(i); }
};
template<class eltype> class VMExprOpLog1p {
  public: static inline eltype op(const eltype i) { return log1p(i); }
};
template<class eltype> class VMExprOpLog2 {
  public: static inline eltype op(const eltype i) { return log2(i); }
};
template<class eltype> class VMExprOpSin {
  public: static inline eltype op(const eltype i) { return sin(i); }
};
template<class eltype> class VMExprOpSinH {
  public: static inline eltype op(const eltype i) { return sinh(i); }
};
template<class eltype> class VMExprOpSqrt {
  public: static inline eltype op(const eltype i) { return sqrt(i); }
};
template<class eltype> class VMExprOpTan {
  public: static inline eltype op(const eltype i) { return tan(i); }
};
template<class eltype> class VMExprOpTanH {
  public: static inline eltype op(const eltype i) { return tanh(i); }
};
template<class eltype> class VMExprOpDotInv {
  public: static inline eltype op(const eltype i) 
    { return static_cast<eltype>(1)/i; }
};

// ------------------------- Expression Template Implementations
template<class eltype, class mat_type, class op, class stor1>
class VM_Unary_Expr : public YA_Common<eltype, VM_Unary> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Unary_Expr>::cref cref;

  VM_Unary_Expr(const mat_type& l) : one(l) {}

  inline cref at(const ya_sizet i) const
    { return op::op(one.at(i)); }
  inline cref at(const ya_sizet i)
    { return op::op(one.at(i)); }
  using YA_Common<eltype, VM_Unary>::operator();
  inline cref at2(const ya_sizet i,
                         const ya_sizet j) const {
    return op::op(one.at2(i,j));
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one.at2(i,j));
  }
  inline ya_sizet rows() const { return one.rows(); }
  inline ya_sizet cols() const { return one.cols(); }
  inline ya_sizet numel() const { return one.numel(); }

 protected:
  stor1 one;
};

#define VM_EMinus VM_Unary_Expr<eltype,mat_type,VMExprOpMin<eltype> >
#define VM_ESquare VM_Unary_Expr<eltype,mat_type,VMExprOpSq<eltype> >
#define VM_ENot VM_Unary_Expr<eltype,mat_type,VMExprOpNot<eltype> >
#define VM_EBCom VM_Unary_Expr<eltype,mat_type,VMExprOpBCom<eltype> >
#define VM_EAbs VM_Unary_Expr<eltype,mat_type,VMExprOpAbs<eltype> >
#define VM_EACos VM_Unary_Expr<eltype,mat_type,VMExprOpACos<eltype> >
#define VM_EACosH VM_Unary_Expr<eltype,mat_type,VMExprOpACosH<eltype> >
#define VM_EASin VM_Unary_Expr<eltype,mat_type,VMExprOpASin<eltype> >
#define VM_EASinH VM_Unary_Expr<eltype,mat_type,VMExprOpASinH<eltype> >
#define VM_EATan VM_Unary_Expr<eltype,mat_type,VMExprOpATan<eltype> >
#define VM_EATanH VM_Unary_Expr<eltype,mat_type,VMExprOpATanH<eltype> >
#define VM_ECeil VM_Unary_Expr<eltype,mat_type,VMExprOpCeil<eltype> >
#define VM_ECos VM_Unary_Expr<eltype,mat_type,VMExprOpCos<eltype> >
#define VM_ECosH VM_Unary_Expr<eltype,mat_type,VMExprOpCosH<eltype> >
#define VM_EErf VM_Unary_Expr<eltype,mat_type,VMExprOpErf<eltype> >
#define VM_EErfC VM_Unary_Expr<eltype,mat_type,VMExprOpErfC<eltype> >
#define VM_EExp VM_Unary_Expr<eltype,mat_type,VMExprOpExp<eltype> >
#define VM_EExp2 VM_Unary_Expr<eltype,mat_type,VMExprOpExp2<eltype> >
#define VM_EExpM1 VM_Unary_Expr<eltype,mat_type,VMExprOpExpM1<eltype> >
#define VM_EFinite VM_Unary_Expr<eltype,mat_type,VMExprOpFinite<eltype> >
#define VM_EFloor VM_Unary_Expr<eltype,mat_type,VMExprOpFloor<eltype> >
#define VM_EIsINF VM_Unary_Expr<eltype,mat_type,VMExprOpIsINF<eltype> >
#define VM_EIsNaN VM_Unary_Expr<eltype,mat_type,VMExprOpIsNaN<eltype> >
#define VM_ELGM VM_Unary_Expr<eltype,mat_type,VMExprOpLGM<eltype> >
#define VM_ELog VM_Unary_Expr<eltype,mat_type,VMExprOpLog<eltype> >
#define VM_ELog10 VM_Unary_Expr<eltype,mat_type,VMExprOpLog10<eltype> >
#define VM_ELog1P VM_Unary_Expr<eltype,mat_type,VMExprOpLog1p<eltype> >
#define VM_ELog2 VM_Unary_Expr<eltype,mat_type,VMExprOpLog2<eltype> >
#define VM_ESin VM_Unary_Expr<eltype,mat_type,VMExprOpSin<eltype> >
#define VM_ESinH VM_Unary_Expr<eltype,mat_type,VMExprOpSinH<eltype> >
#define VM_ESqrt VM_Unary_Expr<eltype,mat_type,VMExprOpSqrt<eltype> >
#define VM_ETan VM_Unary_Expr<eltype,mat_type,VMExprOpTan<eltype> >
#define VM_ETanH VM_Unary_Expr<eltype,mat_type,VMExprOpTanH<eltype> >
#define VM_EDotInv VM_Unary_Expr<eltype,mat_type,VMExprOpDotInv<eltype> >

}

using namespace yala;

// ------------------------- Expression Template Definitions
// ---- Unary Minus
template<class eltype, class mat_type>
inline const VM_EMinus operator-(const YA_BaseT &m)
  { return VM_EMinus(m.stv()); }
// ---- Unary Elementwise Square
template<class eltype, class mat_type>
inline const VM_ESquare square(const YA_BaseT &m)
  { return VM_ESquare(m.stv()); }
// ---- Transpose
template<class eltype, class mat_type>
inline const YA_TransT transpose(YA_BaseT &m)
  { return m.T(); }
// ---- Transpose const
template<class eltype, class mat_type>
inline const YA_Trans<eltype, const mat_type> transpose(const YA_BaseT &m)
  { return m.T(); }
// ---- Diag
template<class eltype, class mat_type>
inline const VM_Diag diag(YA_BaseT &m)
  { return VM_Diag(&(m.stv())); }
// ---- Diag const
template<class eltype, class mat_type>
inline const VecMatDiag<eltype, const mat_type> diag(const YA_BaseT &m)
  { return VecMatDiag<eltype, const mat_type>(&(m.stv())); }
// ---- Reshape
template<class eltype, class mat_type>
inline const VM_Reshape reshape(YA_BaseT &m, const ya_sizet rows,
                                const ya_sizet cols)
  { return VM_Reshape(&(m.stv()),rows,cols); }
// ---- Reshape const
template<class eltype, class mat_type>
inline const VecMatReshape<eltype, const mat_type> reshape(const YA_BaseT &m,
                                                          const ya_sizet rows,
                                                          const ya_sizet cols)
  { return VecMatReshape<eltype, const mat_type>(&(m.stv()),
                                                   rows,cols); }
// ---- RepMat
template<class eltype, class mat_type>
inline const VM_RepMat repmat(YA_BaseT &m, const ya_sizet rows,
                              const ya_sizet cols)
  { return VM_RepMat(&(m.stv()),rows*m.rows(),cols*m.cols()); }
// ---- RepMat const
template<class eltype, class mat_type>
inline const VecMatRepMat<eltype, const mat_type> repmat(const YA_BaseT &m,
                                                           const ya_sizet rows,
                                                           const ya_sizet cols)
  { return VecMatRepMat<eltype, const mat_type>(&(m.stv()),
                                                  rows*m.rows(),
                                                  cols*m.cols());}
// ---- RowRep
template<class eltype, class mat_type>
inline const VM_RowRep rowrep(YA_BaseT &m, const ya_sizet rows)
  { return VM_RowRep(&(m.stv()),rows); }
// ---- RowRep const
template<class eltype, class mat_type>
inline const VecMatRowRep<eltype, const mat_type> rowrep(const YA_BaseT &m,
                                                           const ya_sizet rows)
  { return VecMatRowRep<eltype, const mat_type>(&(m.stv()),rows);}
// ---- ColRep
template<class eltype, class mat_type>
inline const VM_ColRep colrep(YA_BaseT &m, const ya_sizet cols)
  { return VM_ColRep(&(m.stv()),cols); }
// ---- ColRep const
template<class eltype, class mat_type>
inline const VecMatColRep<eltype, const mat_type> colrep(const YA_BaseT &m,
                                                           const ya_sizet cols)
  { return VecMatColRep<eltype, const mat_type>(&(m.stv()),cols);}
// ---- !
template<class eltype, class mat_type>
inline const VM_ENot operator!(const YA_BaseT &m)
  { return VM_ENot(m.stv()); }
// ---- ~
template<class eltype, class mat_type>
inline const VM_EBCom operator~(const YA_BaseT &m)
  { return VM_EBCom(m.stv()); }
// ---- abs
template<class eltype, class mat_type>
inline const VM_EAbs abs(const YA_BaseT &m)
  { return VM_EAbs(m.stv()); }
// ---- acos
template<class eltype, class mat_type>
inline const VM_EACos acos(const YA_BaseT &m)
  { return VM_EACos(m.stv()); }
// ---- acosh
template<class eltype, class mat_type>
inline const VM_EACosH acosh(const YA_BaseT &m)
  { return VM_EACosH(m.stv()); }
// ---- asin
template<class eltype, class mat_type>
inline const VM_EASin asin(const YA_BaseT &m)
  { return VM_EASin(m.stv()); }
// ---- asinh
template<class eltype, class mat_type>
inline const VM_EASinH asinh(const YA_BaseT &m)
  { return VM_EASinH(m.stv()); }
// ---- atan
template<class eltype, class mat_type>
inline const VM_EATan atan(const YA_BaseT &m)
  { return VM_EATan(m.stv()); }
// ---- atanh
template<class eltype, class mat_type>
inline const VM_EATanH atanh(const YA_BaseT &m)
  { return VM_EATanH(m.stv()); }
// ---- ceil
template<class eltype, class mat_type>
inline const VM_ECeil ceil(const YA_BaseT &m)
  { return VM_ECeil(m.stv()); }
// ---- cos
template<class eltype, class mat_type>
inline const VM_ECos cos(const YA_BaseT &m)
  { return VM_ECos(m.stv()); }
// ---- cosh
template<class eltype, class mat_type>
inline const VM_ECosH cosh(const YA_BaseT &m)
  { return VM_ECosH(m.stv()); }
// ---- erf
template<class eltype, class mat_type>
inline const VM_EErf erf(const YA_BaseT &m)
  { return VM_EErf(m.stv()); }
// ---- erfc
template<class eltype, class mat_type>
inline const VM_EErfC erfc(const YA_BaseT &m)
  { return VM_EErfC(m.stv()); }
// ---- exp
#ifdef VM_INTEL_VML
template<class eltype, class mat_type>
inline YA_MatT exp(const YA_BaseT &m) {
  YA_MatT answer(m.rows(),m.cols());
  if (YA_TRAIT::VME_BLAS)
    _vml_exp(m.numel(),m.begin(),answer.begin());
  else {
    YA_MatT temp(m);
    _vml_exp<const eltype *,eltype *>(m.numel(),temp.begin(),
                                          answer.begin());
  }
  return answer;
}
#else
template<class eltype, class mat_type>
inline const VM_EExp exp(const YA_BaseT &m)
  { return VM_EExp(m.stv()); }
#endif
// ---- exp2
template<class eltype, class mat_type>
inline const VM_EExp2 exp2(const YA_BaseT &m)
  { return VM_EExp2(m.stv()); }
// ---- expm1
template<class eltype, class mat_type>
inline const VM_EExpM1 expm1(const YA_BaseT &m)
  { return VM_EExpM1(m.stv()); }
// ---- finite
template<class eltype, class mat_type>
inline const VM_EFinite finite(const YA_BaseT &m)
  { return VM_EFinite(m.stv()); }
// ---- floor
template<class eltype, class mat_type>
inline const VM_EFloor floor(const YA_BaseT &m)
  { return VM_EFloor(m.stv()); }
// ---- isinf
template<class eltype, class mat_type>
inline const VM_EIsINF is_inf(const YA_BaseT &m)
  { return VM_EIsINF(m.stv()); }
// ---- isnan
template<class eltype, class mat_type>
inline const VM_EIsNaN is_nan(const YA_BaseT &m)
  { return VM_EIsNaN(m.stv()); }
// ---- lgamma
template<class eltype, class mat_type>
inline const VM_ELGM lgamma(const YA_BaseT &m)
  { return VM_ELGM(m.stv()); }
// ---- log
#ifdef VM_INTEL_VML
template<class eltype, class mat_type>
inline YA_MatT log(const YA_BaseT &m) {
  YA_MatT answer(m.rows(),m.cols());
  if (YA_TRAIT::VME_BLAS)
    _vml_log(m.numel(),m.begin(),answer.begin());
  else {
    YA_MatT temp(m);
    _vml_log<const eltype *,eltype *>(m.numel(),temp.begin(),
                                          answer.begin());
  }
  return answer;
}
#else
template<class eltype, class mat_type>
inline const VM_ELog log(const YA_BaseT &m)
  { return VM_ELog(m.stv()); }
#endif
// ---- log10
template<class eltype, class mat_type>
inline const VM_ELog10 log10(const YA_BaseT &m)
  { return VM_ELog10(m.stv()); }
// ---- log1p
template<class eltype, class mat_type>
inline const VM_ELog1P log1p(const YA_BaseT &m)
  { return VM_ELog1P(m.stv()); }
// ---- log2
template<class eltype, class mat_type>
inline const VM_ELog2 log_2(const YA_BaseT &m)
  { return VM_ELog2(m.stv()); }
// ---- sin
template<class eltype, class mat_type>
inline const VM_ESin sin(const YA_BaseT &m)
  { return VM_ESin(m.stv()); }
// ---- sinh
template<class eltype, class mat_type>
inline const VM_ESinH sinh(const YA_BaseT &m)
  { return VM_ESinH(m.stv()); }
// ---- sqrt
template<class eltype, class mat_type>
inline const VM_ESqrt sqrt(const YA_BaseT &m)
  { return VM_ESqrt(m.stv()); }
// ---- tan
template<class eltype, class mat_type>
inline const VM_ETan tan(const YA_BaseT &m)
  { return VM_ETan(m.stv()); }
// ---- tanh
template<class eltype, class mat_type>
inline const VM_ETanH tanh(const YA_BaseT &m)
  { return VM_ETanH(m.stv()); }

// ---- dotinv
#ifdef VM_INTEL_VML
template<class eltype, class mat_type>
inline YA_MatT dotinv(const YA_BaseT &m) {
  YA_MatT answer(m.rows(),m.cols());
  if (YA_TRAIT::VME_BLAS)
    _vml_inv(m.numel(),m.begin(),answer.begin());
  else {
    YA_MatT temp(m);
    _vml_inv<const eltype *,eltype *>(m.numel(),temp.begin(),
                                          answer.begin());
  }
  return answer;
}
#else
template<class eltype, class mat_type>
inline const VM_EDotInv dotinv(const YA_BaseT &m)
  { return VM_EDotInv(m.stv()); }
#endif  

#endif
