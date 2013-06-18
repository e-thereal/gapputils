/***************************************************************************
                              vm_binary_expr.h
                             -------------------
                               W. Michael Brown

  Binary Expression Templates for Vector/Matrix

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

#ifndef VM_BINARY_EXPR_H
#define VM_BINARY_EXPR_H

YALANAMESPACE

// ------------------------- Expression Templates

// ------------------------- Operation Functors
template<class eltype> class VMExprOpAdd { 
  public: static inline eltype op(const eltype i, const eltype j) {return i+j;}
};
template<class eltype> class VMExprOpSub {
  public: static inline eltype op(const eltype i, const eltype j) {return i-j;}
};
template<class eltype> class VMExprOpSubR {
  public: static inline eltype op(const eltype i, const eltype j) {return j-i;}
};
template<class eltype> class VMExprOpMul {
  public: static inline eltype op(const eltype i, const eltype j) {return i*j;}
};
template<class eltype> class VMExprOpDiv {
  public: static inline eltype op(const eltype i, const eltype j) {return i/j;}
};
template<class eltype> class VMExprOpDivR {
  public: static inline eltype op(const eltype i, const eltype j) {return j/i;}
};
template<class eltype> class VMExprOpEQ {
  public: static inline int op(const eltype i, const eltype j) {
    if (i==j) return 1; return 0;}
};
template<class eltype> class VMExprOpNEQ {
  public: static inline int op(const eltype i, const eltype j) {
    if (i==j) return 0; return 1;}
};
template<class eltype> class VMExprOpGT {
  public: static inline int op(const eltype i, const eltype j) {
    if (i>j) return 1; return 0;}
};
template<class eltype> class VMExprOpGTE {
  public: static inline int op(const eltype i, const eltype j) {
    if (i>=j) return 1; return 0;}
};
template<class eltype> class VMExprOpLT {
  public: static inline int op(const eltype i, const eltype j) {
    if (i<j) return 1; return 0;}
};
template<class eltype> class VMExprOpLTE {
  public: static inline int op(const eltype i, const eltype j) {
    if (i<=j) return 1; return 0;}
};
template<class eltype> class VMExprOpATan2 {
  public: static inline eltype op(const eltype i, const eltype j)
   { return atan2(i,j); }
};
template<class eltype> class VMExprOpATan2R {
  public: static inline eltype op(const eltype i, const eltype j)
   { return atan2(j,i); }
};
template<class eltype> class VMExprOpAnd {
  public: static inline eltype op(const eltype i, const eltype j)
    { if (i==0) return 0; if (j==0) return 0; return 1; }
};
template<class eltype> class VMExprOpXOr {
  public: static inline eltype op(const eltype i, const eltype j)
    { if (i==0 && j!=0) return 1; if (j==0 && i!=0) return 1; return 0; }
};
template<class eltype> class VMExprOpOr {
  public: static inline eltype op(const eltype i, const eltype j)
    { if (i!=0 || j!=0) return 1; return 0; }
};
template<class eltype> class VMExprOpMod {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i%j; }
};
template<class eltype> class VMExprOpModR {
  public: static inline eltype op(const eltype i, const eltype j)
    { return j%i; }
};
template<class eltype> class VMExprOpShL {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i << j; }
};
template<class eltype> class VMExprOpShR {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i >> j; }
};
template<class eltype> class VMExprOpBAnd {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i & j; }
};
template<class eltype> class VMExprOpBXOr {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i ^ j; }
};
template<class eltype> class VMExprOpBOr {
  public: static inline eltype op(const eltype i, const eltype j)
    { return i | j; }
};
template<class eltype> class VMExprOpPow {
  public: static inline eltype op(const eltype i, const eltype j) 
    { return pow(i,j); }
};
template<class eltype> class VMExprOpPowR {
  public: static inline eltype op(const eltype i, const eltype j)
    { return pow(j,i); }
};


// ------------------------- Expression Template Implementations
template<class eltype, class contype, class mat_type, class op, class stor1>
class VM_Binary_ExprS : public YA_Common<eltype, VM_BinaryS> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Binary_ExprS>::cref cref;

  VM_Binary_ExprS(const mat_type& l, const contype r) : one(l), two(r) {}

  inline cref at(const ya_sizet i) const
    { return op::op(one.at(i),two); }
  inline cref at(const ya_sizet i)
    { return op::op(one.at(i),two); }
  using YA_Common<eltype, VM_BinaryS>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one.at2(i,j),two);
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one.at2(i,j),two);
  }
  inline ya_sizet rows() const { return one.rows(); }
  inline ya_sizet cols() const { return one.cols(); }
  inline ya_sizet numel() const { return one.numel(); }

 protected:
  stor1 one;
  contype two;
};

template<class eltype, class mat_type, class mat_type2, class op, class stor1, class stor2>
class VM_Binary_Expr : public YA_Common<eltype, VM_Binary> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Binary_Expr>::cref cref;

  VM_Binary_Expr(const mat_type &l, const mat_type2 &r) : one(l), two(r) {
    #ifdef YA_DEBUG
    YA_DEBUG_ERROR2( (l.numel()==r.numel()) ||
                     (l.rows()==r.rows() && l.cols()==r.cols()),
                    "Binary expression requires same sized matrices",l,r);
    #endif
  }

  inline cref at(const ya_sizet i) const
    { return op::op(one.at(i),two.at(i)); }
  inline cref at(const ya_sizet i)
    { return op::op(one.at(i),two.at(i)); }
  using YA_Common<eltype, VM_Binary>::operator();
  inline cref at2(const ya_sizet i, const ya_sizet j) const {
    return op::op(one.at2(i,j),two.at2(i,j));
  }
  inline cref at2(const ya_sizet i, const ya_sizet j) {
    return op::op(one.at2(i,j),two.at2(i,j));
  }
  inline ya_sizet rows() const { return one.rows(); }
  inline ya_sizet cols() const { return one.cols(); }
  inline ya_sizet numel() const { return one.rows()*one.cols(); }

 protected:
  stor1 one;
  stor2 two;
};

#define VM_EAddS VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpAdd<eltype> >
#define VM_EAdd VM_Binary_Expr<eltype,mat_type,mat_type2,VMExprOpAdd<eltype> >
#define VM_ESubS VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpSub<eltype> >
#define VM_ESubR VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpSubR<eltype> >
#define VM_ESub VM_Binary_Expr<eltype,mat_type,mat_type2,VMExprOpSub<eltype> >
#define VM_EMulS VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpMul<eltype> >
#define VM_EDMul VM_Binary_Expr<eltype,mat_type,mat_type2,VMExprOpMul<eltype> >
#define VM_EDivS VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpDiv<eltype> >
#define VM_EDivR VM_Binary_ExprS<eltype,const eltype,mat_type,VMExprOpDivR<eltype> >
#define VM_EDDiv VM_Binary_Expr<eltype,mat_type,mat_type2,VMExprOpDiv<eltype> >
#define VM_EQS VM_Binary_ExprS<int,eltype,mat_type,VMExprOpEQ<eltype> >
#define VM_EQ VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpEQ<eltype> >
#define VM_NEQS VM_Binary_ExprS<int,eltype,mat_type,VMExprOpNEQ<eltype> >
#define VM_NEQ VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpNEQ<eltype> >
#define VM_GTS VM_Binary_ExprS<int,eltype,mat_type,VMExprOpGT<eltype> >
#define VM_GT VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpGT<eltype> >
#define VM_LTS VM_Binary_ExprS<int,eltype,mat_type,VMExprOpLT<eltype> >
#define VM_LT VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpLT<eltype> >
#define VM_GTES VM_Binary_ExprS<int,eltype,mat_type,VMExprOpGTE<eltype> >
#define VM_GTE VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpGTE<eltype> >
#define VM_LTES VM_Binary_ExprS<int,eltype,mat_type,VMExprOpLTE<eltype> >
#define VM_LTE VM_Binary_Expr<int,mat_type,mat_type2,VMExprOpLTE<eltype> >
#define VM_ATan2S VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpATan2<eltype> >
#define VM_ATan2SR VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpATan2R<eltype> >
#define VM_ATan2 VM_Binary_Expr<eltype,mat_type,mat_type2,VMExprOpATan2<eltype> >
#define VM_AndS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpAnd<eltype> >
#define VM_And VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpAnd<eltype> >
#define VM_XOrS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpXOr<eltype> >
#define VM_XOr VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpXOr<eltype> >
#define VM_OrS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpOr<eltype> >
#define VM_Or VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpOr<eltype> >
#define VM_ModS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpMod<eltype> >
#define VM_ModSR VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpModR<eltype> >
#define VM_Mod VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpMod<eltype> >
#define VM_ShLS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpShL<eltype> >
#define VM_ShL VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpShL<eltype> >
#define VM_ShRS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpShR<eltype> >
#define VM_ShR VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpShR<eltype> >
#define VM_BAndS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpBAnd<eltype> >
#define VM_BAnd VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpBAnd<eltype> >
#define VM_BXOrS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpBXOr<eltype> >
#define VM_BXOr VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpBXOr<eltype> >
#define VM_BOrS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpBOr<eltype> >
#define VM_BOr VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpBOr<eltype> >
#define VM_PowS VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpPow<eltype> >
#define VM_PowSR VM_Binary_ExprS<eltype,const eltype,mat_type,class VMExprOpPowR<eltype> >
#define VM_Pow VM_Binary_Expr<eltype,mat_type,mat_type2,class VMExprOpPow<eltype> >

}

using namespace yala;

// ------------------------- Expression Template Definitions
// ---- Add
template<class eltype, class mat_type>
inline const VM_EAddS operator+(const eltype s, const YA_BaseT &m)
  { return VM_EAddS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_EAddS operator+(const YA_BaseT &m, const eltype s)
  { return VM_EAddS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_EAdd operator+(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_EAdd(m.stv(),s.stv()); }

// ---- Subtract
template<class eltype, class mat_type>
inline const VM_ESubR operator-(const eltype s, const YA_BaseT &m)
  { return VM_ESubR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_ESubS operator-(const YA_BaseT &m, const eltype s)
  { return VM_ESubS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_ESub operator-(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_ESub(m.stv(),s.stv()); }

// ---- Element-wise multiply
template<class eltype, class mat_type>
inline const VM_EMulS operator*(const eltype s, const YA_BaseT &m)
  { return VM_EMulS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_EMulS operator*(const YA_BaseT &m, const eltype s)
  { return VM_EMulS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_EDMul dot_star(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_EDMul(m.stv(),s.stv()); }

// ---- Element-wise divide
template<class eltype, class mat_type>
inline const VM_EDivR operator/(const eltype s, const YA_BaseT &m)
  { return VM_EDivR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_EDivS operator/(const YA_BaseT &m, const eltype s)
  { return VM_EDivS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_EDDiv dot_div(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_EDDiv(m.stv(),s.stv()); }

// ---- Element-wise equals
template<class eltype, class mat_type>
inline const VM_EQS operator==(const eltype s, const YA_BaseT &m)
  { return VM_EQS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_EQS operator==(const YA_BaseT &m, const eltype s)
  { return VM_EQS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_EQ operator==(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_EQ(m.stv(),s.stv()); }

// ---- Element-wise not equals
template<class eltype, class mat_type>
inline const VM_NEQS operator!=(const eltype s, const YA_BaseT &m)
  { return VM_NEQS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_NEQS operator!=(const YA_BaseT &m, const eltype s)
  { return VM_NEQS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_NEQ operator!=(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_NEQ(m.stv(),s.stv()); }

// ---- Element-wise >, <
template<class eltype, class mat_type>
inline const VM_LTS operator>(const eltype s, const YA_BaseT &m)
  { return VM_LTS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_GTS operator>(const YA_BaseT &m, const eltype s)
  { return VM_GTS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_GT operator>(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_GT(m.stv(),s.stv()); }
// ---- Element-wise <
template<class eltype, class mat_type>
inline const VM_GTS operator<(const eltype s, const YA_BaseT &m)
  { return VM_GTS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_LTS operator<(const YA_BaseT &m, const eltype s)
  { return VM_LTS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_LT operator<(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_LT(m.stv(),s.stv()); }


// ---- Element-wise >=, <=
template<class eltype, class mat_type>
inline const VM_LTES operator>=(const eltype s, const YA_BaseT &m)
  { return VM_LTES(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_GTES operator>=(const YA_BaseT &m, const eltype s)
  { return VM_GTES(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_GTE operator>=(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_GTE(m.stv(),s.stv()); }
// ---- Element-wise <=
template<class eltype, class mat_type>
inline const VM_GTES operator<=(const eltype s, const YA_BaseT &m)
  { return VM_GTES(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_LTES operator<=(const YA_BaseT &m, const eltype s)
  { return VM_LTES(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_LTE operator<=(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_LTE(m.stv(),s.stv()); }

// ---- Element-wise atan2
template<class eltype, class mat_type>
inline const VM_ATan2SR atan2(const eltype s, const YA_BaseT &m)
  { return VM_ATan2SR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_ATan2S atan2(const YA_BaseT &m, const eltype s)
  { return VM_ATan2S(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_ATan2 atan2(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_ATan2(m.stv(),s.stv()); }

// ---- Element-wise and
template<class eltype, class mat_type>
inline const VM_AndS operator&&(const eltype s, const YA_BaseT &m)
  { return VM_AndS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_AndS operator&&(const YA_BaseT &m, const eltype s)
  { return VM_AndS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_And operator&&(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_And(m.stv(),s.stv()); }

// ---- Element-wise xor
template<class eltype, class mat_type>
inline const VM_XOrS x_or(const eltype s, const YA_BaseT &m)
  { return VM_XOrS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_XOrS x_or(const YA_BaseT &m, const eltype s)
  { return VM_XOrS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_XOr x_or(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_XOr(m.stv(),s.stv()); }

// ---- Element-wise or
template<class eltype, class mat_type>
inline const VM_OrS operator||(const eltype s, const YA_BaseT &m)
  { return VM_OrS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_OrS operator||(const YA_BaseT &m, const eltype s)
  { return VM_OrS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_Or operator||(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_Or(m.stv(),s.stv()); }

// ---- Element-wise Modulus
template<class eltype, class mat_type>
inline const VM_ModSR operator%(const eltype s, const YA_BaseT &m)
  { return VM_ModSR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_ModS operator%(const YA_BaseT &m, const eltype s)
  { return VM_ModS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_Mod operator%(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_Mod(m.stv(),s.stv()); }

// ---- Element-wise Left Shift
template<class eltype, class mat_type>
inline const VM_ShLS operator<<(const eltype s, const YA_BaseT &m)
  { return VM_ShLS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_ShLS operator<<(const YA_BaseT &m, const eltype s)
  { return VM_ShLS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_ShL operator<<(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_ShL(m.stv(),s.stv()); }

// ---- Element-wise Left Shift
template<class eltype, class mat_type>
inline const VM_ShRS operator>>(const eltype s, const YA_BaseT &m)
  { return VM_ShRS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_ShRS operator>>(const YA_BaseT &m, const eltype s)
  { return VM_ShRS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_ShR operator>>(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_ShR(m.stv(),s.stv()); }

// ---- Element-wise Bitwise And
template<class eltype, class mat_type>
inline const VM_BAndS operator&(const eltype s, const YA_BaseT &m)
  { return VM_BAndS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_BAndS operator&(const YA_BaseT &m, const eltype s)
  { return VM_BAndS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_BAnd operator&(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_BAnd(m.stv(),s.stv()); }

// ---- Element-wise Bitwise XOr
template<class eltype, class mat_type>
inline const VM_BXOrS b_xor(const eltype s, const YA_BaseT &m)
  { return VM_BXOrS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_BXOrS b_xor(const YA_BaseT &m, const eltype s)
  { return VM_BXOrS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_BXOr b_xor(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_BXOr(m.stv(),s.stv()); }

// ---- Element-wise Bitwise Or
template<class eltype, class mat_type>
inline const VM_BOrS operator|(const eltype s, const YA_BaseT &m)
  { return VM_BOrS(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_BOrS operator|(const YA_BaseT &m, const eltype s)
  { return VM_BOrS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_BOr operator|(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_BOr(m.stv(),s.stv()); }

// ---- Element-wise Power
template<class eltype, class mat_type>
inline const VM_PowSR pow(const eltype s, const YA_BaseT &m)
  { return VM_PowSR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_PowS pow(const YA_BaseT &m, const eltype s)
  { return VM_PowS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_Pow pow(const YA_BaseT &m, const YA_BaseT2 &s) 
  { return VM_Pow(m.stv(),s.stv()); }
template<class eltype, class mat_type>
inline const VM_PowSR operator^(const eltype s, const YA_BaseT &m)
  { return VM_PowSR(m.stv(),s); }
template<class eltype, class mat_type>
inline const VM_PowS operator^(const YA_BaseT &m, const eltype s)
  { return VM_PowS(m.stv(),s); }
template<class eltype, class mat_type, class mat_type2>
inline const VM_Pow operator^(const YA_BaseT &m, const YA_BaseT2 &s)
  { return VM_Pow(m.stv(),s.stv()); }

// ---- Concatenation (Column)
template<class eltype, class mat_type, class mat_type2>
inline const VM_Concat concat(YA_BaseT &m, mat_type2 &m2)
  { return VM_Concat(&(m.stv()),&m2.stv()); }
// ---- Concatenation (Column) const
template<class eltype, class mat_type, class mat_type2>
inline const VecMatConcat<eltype, const mat_type, const mat_type2> 
concat(const YA_BaseT &m,const YA_BaseT2 &m2)
  { return VecMatConcat<eltype, const mat_type, const mat_type2> 
             (&(m.stv()),&m2.stv()); }

// ---- Concatenation (Row)
template<class eltype, class mat_type, class mat_type2>
inline const VM_ConcatR concat(YA_BaseT &m, const std::string c, mat_type2 &m2)
  { return VM_ConcatR(&(m.stv()),&m2.stv()); }
// ---- Concatenation (Row) const
template<class eltype, class mat_type, class mat_type2>
inline const VecMatConcatR<eltype, const mat_type, const mat_type2> 
concat(const YA_BaseT &m,const std::string colon, const YA_BaseT2 &m2)
  { return VecMatConcatR<eltype, const mat_type, const mat_type2> 
             (&(m.stv()),&m2.stv()); }

#endif
