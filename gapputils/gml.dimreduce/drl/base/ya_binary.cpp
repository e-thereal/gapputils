/***************************************************************************
                                ya_binary.cpp
                             -------------------
                               W. Michael Brown

  Binary operations not using template expressions
 
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_BINARY_S_H
#define YA_BINARY_S_H

YALANAMESPACE

// ---------------- Loop Helpers
template <class ref, class eltype> struct _ya_ufor_pluse {
  static inline void op(ref r, eltype v) { r+=v; } 
};
template <class ref, class eltype> struct _ya_ufor_mine {
  static inline void op(ref r, eltype v) { r-=v; } 
};
template <class ref, class eltype> struct _ya_ufor_timese {
  static inline void op(ref r, eltype v) { r*=v; } 
};
template <class ref, class eltype> struct _ya_ufor_dive {
  static inline void op(ref r, eltype v) { r/=v; } 
};

}

using namespace yala;

template<class eltype, class mat_type>
YA_BaseT& operator+=(YA_BaseT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_pluse<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator+=(const YA_WrapperT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_pluse<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

template<class eltype, class mat_type>
YA_BaseT& operator-=(YA_BaseT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_mine<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator-=(const YA_WrapperT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_mine<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

// Helper function for *=
template<class eltype, int blas_type>
struct _ya_times_equals {
  template <class ya_type>
  inline static void op(ya_type &mat, const eltype num) {
    ya_no_order_loop<_ya_ufor_timese<typename ya_type::ref,const eltype> >::lp
                                                           (mat.stv(),num);
  }
};

#ifdef YA_BLAS
// Helper function for *=
template<class eltype>
struct _ya_times_equals<eltype,1> {
  template <class ya_type>
  inline static void op(ya_type &one, const eltype num) {
    cblas_scal(num,one);
  }
};
#endif

template<class eltype, class mat_type>
YA_BaseT& operator*=(YA_BaseT &mat, const eltype num) {
  _ya_times_equals<eltype,YA_TRAIT::VME_USE_BLAS>::op(mat,num);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator*=(const YA_WrapperT &mat, const eltype num) {
  _ya_times_equals<eltype,YA_TRAIT::VME_USE_BLAS>::op(mat,num);
  return mat;
}

template<class eltype, class mat_type>
YA_BaseT& operator/=(YA_BaseT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_dive<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator/=(const YA_WrapperT &mat, const eltype num) {
  ya_no_order_loop<_ya_ufor_dive<typename mat_type::ref,const eltype> >::lp
                                                         (mat.stv(),num);
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator+=(YA_BaseT &mat, const ya_type2 &two) {
  YA_DEBUG_ERROR2(two.rows()==mat.rows() && two.cols()==mat.cols(),
                  "Operator += on different sized matrices",mat,two);
  ya_binary_loop<_ya_ufor_pluse<typename mat_type::ref,
                            typename ya_type2::cref> >::nz(mat.stv(),two.stv());
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator+=(const YA_WrapperT &mat, const ya_type2 &two) {
  YA_DEBUG_ERROR2(two.rows()==mat.rows() && two.cols()==mat.cols(),
                  "Operator += on different sized matrices",mat,two);
  ya_binary_loop<_ya_ufor_pluse<typename mat_type::ref,
                            typename ya_type2::cref> >::nz(mat.stv(),two.stv());
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator-=(YA_BaseT &mat, const ya_type2 &two) {
  YA_DEBUG_ERROR2(two.rows()==mat.rows() && two.cols()==mat.cols(),
                  "Operator -= on different sized matrices",mat,two);
  ya_binary_loop<_ya_ufor_mine<typename mat_type::ref,
                            typename ya_type2::cref> >::nz(mat.stv(),two.stv());
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator-=(const YA_WrapperT &mat, const ya_type2 &two) {
  YA_DEBUG_ERROR2(two.rows()==mat.rows() && two.cols()==mat.cols(),
                  "Operator -= on different sized matrices",mat,two);
  ya_binary_loop<_ya_ufor_mine<typename mat_type::ref,
                            typename ya_type2::cref> >::nz(mat.stv(),two.stv());
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator*=(YA_BaseT &mat, const ya_type2 &two) {
  mat=copy(mat)*two;
  return mat;
}

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator*=(const YA_WrapperT &mat, const ya_type2 &two) {
  mat=copy(mat)*two;
  return mat;
}

#endif
