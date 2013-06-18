/***************************************************************************
                                 ya_unary.cpp
                             -------------------
                               W. Michael Brown

  Unary operations not using template expressions
 
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

#ifndef YA_UNARY_S_H
#define YA_UNARY_S_H

using namespace yala;

// -- Loop helpers
template <class ref> struct _ya_ufor_inc {
  static inline void op(ref r) { r++; }
};
template <class ref> struct _ya_ufor_dec {
  static inline void op(ref r) { r--; }
};

// ------- ++ 

template<class eltype, class mat_type>
void operator++(YA_BaseT &mat,int) {
  ya_no_order_loop<_ya_ufor_inc<typename mat_type::ref> >::lp(mat);
}

template<class eltype, class mat_type>
void operator++(const YA_WrapperT &mat,int) {
  ya_no_order_loop<_ya_ufor_inc<typename mat_type::ref> >::lp(mat);
}

template<class eltype, class mat_type>
YA_BaseT& operator++(YA_BaseT &mat) {
  ya_no_order_loop<_ya_ufor_inc<typename mat_type::ref> >::lp(mat);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator++(const YA_WrapperT &mat) {
  ya_no_order_loop<_ya_ufor_inc<typename mat_type::ref> >::lp(mat);
  return mat;
}

// ------- -- 

template<class eltype, class mat_type>
void operator--(YA_BaseT &mat,int) {
  ya_no_order_loop<_ya_ufor_dec<typename mat_type::ref> >::lp(mat);
}

template<class eltype, class mat_type>
void operator--(const YA_WrapperT &mat,int) {
  ya_no_order_loop<_ya_ufor_dec<typename mat_type::ref> >::lp(mat);
}

template<class eltype, class mat_type>
YA_BaseT& operator--(YA_BaseT &mat) {
  ya_no_order_loop<_ya_ufor_dec<typename mat_type::ref> >::lp(mat);
  return mat;
}

template<class eltype, class mat_type>
const YA_WrapperT& operator--(const YA_WrapperT &mat) {
  ya_no_order_loop<_ya_ufor_dec<typename mat_type::ref> >::lp(mat);
  return mat;
}

#endif
