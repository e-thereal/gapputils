/***************************************************************************
                               ya_zero_matrix.h
                             -------------------
                               W. Michael Brown

  Routines for zeroing matrices and setting matrices to a constant value
                               
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

#ifndef YA_ZERO_MATRIX_H
#define YA_ZERO_MATRIX_H

YALANAMESPACE

template<class eltype,int,int> struct _ya_zero_matrix1;
template<int> struct _ya_zero_matrix2;

template<class eltype, class mat_type>
inline void ya_zero_matrix(YA_BaseT &target) {
  _ya_zero_matrix1<eltype,YA_TRAIT::VME_MEMORY,
                   ya_direct_set<eltype>::ans>::mc(target.stv());
}

template<class eltype, class mat_type>
inline void ya_zero_matrix(const YA_WrapperT &target) {
  _ya_zero_matrix1<eltype,YA_TRAIT::VME_MEMORY,
                   ya_direct_set<eltype>::ans>::mc(target.stv());
}

template<class eltype, class mat_type>
inline void ya_copy_scalar(YA_BaseT &target, const eltype value) {
  _ya_zero_matrix1<eltype,YA_TRAIT::VME_MEMORY,
                   ya_direct_setv<eltype>::ans>::mc(target.stv(),value);
}

template<class eltype, class mat_type>
inline void ya_copy_scalar(const YA_WrapperT &target, const eltype value) {
  _ya_zero_matrix1<eltype,YA_TRAIT::VME_MEMORY,
                   ya_direct_setv<eltype>::ans>::mc(target.stv(),value);
}

// ------------------------- Direct Memory Zero?
template<class eltype> struct _ya_zero_matrix1<eltype,1,1> {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=0 1: memset) " << endl;
    #endif
    memset(target.c_ptr(),0,sizeof(eltype)*target.numel());
  }
  template <class ya_type>
  static inline void mc(ya_type &target,const eltype value) {
    memset(target.c_ptr(),value,sizeof(eltype)*target.numel());
  }
};
template<class eltype> struct _ya_zero_matrix1<eltype,2,1> {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=0 2: memset) " << endl;
    #endif
    memset(target.c_ptr(),0,sizeof(eltype)*target.numel());
  }
  template <class ya_type>
  static inline void mc(ya_type &target,const eltype value) {
    memset(target.c_ptr(),value,sizeof(eltype)*target.numel());
  }
};
template<class eltype> struct _ya_zero_matrix1<eltype,3,1> {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=0 3: STL Vec) " << endl;
    #endif
    target.zeromatrix();
  }
  template <class ya_type>
  static inline void mc(ya_type &target,const eltype value) {
    _ya_zero_matrix2<YATRAIT::VME_DIAGONAL>::mc(target,value);
  }
};
template<class eltype> struct _ya_zero_matrix1<eltype,4,1> {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=0 4: STL Map) " << endl;
    #endif
    target.zeromatrix();
  }
  template <class ya_type>
  static inline void mc(ya_type &target,const eltype value) {
    _ya_zero_matrix2<YATRAIT::VME_DIAGONAL>::mc(target,value);
  }
};
template<class eltype,int,int> struct _ya_zero_matrix1 {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    _ya_zero_matrix2<YATRAIT::VME_DIAGONAL>::mc(target);
  }
  template <class ya_type>
  static inline void mc(ya_type &target,const eltype value) {
    _ya_zero_matrix2<YATRAIT::VME_DIAGONAL>::mc(target,value);
  }
};

// ------------------------- For Loop Zero?

// Target is diagonal
template<>
struct _ya_zero_matrix2<1> {
  template <class ya_type>
  static inline void mc(ya_type &target) {
   #ifdef YA_MEMDBG
   cout << endl << " (MDB=0 5: Diagonal) " << endl;
   #endif
   ya_sizet nR=target.rows();
    for (ya_sizet i=0; i<nR; i++)
      target(i,i)=0;
  }
  template <class ya_type, class eltype>
  static inline void mc(ya_type &target,const eltype value) {
    ya_sizet nR=target.rows();
    for (ya_sizet i=0; i<nR; i++)
      target(i,i)=value;
  }
};

// Target is not diagonal

// -- Loop helpers
template <class ref> struct _ya_ufor_zero {
  static inline void op(ref r) { r=0; }
};
template <class ref, class eltype> struct _ya_ufor_set {
  static inline void op(ref r, const eltype v) { r=v; } 
};

template<int a>
struct _ya_zero_matrix2 {
  template <class ya_type>
  static inline void mc(ya_type &target) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=0 6: No_at) " << endl;
    #endif
    ya_no_order_loop<_ya_ufor_zero<typename ya_type::ref> >::lp(target);
  }
  template <class ya_type, class eltype>
  static inline void mc(ya_type &target, const eltype value) {
    ya_no_order_loop<_ya_ufor_set<typename ya_type::ref,eltype> >::lp(target,
                                                                        value);
  }
};

}
#endif
