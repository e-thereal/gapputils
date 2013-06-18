/***************************************************************************
                             ya_compare_matrix.h
                             -------------------
                               W. Michael Brown

  Routines for comparing matrices
                               
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

#ifndef YA_COMPARE_MATRIX_H
#define YA_COMPARE_MATRIX_H

YALANAMESPACE

// ----------------------------------------------------------------
// All routines assume that the target matrix is the same size as
// the source matrix.
// ----------------------------------------------------------------

template<class eltype,int,int,int,int> struct _ya_comp_matrix1;
template<int> struct _ya_comp_matrix2;

template<class eltype, class mat_type, class eltype2, class mat_type2>
inline bool ya_compare_matrix(const YA_BaseT &target, const YA_BaseT2T &source) {
  YA_DEBUG_ERROR(target.rows()==source.rows() && 
                 target.cols()==source.cols(),
                 "Cannot compare different sized matrices");
  return _ya_comp_matrix1<eltype,YA_TRAIT::VME_MEMORY,YA_TRAIT2::VME_MEMORY,
                     ya_direct_mem<eltype,eltype2>::ans,
                     _YA_TSAME_ORDER<YA_TRAIT::VME_ORDER,
                     YA_TRAIT2::VME_ORDER>::ans>::mc(target.stv(),source.stv());
}

// ------------------------- Direct Memory Compare?
template<class eltype> struct _ya_comp_matrix1<eltype,1,1,1,1> {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==1: memcmp) " << endl;
    #endif
    if (memcmp(target.c_ptr(),source.c_ptr(),
               target.numel()*sizeof(eltype))==0)
      return true;
    return false;
  }
};
template<class eltype> struct _ya_comp_matrix1<eltype,2,2,1,1> {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==2: memcmp) " << endl;
    #endif
    if (memcmp(target.c_ptr(),source.c_ptr(),
               target.numel()*sizeof(eltype))==0)
      return true;
    return false;
  }
};
template<class eltype> struct _ya_comp_matrix1<eltype,3,3,1,1> {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==3: STL Vec) " << endl;
    #endif
    return target.comparematrix(source);
  }
};
template<class eltype> struct _ya_comp_matrix1<eltype,4,4,1,1> {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==4: STL Map) " << endl;
    #endif
    return target.comparematrix(source);
  }
};
template<class eltype,int,int,int,int> struct _ya_comp_matrix1 {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    return _ya_comp_matrix2<_YA_TAND<YATRAIT1::VME_DIAGONAL,
                              YATRAIT2::VME_DIAGONAL>::ans>::mc(target,source);
  }
};

// ------------------------- For Loop Compare?

// Target and source are diagonal
template<>
struct _ya_comp_matrix2<1> {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==5: Diagonal) " << endl;
    #endif
    ya_sizet nR=target.rows();
    for (ya_sizet i=0; i<nR; i++)
      if (target(i,i)!=source(i,i))
        return false;
    return true;
  }
};

// -- Loop helpers
template <class ref1, class ref2> struct _ya_bfor_cmp {
  static inline bool op(ref1 r,ref2 s) { return r==s; }
};


// Use loop template
template<int> struct _ya_comp_matrix2 {
  template <class ya_type1, class ya_type2>
  static inline bool mc(const ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB==6: Template loop) " << endl;
    #endif
    return ya_binary_loop<_ya_bfor_cmp<typename ya_type1::cref,
                                typename ya_type2::cref> >::lpr(target,source);
  }
};

}
#endif
