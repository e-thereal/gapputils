/***************************************************************************
                               ya_copy_matrix.h
                             -------------------
                               W. Michael Brown

  Routines for copying matrices
                               
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

#ifndef YA_COPY_MATRIX_H
#define YA_COPY_MATRIX_H

YALANAMESPACE

// ----------------------------------------------------------------
// All routines assume that the target matrix is the same size as
// the source matrix.
// ----------------------------------------------------------------

template<class eltype,int,int,int,int> struct _ya_copy_matrix1;
template<int,int> struct _ya_copy_matrix2;

template<class eltype, class mat_type, class eltype2, class mat_type2>
inline void ya_copy_matrix(YA_BaseT &target, const YA_BaseT2T &source) {
  #ifdef YA_SAFE
  YA_MatT new_mat(source.rows(),source.cols());
  _ya_copy_matrix1<eltype,YA_Traits<YA_MatT>::VME_MEMORY,YA_TRAIT2::VME_MEMORY,
                   ya_direct_mem<eltype,eltype2>::ans,
                   _YA_TSAME_ORDER<YA_Traits<YA_MatT>::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(new_mat.stv(),source.stv());
  _ya_copy_matrix1<eltype,YA_TRAIT::VME_MEMORY,YA_Traits<YA_MatT>::VME_MEMORY,
                   ya_direct_mem<eltype,eltype>::ans,
                   _YA_TSAME_ORDER<YA_TRAIT::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(target.stv(),new_mat.stv());
  #else
  _ya_copy_matrix1<eltype,YA_TRAIT::VME_MEMORY,YA_TRAIT2::VME_MEMORY,
                   ya_direct_mem<eltype,eltype2>::ans,
                   _YA_TSAME_ORDER<YA_TRAIT::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(target.stv(),source.stv());
  #endif
}

template<class eltype, class mat_type, class eltype2, class mat_type2>
inline void ya_copy_matrix(const YA_WrapperT &target, const YA_BaseT2T &source) {
  #ifdef YA_SAFE
  YA_MatT new_mat(source.rows(),source.cols());
  _ya_copy_matrix1<eltype,YA_Traits<YA_MatT>::VME_MEMORY,YA_TRAIT2::VME_MEMORY,
                   ya_direct_mem<eltype,eltype2>::ans,
                   _YA_TSAME_ORDER<YA_Traits<YA_MatT>::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(new_mat.stv(),source.stv());
  _ya_copy_matrix1<eltype,YA_TRAIT::VME_MEMORY,YA_Traits<YA_MatT>::VME_MEMORY,
                   ya_direct_mem<eltype,eltype>::ans,
                   _YA_TSAME_ORDER<YA_TRAIT::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(target.stv(),new_mat.stv());
  #else
  _ya_copy_matrix1<eltype,YA_TRAIT::VME_MEMORY,YA_TRAIT2::VME_MEMORY,
                   ya_direct_mem<eltype,eltype2>::ans,
                   _YA_TSAME_ORDER<YA_TRAIT::VME_ORDER,
                   YA_TRAIT2::VME_ORDER>::ans>::mc(target.stv(),source.stv());
  #endif
}

// ------------------------- Direct Memory Copy?
template<class eltype> struct _ya_copy_matrix1<eltype,1,1,1,1> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=1: memcpy) " << endl << flush;
    #endif
    YA_DEBUG_ERROR1(target.c_ptr()!=source.c_ptr(),
                    "Attempt to copy matrix onto itself.",target);
    memcpy(target.c_ptr(),source.c_ptr(),target.numel()*sizeof(eltype));
  }
};
template<class eltype> struct _ya_copy_matrix1<eltype,2,2,1,1> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=2: memcpy) " << endl << flush;
    #endif
    YA_DEBUG_ERROR1(target.c_ptr()!=source.c_ptr(),
                    "Attempt to copy matrix onto itself.",target);
    memcpy(target.c_ptr(),source.c_ptr(),target.numel()*sizeof(eltype));
  }
};
template<class eltype> struct _ya_copy_matrix1<eltype,3,3,1,1> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=3: STL Vec) " << endl << flush;
    #endif
    target.copymatrix(source);
  }
};
template<class eltype> struct _ya_copy_matrix1<eltype,4,4,1,1> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=4: STL Map) " << endl << flush;
    #endif
    target.copymatrix(source);
  }
};
template<class eltype, int a,int b,int c> 
struct _ya_copy_matrix1<eltype,a,6,b,c> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=5: zeros) " << endl << flush;
    #endif
    zero(target);
  }
};
template<class eltype, int,int,int,int> struct _ya_copy_matrix1 {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    _ya_copy_matrix2<YATRAIT1::VME_DIAGONAL,
                     YATRAIT2::VME_DIAGONAL>::mc(target,source);
  }
};

// ------------------------- For Loop Copy?

// Target is not diagonal, source is
template<> struct _ya_copy_matrix2<0,1> {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=7: from diagonal) " << endl << flush;
    #endif
    zero(target);
    ya_sizet nR=target.rows();
    for (ya_sizet i=0; i<nR; i++)
      target(i,i)=source(i,i);
  }
};

// -- Loop helpers
template <class ref1, class ref2> struct _ya_bfor_copy {
  static inline void op(ref1 r,ref2 s) { r=s; }
};


// Use loops
template<int, int> struct _ya_copy_matrix2 {
  template <class ya_type1, class ya_type2>
  static inline void mc(ya_type1 &target, const ya_type2 &source) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB=8: for template) " << endl << flush;
    #endif
    ya_binary_loop<_ya_bfor_copy<typename ya_type1::ref,
                                 typename ya_type2::cref> >::lm(target,source);
  }
};

}
#endif
