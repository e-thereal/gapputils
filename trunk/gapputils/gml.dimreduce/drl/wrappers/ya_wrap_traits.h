/***************************************************************************
                               ya_wrap_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_Wrap

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sun Jun 8 2008
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_WRAP_TRAITS_H
#define YA_WRAP_TRAITS_H

YALANAMESPACE

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,eltype *,0,itype> {
  typedef eltype* iter;
  typedef const eltype* citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=1, blas_stride=0 };
  static inline iter begin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,const eltype *,0,itype> {
  typedef const eltype* iter;
  typedef const eltype* citer;
  typedef YA_ConstRowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=1, blas_stride=0 };
  static inline iter begin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,eltype *,1,itype> {
  typedef YA_Iter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef eltype* rowiter;
  typedef const eltype* crowiter;
  enum { memory=1, blas_stride=0 };
  static inline rowiter rowbegin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,const eltype *,1,itype> {
  typedef YA_ConstIter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef const eltype* rowiter;
  typedef const eltype* crowiter;
  enum { memory=1, blas_stride=0 };
  static inline rowiter rowbegin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,eltype *,2,itype> {
  typedef eltype* iter;
  typedef const eltype* citer;
  typedef eltype* rowiter;
  typedef const eltype* crowiter;
  enum { memory=1, blas_stride=0 };
  static inline iter begin(const itype &mat) { return mat.c_ptr(); }
  static inline iter rowbegin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,const eltype *,2,itype> {
  typedef const eltype* iter;
  typedef const eltype* citer;
  typedef const eltype* rowiter;
  typedef const eltype* crowiter;
  enum { memory=1, blas_stride=0 };
  static inline iter begin(const itype &mat) { return mat.c_ptr(); }
  static inline iter rowbegin(const itype &mat) { return mat.c_ptr(); }
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,eltype *,3,itype> {
  typedef YA_Iter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=1, blas_stride=1 };
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,const eltype *,3,itype> {
  typedef YA_Iter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=1, blas_stride=1 };
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype>
struct YA_TITER {
  typedef YA_Iter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  static inline iter begin(itype &mat) { return iter(&mat); }
  static inline rowiter rowbegin(itype &mat) { return rowiter(&mat); }
  enum { memory=0, blas_stride=0 };
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype>
struct YA_TITER<eltype,const mat_type,order,itype> {
  typedef YA_ConstIter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef YA_ConstRowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  static inline iter begin(itype &mat) { return iter(&mat); }
  static inline rowiter rowbegin(itype &mat) { return rowiter(&mat); }
  enum { memory=0, blas_stride=0 };
};

// Matrix Specific Traits
template <class eltype, class mat_type, int order, int nrows, int ncols,
          class reft, class creft, int fixt>
class YA_Traits<YA_WrapT > { 
 public:
  // Iterators
  typedef typename YA_TITER<eltype,mat_type,order,YA_WrapT>::iter iter; 
  typedef typename YA_TITER<eltype,mat_type,order,YA_WrapT>::citer citer;
  typedef typename YA_TITER<eltype,mat_type,order,YA_WrapT>::rowiter rowiter;
  typedef typename YA_TITER<eltype,mat_type,order,YA_WrapT>::crowiter crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef reft ref; 
  typedef creft cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = YA_TITER<eltype,mat_type,order,YA_WrapT>::memory,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<nrows,ncols>::isone,
    VME_COL_VECTOR = _VM_TEMPLATE_VALUE<ncols>::isone,
    VME_ROW_VECTOR = _VM_TEMPLATE_VALUE<nrows>::isone,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = ncols,
    VME_NROWS = nrows,
    VME_ORDER = order,
    VME_BLAS = _YA_TAND<VME_MEMORY,_VM_BLAS_TYPE<eltype>::supported>::ans,
    VME_BLAS_STRIDE = _YA_TOR<VME_BLAS,YA_TITER<eltype,mat_type,order,YA_WrapT>::blas_stride>::ans,
    #ifdef YA_BLAS
    VME_USE_BLAS = _YA_TAND<VME_BLAS,_YA_TNOT<fixt>::ans>::ans,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

template <class eltype, class mat_type, int order, int nrows, int ncols,
          class reft, class creft, int fixt>
class YA_Traits<const YA_WrapT > : 
public YA_Traits<YA_WrapT > { };

}
#endif
