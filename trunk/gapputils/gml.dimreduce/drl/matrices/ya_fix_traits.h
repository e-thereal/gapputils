/***************************************************************************
                               ya_fix_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_Fix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Apr 17 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_FIX_TRAITS_H
#define YA_FIX_TRAITS_H

YALANAMESPACE

// Matrix Specific Traits
template <class eltype, ya_sizet nrows, ya_sizet ncols>
class YA_Traits<YA_FixT > { 
 public:
  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 1,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<nrows,ncols>::isone,
    VME_COL_VECTOR = _VM_TEMPLATE_VALUE<ncols>::isone,
    VME_ROW_VECTOR = _VM_TEMPLATE_VALUE<nrows>::isone,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = ncols,
    VME_NROWS = nrows,
    VME_ORDER = _YA_TEMPLATE_00_12<VME_VECTOR>::ans,
    VME_BLAS = _VM_BLAS_TYPE<eltype>::supported,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = 0,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };

  // Iterators
  typedef eltype* iter; 
  typedef const eltype * citer;
  typedef typename _YA_CHOOSETYPE1<VME_VECTOR,eltype*,
                                   YA_RowIter<eltype,YA_FixT> >::ans rowiter;
  typedef typename _YA_CHOOSETYPE1<VME_VECTOR,const eltype*,
                             YA_ConstRowIter<eltype,YA_FixT> >::ans crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

};

// Traits for const Matrices
template <class eltype, ya_sizet nrows, ya_sizet ncols>
class YA_Traits<const YA_FixT > :
public YA_Traits<YA_FixT > { 
 public:
  // Iterators
  typedef const eltype * iter; 
  typedef YA_ConstRowIter<eltype,const YA_FixT > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype& ref; 
};

}
#endif
