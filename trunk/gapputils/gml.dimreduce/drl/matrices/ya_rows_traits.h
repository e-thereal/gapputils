/***************************************************************************
                               ya_rows_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_Rows

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

#ifndef YA_ROWS_TRAITS_H
#define YA_ROWS_TRAITS_H

YALANAMESPACE

// Matrix Specific Traits
template <class eltype, ya_sizet nrows>
class YA_Traits<YA_RowsT > { 
 public:
  // Iterators
  typedef eltype* iter; 
  typedef const eltype * citer;
  typedef typename _YA_CHOOSETYPE1<nrows,eltype*,
                                   YA_RowIter<eltype,YA_RowsT> >::ans rowiter;
  typedef typename _YA_CHOOSETYPE1<nrows,const eltype*,
                             YA_ConstRowIter<eltype,YA_RowsT> >::ans crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 1,
    VME_VECTOR = _VM_TEMPLATE_VALUE<nrows>::isone,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = VME_VECTOR,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = nrows,
    VME_ORDER = _YA_TEMPLATE_00_12<VME_VECTOR>::ans,
    VME_BLAS = _VM_BLAS_TYPE<eltype>::supported,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, ya_sizet nrows>
class YA_Traits<const YA_RowsT > :
public YA_Traits<YA_RowsT > { 
 public:
  // Iterators
  typedef const eltype * iter; 
  typedef const eltype* rowiter;

  // Pointers for BLAS, etc.
  typedef typename YA_Traits<YA_RowsT >::citer cptr; 

  // Reference Types
  typedef const eltype& ref; 
};


}
#endif
