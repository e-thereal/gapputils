/***************************************************************************
                               ya_trans_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_Trans

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Oct 18 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_TRANS_TRAITS_H
#define YA_TRANS_TRAITS_H

YALANAMESPACE

template<class eltype, class mat_type>
class YA_Traits<YA_TransT> {
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::rowiter iter;
  typedef typename YA_Traits<mat_type>::crowiter citer;
  typedef typename YA_Traits<mat_type>::iter rowiter;
  typedef typename YA_Traits<mat_type>::citer crowiter;

  // Pointers for BLAS, etc.
  // Pointers for BLAS, etc.
  typedef typename YA_Traits<mat_type>::cptr cptr; 
  typedef typename YA_Traits<mat_type>::const_cptr const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = YA_TRAIT::VME_MEMORY,
    VME_VECTOR = YA_TRAIT::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_SYMMETRIC = YA_TRAIT::VME_SYMMETRIC,
    VME_DIAGONAL = YA_TRAIT::VME_DIAGONAL,
    VME_NCOLS = YA_TRAIT::VME_NROWS,
    VME_NROWS = YA_TRAIT::VME_NCOLS,
    VME_ORDER = _ya_trans_order<YA_TRAIT::VME_ORDER>::ans,
    VME_BLAS = YA_TRAIT::VME_BLAS,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 1
  };
};    

template <class eltype, class mat_type>
class YA_Traits<const YA_TransT> : public YA_Traits<YA_TransT> {};


}

#endif
