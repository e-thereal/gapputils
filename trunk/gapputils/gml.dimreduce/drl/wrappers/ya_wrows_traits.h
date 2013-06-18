/***************************************************************************
                              ya_wrows_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_WRows

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue July 1 2008
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_WROWS_TRAITS_H
#define YA_WROWS_TRAITS_H

YALANAMESPACE

// Matrix Specific Traits
template <class eltype, class mat_type, int nrows, int ncols,
          class reft, class creft, int fixt>
class YA_Traits<YA_WRowsT > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,YA_WRowsT > iter; 
  typedef YA_ConstIter<eltype,const YA_WRowsT > citer;
  typedef YA_RowIter<eltype,YA_WRowsT > rowiter;
  typedef YA_ConstRowIter<eltype,const YA_WRowsT > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef reft ref; 
  typedef creft cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<nrows,ncols>::isone,
    VME_COL_VECTOR = _VM_TEMPLATE_VALUE<ncols>::isone,
    VME_ROW_VECTOR = _VM_TEMPLATE_VALUE<nrows>::isone,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = ncols,
    VME_NROWS = nrows,
    VME_ORDER = 4,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = 0,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

template <class eltype, class mat_type, int nrows, int ncols,
          class reft, class creft, int fixt>
class YA_Traits<const YA_WRowsT > : 
public YA_Traits<YA_WRowsT > { };

}
#endif
