/***************************************************************************
                              ya_matrm_traits.h
                             -------------------
                               W. Michael Brown

  Traits for YA_MatRM

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

#ifndef YA_MATRM_TRAITS_H
#define YA_MATRM_TRAITS_H

YALANAMESPACE

// Matrix Specific Traits
template <class eltype>
class YA_Traits<YA_MatRMT > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,YA_MatRMT > iter;
  typedef YA_ConstIter<eltype,const YA_MatRMT > citer;
  typedef eltype* rowiter; 
  typedef const eltype * crowiter;

  // Pointers for BLAS, etc.
  typedef rowiter cptr; 
  typedef crowiter const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 1,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 1,
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
template <class eltype>
class YA_Traits<const YA_MatRMT > : 
public YA_Traits<YA_MatRMT > {
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const YA_MatRMT > iter;
  typedef const eltype* rowiter;

  // Pointers for BLAS, etc.
  typedef typename YA_Traits<YA_MatRMT >::crowiter cptr;

  // Reference Types
  typedef const eltype& ref;
};

}
#endif
