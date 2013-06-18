/***************************************************************************
                               ya_base_traits.h
                             -------------------
                               W. Michael Brown

  Traits for Vector/Matrix Base Class

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

#ifndef YA_BASE_TRAITS_H
#define YA_BASE_TRAITS_H

YALANAMESPACE

template <class eltype, class mat_type>
class YA_Traits<YA_Base<eltype,mat_type> > {
 public:
  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = YA_Traits<mat_type>::VME_MEMORY,
    VME_VECTOR = YA_Traits<mat_type>::VME_VECTOR,
    VME_COL_VECTOR = YA_Traits<mat_type>::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_Traits<mat_type>::VME_ROW_VECTOR,
    VME_SYMMETRIC = YA_Traits<mat_type>::VME_SYMMETRIC,
    VME_DIAGONAL = YA_Traits<mat_type>::VME_DIAGONAL,
    VME_NCOLS = YA_Traits<mat_type>::VME_NCOLS,
    VME_NROWS = YA_Traits<mat_type>::VME_NROWS,
    VME_ORDER = YA_Traits<mat_type>::VME_ORDER,
    VME_BLAS = YA_Traits<mat_type>::VME_BLAS,
    VME_BLAS_STRIDE = YA_Traits<mat_type>::VME_BLAS_STRIDE,
    VME_USE_BLAS = YA_Traits<mat_type>::VME_USE_BLAS
  };
};

template <class eltype, class mat_type>
class YA_Traits<const YA_Base<eltype,mat_type> > :
public YA_Traits<YA_Base<eltype,mat_type> > { };

}
#endif
