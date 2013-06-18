/***************************************************************************
                                 ya_int_eig.h
                             -------------------
                               W. Michael Brown

  Internal functions for eigen decomposition
 
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

#ifndef YA_INT_EIG_H
#define YA_INT_EIG_H

YALANAMESPACE

// Full Eigen decomposition of square symmetric matrix
// NOTE: The matrix input matrix is not preserved
// Eigen values are returned in ascending order
template<class eltype, class ya_type1, class ya_type2>
inline void ya_simple_eig(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values);

// Full Eigen decomposition of square symmetric matrix
// NOTE: The matrix input matrix is not preserved
// Eigen values are returned in descending order
template<class eltype, class ya_type1, class ya_type2>
inline void ya_simple_eighl(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values);

// Full Eigen decomposition of square symmetric matrix
// NOTE: The matrix input matrix is not preserved
// Eigen values are returned according to sortop order
template<class eltype, class ya_type1, class ya_type2, class sortop>
inline void ya_simple_eig_s(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values);

}
#endif
