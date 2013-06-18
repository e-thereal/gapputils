/***************************************************************************
                                   ya_fun.h
                             -------------------
                               W. Michael Brown

  Misc functions
 
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

#ifndef YA_FUN_H
#define YA_FUN_H

/*! \file */

YALANAMESPACE

/// Generate a temporary copy of a matrix
template<class eltype, class mat_type>
inline typename YA_COPY_TYPE copy(const YA_BaseT &mat);

/// Set all elements in the matrix to be 0
template<class eltype, class mat_type>
inline void zero(YA_BaseT &mat);

// Const copy
template<class eltype, class mat_type>
inline void zero(const YA_WrapperT &mat);

/// Set each element in the matrix to a value
/** \sa zero() **/
template<class eltype, class mat_type>
inline void setall(YA_BaseT &mat, const eltype value);

// Const copy
template<class eltype, class mat_type>
inline void setall(const YA_WrapperT &mat, const eltype value);

/// Test if two matrices are equal (Matrices must be same size!)
template<class eltype, class mat_type, class ya_type2>
inline bool identical(const YA_BaseT &one, const ya_type2 &two);

/// Swap two matrices, slices, etc.
template<class eltype, class mat_type, class ya_type2>
inline void ya_swap(YA_BaseT &one, ya_type2 &two);

// Const copy
template<class eltype, class mat_type, class ya_type2>
inline void ya_swap(const YA_WrapperT &one, ya_type2 &two);

/// Reverse the elements in a matrix in column major order
template<class eltype, class mat_type>
inline void ip_flipcm(YA_BaseT &mat);

// Const copy
template<class eltype, class mat_type>
inline void ip_flipcm(const YA_WrapperT &mat);

/// Reverse order of the columns in matrix or vector
/** \note for a column vector this does nothing **/
template<class eltype, class mat_type>
inline void ip_fliplr(YA_BaseT &mat);
  
// Const copy
template<class eltype, class mat_type>
inline void ip_fliplr(const YA_WrapperT &mat);

/// Transpose the matrix (mat=copy(mat.T());
template<class eltype, class mat_type>
inline void ip_transpose(YA_BaseT &mat);

// Const copy
template<class eltype, class mat_type>
inline void ip_transpose(const YA_WrapperT &mat);

/// Return the L2-norm
template<class eltype, class mat_type>
eltype l2_norm(const YA_BaseT &mat);

/// Return the sum of all elements
/** \note For VecMatSym, the sum is only over the triangle **/
template<class eltype, class mat_type>
eltype total(const YA_BaseT &mat);

/// Return the product of all elements
/** \note For VecMatSym, the product is only over the triangle **/
template<class eltype, class mat_type>
eltype product(const YA_BaseT &mat);

/// Return the minimum element in the matrix
template<class eltype, class mat_type>
eltype minimum(const YA_BaseT &mat);

/// Return the maximum element in the matrix
template<class eltype, class mat_type>
eltype maximum(const YA_BaseT &mat);

/// Return the minimum, maximum, and mean
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean);

/// Return the minimum, maximum, mean, and standard deviation
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean,
           stype &vstd);

/// Return the minimum, maximum, and mean of each column
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, YA_RowT &vmin, YA_RowT &vmax, 
           YA_Rows<stype,1> &vmean);

/// Return the minimum, maximum, mean, and standard deviation of each column
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, YA_RowT &vmin, YA_RowT &vmax, 
           YA_Rows<stype,1> &vmean, YA_Rows<stype,1> &vstd);

/// Compute the squared euclidean distance elements in row1 and row2
template<class eltype, class mat_type, class ya_type2>
eltype dist2(const YA_BaseT &one, const ya_type2 &two);

/// Return a vector with indices to non-zero elements in the matrix
template<class eltype, class mat_type>
YA_DynI find(const YA_BaseT &mat);

}
#endif
