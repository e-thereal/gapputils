/***************************************************************************
                                  ya_sort.h
                             -------------------
                               W. Michael Brown

  Sorting functions
 
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

/*! \file */

#ifndef YA_SORT_H
#define YA_SORT_H

YALANAMESPACE

/// In place sort of all elements placed in column major order
/** \note Not supported for VecMatSym matrices
  * \sa ipsort() **/
template<class eltype, class mat_type>
void ipsort_cm(YA_BaseT &mat);

// Const copy
template<class eltype, class mat_type>
void ipsort_cm(const YA_WrapperT &mat);

/// In place sort. For matrices, each column is sorted separately
/** \note Not supported for VecMatSym matrices
  * \sa ipsort_cm() **/
template<class eltype, class mat_type>
void ipsort(YA_BaseT &mat);

// Const copy
template<class eltype, class mat_type>
void ipsort(const YA_WrapperT &mat);

/// Return indices that result in column_major sort of all elements in mat
/** Indices is always returned as a column vector. If smat is the resulting 
  * sorted mat, smat=mat(reshape(indices,mat.rows(),mat.cols())); 
  * \note Not supported for VecMatSym matrices
  * \sa sort_indices() **/
template<class eltype, class mat_type, class ya_type2>
void sort_indices_cm(const YA_BaseT &mat, ya_type2 &indices);

/// Return indices for sort. For matrices, each column is sorted separately
/** If smat is the resulting sorted mat, smat=mat(indices); 
  * \note Not supported for VecMatSym matrices
  * \sa sort_indices_cm() **/
template<class eltype, class mat_type, class ya_type2>
void sort_indices(const YA_BaseT &mat, ya_type2 &indices);

}
#endif
