/***************************************************************************
                                 ya_sort.cpp
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

#ifndef YA_SORT_S_H
#define YA_SORT_S_H

YALANAMESPACE

template<class eltype, class mat_type>
void ipsort_cm(YA_BaseT &mat) {
  std::sort(mat.begin(),mat.end());
}

template<class eltype, class mat_type>
void ipsort_cm(const YA_WrapperT &mat) {
  std::sort(mat.begin(),mat.end());
}

template<class eltype, class mat_type>
void ipsort(YA_BaseT &mat) {
  if (YA_TRAIT::VME_VECTOR)
    ipsort_cm(mat);
  else if (mat.rows()==1 || mat.cols()==1)
    ipsort_cm(mat);
  else {
    YA_DEBUG_ERROR(mat.rows()*mat.cols()==mat.numel(),
                   "Can't sort symmetric matrix");
    typename YA_BaseT::iter j=mat.begin();
    ya_sizet mC=mat.cols();
    ya_sizet mR=mat.rows();
    for (ya_sizet i=0; i<mC; i++) {
      std::sort(j,j+mR);
      j=j+mR;
    }
  }
}

template<class eltype, class mat_type>
void ipsort(const YA_WrapperT &mat) {
  if (YA_TRAIT::VME_VECTOR)
    ipsort_cm(mat);
  else if (mat.rows()==1 || mat.cols()==1)
    ipsort_cm(mat);
  else {
    YA_DEBUG_ERROR(mat.rows()*mat.cols()==mat.numel(),
                   "Can't sort symmetric matrix");
    typename YA_BaseT::citer j=mat.begin();
    ya_sizet mC=mat.cols();
    ya_sizet mR=mat.rows();
    for (ya_sizet i=0; i<mC; i++) {
      std::sort(j,j+mR);
      j=j+mR;
    }
  }
}

template<class eltype, class mat_type, class ya_type2>
void sort_indices_cm(const YA_BaseT &mat, ya_type2 &indices) {
  indices=VM_ZeroToN(mat.numel());
  std::sort(indices.begin(),indices.end(),
            ya_index_comparator<mat_type>(mat.stv()));
}

template<class eltype, class mat_type, class ya_type2>
void sort_indices(const YA_BaseT &mat, ya_type2 &indices) {
  YA_DEBUG_ERROR(mat.rows()*mat.cols()==mat.numel(),
                 "Can't sort symmetric matrix");
  indices=VM_CountMatU(mat.rows(),mat.cols(),0);
  if (YA_TRAIT::VME_VECTOR)
    std::sort(indices.begin(),indices.end(),
              ya_index_comparator<mat_type>(mat.stv()));
  else if (mat.rows()==1 || mat.cols()==1)
    std::sort(indices.begin(),indices.end(),
              ya_index_comparator<mat_type>(mat.stv()));
  else {
    typename ya_type2::iter j=indices.begin();
    ya_sizet mC=mat.cols();
    ya_sizet mR=mat.rows();
    for (ya_sizet i=0; i<mC; i++) {
      std::sort(j,j+mR,ya_index_comparator<mat_type>(mat.stv()));
      j=j+mR;
    }
  }
}

}
#endif
