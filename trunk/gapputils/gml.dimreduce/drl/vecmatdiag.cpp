/***************************************************************************
                                vecmatdiag.h
                             -------------------
                               W. Michael Brown

  Create diagonal matrix from a vector

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

#ifndef VECMATDIAG_CPP
#define VECMATDIAG_CPP
 
using namespace yala;

// Initialize vector
template<class eltype, class mat_type>
VM_Diag::VecMatDiag(mat_type *v) : parent(v), offdiag(0) {
}

// Initialize vector
template<class eltype, class mat_type>
VM_Diag::VecMatDiag(const VM_Diag &v) : parent(v.parent), offdiag(0) {
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Diag::ref VM_Diag::at(const ya_sizet i) const {
  return this->parent->at(i);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Diag::ref VM_Diag::at(const ya_sizet i) {
  return this->parent->at(i);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Diag::ref VM_Diag::at2(const ya_sizet row,
                                           const ya_sizet col) const {
  if (row!=col)
    return 0;
  return parent->at(row);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Diag::ref VM_Diag::at2(const ya_sizet row,
                                           const ya_sizet col) {
  if (row!=col)
    return 0;
  return parent->at(row);
}

#endif
