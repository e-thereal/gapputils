/***************************************************************************
                                vecmatr.cpp
                             -------------------
                               W. Michael Brown

  Row vector of matrix

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

#ifndef VECMATR_CPP
#define VECMATR_CPP
 
using namespace yala;

// Initialize vector
template<class eltype, class mat_type>
VM_Row::VecMatR(mat_type *v, const ya_sizet rnum) : parent(v), row_num(rnum) {
  YA_INDEX_ERROR(rnum>=0 && rnum<v->rows(),
                 "Row number for slice bigger than matrix.",*v,rnum);
}

// Initialize vector
template<class eltype, class mat_type>
VM_Row::VecMatR(const VM_Row &v) : parent(v.parent), row_num(v.row_num) {
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Row::ref VM_Row::at(const ya_sizet i) const {
  return this->parent->at2(row_num,i);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Row::ref VM_Row::at(const ya_sizet i) {
  return this->parent->at2(row_num,i);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Row::ref VM_Row::rowmajor_at(const ya_sizet i) const {
  return this->parent->at2(row_num,i);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Row::ref VM_Row::rowmajor_at(const ya_sizet i) {
  return this->parent->at2(row_num,i);
}

#endif
