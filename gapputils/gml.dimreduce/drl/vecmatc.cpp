/***************************************************************************
                                vecmatc.cpp
                             -------------------
                               W. Michael Brown

  Column vector of matrix
  This is a slice of another matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Oct 17 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATC_CPP
#define VECMATC_CPP
 
using namespace yala;

// Initialize vector
template<class eltype, class mat_type>
VM_Column::VecMatC(mat_type *v, const ya_sizet col_num) : parent(v),
                                                         column_num(col_num) {
  YA_INDEX_ERROR(col_num>=0 && col_num<v->cols(),
                 "Column for slice too big for matrix",*v,col_num);
}

// Initialize vector
template<class eltype, class mat_type>
VM_Column::VecMatC(const VM_Column &v) : parent(v.parent),
                                         column_num(v.column_num) {
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Column::ref VM_Column::at(const ya_sizet i) const {
  return parent->at2(i,column_num);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Column::ref VM_Column::at(const ya_sizet i) {
  return parent->at2(i,column_num);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Column::ref VM_Column::rowmajor_at(const ya_sizet i) const {
  return parent->at2(i,column_num);
}

// Access vector element
template<class eltype, class mat_type>
typename VM_Column::ref VM_Column::rowmajor_at(const ya_sizet i) {
  return parent->at2(i,column_num);
}

#endif
