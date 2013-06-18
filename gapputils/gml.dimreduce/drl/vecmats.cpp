/***************************************************************************
                                vecmats.cpp
                             -------------------
                               W. Michael Brown

  General Slice of another matrix

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

#ifndef VECMATS_CPP
#define VECMATS_CPP
 
using namespace yala;

// Initialize 
template<class eltype, class mat_type, class rows_type, class cols_type>
VM_Slice::VecMatS(mat_type *v, const rows_type &rowsin,
                  const cols_type &colsin) : parent(v), rowsi(rowsin),
                  colsi(colsin), nlength(rowsi.numel()*colsi.numel()) {
}

// Initialize vector
template<class eltype, class mat_type, class rows_type, class cols_type>
VM_Slice::VecMatS(const VM_Slice &v) : parent(v.parent), rowsi(v.rowsi),
                                       colsi(v.colsi),nlength(v.nlength) {
}

// Access vector element
template<class eltype, class mat_type, class rows_type, class cols_type>
typename VM_Slice::ref VM_Slice::at(const ya_sizet i) const {
  div_t index=div(i,rows());
  return at2(index.rem,index.quot);
}

// Access vector element
template<class eltype, class mat_type, class rows_type, class cols_type>
typename VM_Slice::ref VM_Slice::at(const ya_sizet i) {
  div_t index=div(i,rows());
  return at2(index.rem,index.quot);
}

// Access vector element
template<class eltype, class mat_type, class rows_type, class cols_type>
typename VM_Slice::ref VM_Slice::at2(const ya_sizet row,
                                          const ya_sizet col) const {
  return parent->at2(rowsi(row), colsi(col));
}

// Access vector element
template<class eltype, class mat_type, class rows_type, class cols_type>
typename VM_Slice::ref VM_Slice::at2(const ya_sizet row,
                                            const ya_sizet col) {
  return parent->at2(rowsi(row), colsi(col));
}

#endif
