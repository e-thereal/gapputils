/***************************************************************************
                                vecmatr.cpp
                             -------------------
                               W. Michael Brown

  Transpose of another matrix

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

#ifndef VECMATTRANS_CPP
#define VECMATTRANS_CPP
 
using namespace yala;

// Initialize vector
template<class eltype, class mat_type>
YA_TransT::YA_Trans(mat_type *v) : parent(v) {
}

// Initialize vector
template<class eltype, class mat_type>
YA_TransT::YA_Trans(const YA_TransT &v) : parent(v.parent) {
}

// Access vector element
template<class eltype, class mat_type>
typename YA_TransT::ref YA_TransT::at(const ya_sizet i) const {
  div_t index=div(i,rows());
  return parent->at2(index.quot,index.rem);
}

// Access vector element
template<class eltype, class mat_type>
typename YA_TransT::ref YA_TransT::at(const ya_sizet i) {
  div_t index=div(i,rows());
  return parent->at2(index.quot,index.rem);
}

// Access vector element
template<class eltype, class mat_type>
typename YA_TransT::ref YA_TransT::at2(const ya_sizet row,
                                             const ya_sizet col) const {
  return parent->at2(col,row);
}

// Access vector element
template<class eltype, class mat_type>
typename YA_TransT::ref YA_TransT::at2(const ya_sizet row,
                                                      const ya_sizet col) {
  return parent->at2(col,row);
}

#endif
