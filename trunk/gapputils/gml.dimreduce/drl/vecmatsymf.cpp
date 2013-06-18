/***************************************************************************
                                vecmatsym.cpp
                             -------------------
                               W. Michael Brown

  Symmetric Matrix Class
  Insert/Access in constant time

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

#ifndef VECMATSYMF_CPP
#define VECMATSYMF_CPP

using namespace yala;

/// Initialize vector
template<class eltype, ya_sizet nrows>
VM_SymMatF::VecMatSymF() {
}

/// Copy constructor
template<class eltype, ya_sizet nrows>
VM_SymMatF::VecMatSymF(const VecMatSymF &v) {
  (*this)=v;
}

// Initialize matrix
template<class eltype, ya_sizet nrows>
VM_SymMatF::VecMatSymF(const ya_sizet rowi, const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi==columni && rowi==nrows,
    "Attempt to setup symmetric fixed size matrix with different dimensions");
}

template<class eltype, ya_sizet nrows>
template<class ya_type2>
VM_SymMatF::VecMatSymF(const ya_type2 &v) {
  (*this)=v;
}

// Clear any old matrix and set up a new one
template<class eltype, ya_sizet nrows>
void VM_SymMatF::setup(const ya_sizet rowi,const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi==columni && rowi==nrows,
    "Attempt to setup symmetric fixed size matrix with different dimensions");
}

// Clear any old matrix and set up a new one
template<class eltype, ya_sizet nrows>
void VM_SymMatF::setup(const ya_sizet length) {
  YA_DEBUG_ERROR(length==nrows*nrows,
    "Attempt to setup symmetric fixed size matrix with different length");
}

// Clear the matrix
template<class eltype, ya_sizet nrows>
void VM_SymMatF::clear() {
  YA_DEBUG_ERROR(1==0,
    "Attempt to clear fixed size matrix.");
}

// Access matrix element
template<class eltype, ya_sizet nrows>
typename VM_SymMatF::cref VM_SymMatF::at2(const ya_sizet row,
                                                const ya_sizet col) const {
  if (row>col)
    return (this->array)[row*(row+1)/2+col];
  else
    return (this->array)[col*(col+1)/2+row];
}

// Access matrix element
template<class eltype, ya_sizet nrows>
typename VM_SymMatF::ref VM_SymMatF::at2(const ya_sizet row, const ya_sizet col) {
  if (row>col)
    return (this->array)[row*(row+1)/2+col];
  else
    return (this->array)[col*(col+1)/2+row];
}

// Access vector element
template<class eltype, ya_sizet nrows>
typename VM_SymMatF::cref VM_SymMatF::at(const ya_sizet i) const {
  return array[i];
}

// Access vector element
template<class eltype, ya_sizet nrows>
typename VM_SymMatF::ref VM_SymMatF::at(const ya_sizet i) {
  return array[i];
}

#endif
