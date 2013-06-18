/***************************************************************************
                                vecmatscalar.h
                             -------------------
                               W. Michael Brown

  1 x 1 Matrix
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Jan 9 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATSCALAR_CPP
#define VECMATSCALAR_CPP

using namespace yala;

// Initialize
template<class eltype>
VM_Scalar::VecMatScalar() {
}

// Copy constructor
template<class eltype>
VM_Scalar::VecMatScalar(const VecMatScalar &v) : array(v.array) {
}

// Initialize vector
template<class eltype>
VM_Scalar::VecMatScalar(const eltype v) : array(v) {
}

// Clear the matrix
template<class eltype>
void VM_Scalar::clear() {
  YA_DEBUG_ERROR(1==0,"Can't do clear() on 'scalar' matrix.");
}

// Clear any old vector and set up a new one with 'length' elements
template<class eltype>
void VM_Scalar::setup(const ya_sizet len) {
  YA_DEBUG_ERROR(len==1,"Can't setup 'scalar' bigger than 1");
}

// Clear any old matrix and set up a new one
template<class eltype>
void VM_Scalar::setup(const ya_sizet rows, const ya_sizet columns) {
  YA_DEBUG_ERROR(rows==1 && columns==1,"Can't setup 'scalar' bigger than 1");
}

// Access vector element
template<class eltype>
typename VM_Scalar::cref VM_Scalar::at(const ya_sizet i) const {
  return array;
}

// Access vector element
template<class eltype>
typename VM_Scalar::ref VM_Scalar::at(const ya_sizet i) {
  return array;
}

#endif
