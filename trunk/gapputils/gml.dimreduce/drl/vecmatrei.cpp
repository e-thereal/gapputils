/***************************************************************************
                                vecmatrei.cpp
                             -------------------
                               W. Michael Brown

  Reindex of a matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Mar 23 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATREI_CPP
#define VECMATREI_CPP
 
using namespace yala;

// Initialize vector
template<class eltype, class mat_type, class mat_type2>
VM_ReI::VecMatReI(mat_type* v, const mat_type2 *ind ) : parent(v),indices(ind) {
}

// Initialize vector
template<class eltype, class mat_type, class mat_type2>
VM_ReI::VecMatReI(const VM_ReI &v) : parent(v.parent),indices(v.indices) {
}

// Access vector element
template<class eltype, class mat_type, class mat_type2>
typename VM_ReI::ref VM_ReI::at(const ya_sizet i) const {
  return parent->at((*indices)(i));
}

// Access vector element
template<class eltype, class mat_type, class mat_type2>
typename VM_ReI::ref VM_ReI::at(const ya_sizet i) {
  return parent->at((*indices)(i));
}

// Access vector element
template<class eltype, class mat_type, class mat_type2>
typename VM_ReI::ref VM_ReI::at2(const ya_sizet row,
                                          const ya_sizet col) const {
  return parent->at((*indices)(row,col));
}

// Access vector element
template<class eltype, class mat_type, class mat_type2>
typename VM_ReI::ref VM_ReI::at2(const ya_sizet row,
                                        const ya_sizet col) {
  return parent->at((*indices)(row,col));
}

#endif
