/***************************************************************************
                               ya_iter_misc.h
                             -------------------
                               W. Michael Brown

  Helpers functions for iterators

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Feb 12 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_ITER_MISC_H
#define YA_ITER_MISC_H
  
//using namespace std;

YALANAMESPACE

// Used as an iterator to indices of a matrix, rather than the matrix itself
template<class mat_type>
class ya_index_comparator : public std::binary_function<ya_sizet,ya_sizet,bool> {
  const mat_type &parent;
 public:
  ya_index_comparator(const mat_type &m) : parent(m) {}
  bool operator()(const ya_sizet a, const ya_sizet b) const {
    return parent(a)<parent(b);
  }
};

}
#endif
