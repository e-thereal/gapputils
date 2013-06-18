/***************************************************************************
                               ya_allocate.h
                             -------------------
                               W. Michael Brown

  Functions for allocation of memory for matrices/vectors
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

#ifndef YA_ALLOCATE_H
#define YA_ALLOCATE_H
 
YALANAMESPACE

// Clear the matrix
template<class eltype>
inline void ya_free_memory(eltype *& array) {
  if (array!=NULL) {
    delete[] array;
    array=NULL;
  }
}

// Reserve memory for vecmat
template<class eltype>
inline void ya_reserve_memory(eltype *& array,const int length) {
  array=new eltype[length];
}

/*
// Reserve memory for vecmat
template<class eltype>
inline void ya_reserve_memory(eltype *& array,const int length,Error &error) {
  try {
    array=new eltype[length];
  }
  catch(bad_alloc errora) {
    error.buffer() << "Memory allocation error. The vector/matrix requested "
                   << "requires " << length*sizeof(eltype)/(1024*1024)
                   << "MB\nSYSTEM: " << errora.what();
    error.addbuf(300,19,"YALA++");
  }
}
*/

}
#endif
