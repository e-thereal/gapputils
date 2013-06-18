/***************************************************************************
                                  ya_debug.h
                             -------------------
                               W. Michael Brown

  Debugging helper functions for YALA++

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Jan 5 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_DEBUG_H
#define YA_DEBUG_H
#include <cassert>  
#include <iostream>
//using namespace std;

YALANAMESPACE

inline void ya_check_debug() {
  #ifdef YA_DEBUG
  std::cerr << "Running in DEBUG mode. This will affect performance.\n";
  #endif
  #ifdef YA_SAFE
  std::cerr << "Running in SAFE mode. This will affect performance.\n";
  #endif
}

#ifdef YA_DEBUG
#define YA_DEBUG2
#endif

#ifdef YA_DEBUG1
#define YA_DEBUG
#define YA_DEBUG_ERROR(a,b) ya_debug_error(a,b); assert(a); 
#define YA_DEBUG_ERROR1(a,b,c) ya_debug_error(a,b); assert(a); 
#define YA_INDEX_ERROR(a,b,c,d) ya_debug_error(a,b); assert(a); 
#define YA_INDEX2_ERROR(a,b,c,d,e) ya_debug_error(a,b); assert(a); 
#define YA_DEBUG_ERROR2(a,b,c,d) ya_debug_error(a,b); assert(a); 
#endif

#ifdef YA_DEBUG2
#define YA_DEBUG
#define YA_DEBUG_ERROR(a,b) ya_debug_error(a,b); assert(a); 
#define YA_DEBUG_ERROR1(a,b,c) ya_debug_error(a,b,c); assert(a); 
#define YA_INDEX_ERROR(a,b,c,d) ya_debug_error(a,b,c,d); assert(a); 
#define YA_INDEX2_ERROR(a,b,c,d,e) ya_debug_error(a,b,c,d,e); assert(a); 
#define YA_DEBUG_ERROR2(a,b,c,d) ya_debug_error2(a,b,c,d); assert(a); 
#endif

#ifdef YA_DEBUG

// Generate a runtime error with no matrix information
inline void ya_runtime_error(const string message);
inline void ya_debug_error(bool condition, const string message) {
  if (!condition) ya_runtime_error(message);
}

// Generate a runtime error with information on 1 matrix
template <class vmtype>
void ya_runtime_error(const string message,const vmtype &one);
template <class vmtype>
inline void ya_debug_error(bool condition, const string message,
                            const vmtype &one) {
  if (!condition) ya_runtime_error(message,one);
}

// Generate a runtime indexing error (1D index)
template <class vmtype>
void ya_runtime_error(const string message,const vmtype &one, const ya_sizet i);
template <class vmtype>
inline void ya_debug_error(bool condition, const string message,
                           const vmtype &one, const ya_sizet i) {
  if (!condition) ya_runtime_error(message,one,i);
}

// Generate a runtime indexing error (2D index)
template <class vmtype>
void ya_runtime_error(const string message,
                      const vmtype &one, const ya_sizet i, const ya_sizet j);
template <class vmtype>
inline void ya_debug_error(bool condition, const string message,
                           const vmtype &one, const ya_sizet i,
                           const ya_sizet j) {
  if (!condition) ya_runtime_error(message,one,i,j);
}

// Generate a runtime error with information on 2 matrices
template <class vmtype, class vmtype2>
void ya_runtime_error2(const string message, const vmtype &one, 
                       const vmtype2 &two);
template <class vmtype, class vmtype2>
inline void ya_debug_error2(bool condition, const string message,
                            const vmtype &one, const vmtype2 &two) {
  if (!condition) ya_runtime_error2(message,one,two);
}

// Generate a runtime error with information on 1 matrix
template <class vmtype>
void ya_runtime_error(const string message,const vmtype &one) {
  ya_runtime_error(message);
  cerr << "Matrix has a size of: " << one.rows() << " , " << one.cols();
  if (one.numel()!=0) {
    ya_sizet column_out=3;
    if (one.cols()<3)
      column_out=one.cols();
    ya_sizet row_out=3;
    if (one.rows()<3)
      row_out=one.rows();
    std::cerr << " and first " << row_out << "x" << column_out << " elements:\n";
    for (ya_sizet i=0; i<row_out; i++) {
      for (ya_sizet j=0; j<column_out; j++)
        std::cerr << one(i,j) << " ";
      std::cerr << std::endl;
    }
  }
  else
    std::cerr << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
}

// Generate a runtime indexing error (1D index)
template <class vmtype>
void ya_runtime_error(const string message,const vmtype &one, const ya_sizet i){
  ya_runtime_error(message,one);
  std::cerr << "  Attempted to access index: " << i << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
}

// Generate a runtime indexing error (2D index)
template <class vmtype>
void ya_runtime_error(const string message,
                      const vmtype &one, const ya_sizet i, const ya_sizet j) {
  ya_runtime_error(message,one);
  std::cerr << "  Attempted to access index: " << i << " , " << j << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
}

// Generate a runtime error with information on 2 matrices
template <class vmtype, class vmtype2>
void ya_runtime_error2(const string message,
                       const vmtype &two, const vmtype2 &one) {
  ya_runtime_error(message,two);
  std::cerr << "Matrix 2 has a size of: " << one.rows() << " , " << one.cols();
  if (one.numel()!=0) {
    ya_sizet column_out=3;
    if (one.cols()<3)
      column_out=one.cols();
    ya_sizet row_out=3;
    if (one.rows()<3)
      row_out=one.rows();
    std::cerr << " and first " << row_out << "x" << column_out << " elements:\n";
    for (ya_sizet i=0; i<row_out; i++) {
      for (ya_sizet j=0; j<column_out; j++)
        std::cerr << one(i,j) << " ";
      std::cerr << std::endl;
    }
  }
  else
    std::cerr << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
}

// Generate a runtime error with no matrix information
void ya_runtime_error(const string message) {
  std::cerr << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
  std::cerr << "       Run-time developer error occured in YALA++\n";
  std::cerr << "-----------------------------------------------------------------\n";
  std::cerr << message << std::endl;
  std::cerr << "-----------------------------------------------------------------\n";
}  
 
#else

inline void ya_no_debug() {}
#define YA_DEBUG_ERROR(a,b) ya_no_debug();
#define YA_DEBUG_ERROR1(a,b,c) ya_no_debug();
#define YA_INDEX_ERROR(a,b,c,d) ya_no_debug();
#define YA_INDEX2_ERROR(a,b,c,d,e) ya_no_debug();
#define YA_DEBUG_ERROR2(a,b,c,d) ya_no_debug();

#endif

}
#endif
