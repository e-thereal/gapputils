/***************************************************************************
                                  ya_fix.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix as stack array
  Insert/Access in constant time

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Apr 17 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_FIX_H
#define YA_FIX_H

YALANAMESPACE

/// Vector/Matrix as stack array
/** Both vectors and matrices are stored in a 1D fixed sized array on stack.
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Stack \e Column-Major
  * - Size Specificaiton: \e Compile-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e Yes
  * - BLAS: \e Yes \e (Not Default)
  * - Best Access: \e 1D \e Column-Major - \e Matrices; \e Any - \e Vectors
  **/
template<class eltype, ya_sizet nrows, ya_sizet ncols>
class YA_Fix : public YA_Common<eltype, YA_FixT> {
 public:
  typedef typename YA_Traits<YA_FixT>::ref ref;
  typedef typename YA_Traits<YA_FixT>::cref cref;
  typedef typename YA_Traits<YA_FixT>::iter iter;
  typedef typename YA_Traits<YA_FixT>::citer citer;

  using YA_Common<eltype,YA_FixT>::operator();
  using YA_Common<eltype,YA_FixT>::operator=;

  YA_Fix() {}

  /// Copy from another YA_Fix
  YA_Fix(const YA_Fix &v) { *this=v; }
  /// Assign from string
  YA_Fix(const std::string &v) { *this=v; }
  /// Assign from string
  YA_Fix(const char *v) { *this=v; }
  /// Copy from another matrix
  template <class ya_type2>
  YA_Fix(const ya_type2 &v) { *this=v; }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Column-major index into matrix
  inline ref at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline cref at (const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline iter begin() { return array; }
  /// Return iterator past last element
  inline iter end() { return array+this->numel(); }
  /// Return iterator at first element
  inline citer begin() const { return array; }
  /// Return iterator past last element
  inline citer end() const { return array+this->numel(); }
  /// Return the pointer to the begining of the array
  inline eltype* c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline const eltype* c_ptr() const { return array; }

  inline const YA_FixT& operator=(const YA_FixT &v)
    { this->setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_FixT& operator=(const ya_type2 &v)
    { this->setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  eltype array[nrows*ncols];
};

}
#endif
