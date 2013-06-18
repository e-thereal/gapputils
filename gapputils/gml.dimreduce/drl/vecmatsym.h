/***************************************************************************
                                 vecmatsym.h
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

#ifndef VECMATSYM_H
#define VECMATSYM_H

//using namespace std;

YALANAMESPACE

/// Symmetric Matrix Class
template<class eltype>
class VecMatSym : public YA_Common<eltype,VM_SymMat> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_SymMat>::ref ref;
  typedef typename YA_Traits<VM_SymMat>::cref cref;
  /// Iterator to a matrix element
  typedef typename YA_Traits<VM_SymMat>::iter iter;
  typedef typename YA_Traits<VM_SymMat>::citer citer;

  /// Empty matrix
  VecMatSym();
  /// Copy
  VecMatSym(const VecMatSym &v);
  /// Initialize a matrix
  /** Generates error 300 L 19 for insufficient memory **/
  VecMatSym(const ya_sizet rows, const ya_sizet columns);
  template <class ya_type2>
  VecMatSym(const ya_type2 &);

  /// Clear any old matrix and set up a new one
  /** No error checking for memory **/
  inline void setup(const ya_sizet rows, const ya_sizet columns);
  /// Clear any old matrix and set up a new one
  /** Generates error 300 L 19 for insufficient memory
      Generates error 302 L 19 if rows!=columns **/
  inline int safe_setup(const ya_sizet rows, const ya_sizet columns);
  /// 1D Setup is not allowed - This generates an error
  inline void setup(const ya_sizet length);
  
  /// Clear the matrix and size
  inline void clear();

  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return nrows; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Return the pointer to the begining of the array
  inline eltype* c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline const eltype* c_ptr() const { return array; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_Common<eltype,VM_SymMat>::operator();
  /// Access element in matrix
  inline cref at2(const ya_sizet row,
                          const ya_sizet col) const;
  inline ref at2(const ya_sizet row, const ya_sizet col);

  using YA_Common<eltype,VM_SymMat>::operator=;
  inline const VM_SymMat& operator=(const VM_SymMat &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_SymMat& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

   // Destructor (should be protected)
  inline void destruct();
  
  template<class one, ya_sizet two> friend class VecMatSymF;

protected:
  eltype *array;
  ya_sizet nrows;            // size of matrix
  ya_sizet nlength;         // size of vector

  // Reserve space for 'length' elements
  inline int reserve(const ya_sizet length);

};

}
#endif
