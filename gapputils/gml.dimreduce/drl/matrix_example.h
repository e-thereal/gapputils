/***************************************************************************
                               matrix_example.h
                             -------------------
                               W. Michael Brown

  Example matrix for YALA

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Sep 19 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef MATRIX_EXAMPLE_H
#define MATRIX_EXAMPLE_H

YALANAMESPACE

template<class eltype>
class YA_Traits<ExampleMatrix<eltype> > {
 public:
  // Iterators
  typedef eltype* iter; 
  typedef const eltype * citer;
  typedef YA_RowIter<eltype,YA_MatT > rowiter;
  typedef YA_ConstRowIter<eltype,const YA_MatT > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

  enum MatrixInfo {
    // ---- Memory storage of matrix
    // 0 - Other
    // 1 - C-style 1D array
    // 2 - C-style 1D array storing lower triangle of symmetric matrix
    // 3 - STL vector 1D column-major
    // 4 - STL map<index,value> storing sparse matrix column-major
    // 5 - Matrix Wrapper
    // 6 - Matrix of all zeros
    VME_MEMORY = 1,
    // ---- Known to be vector at compile time (0-no, 1-yes)
    VME_VECTOR = 0,
    // ---- Known to be column vector at compile time (0-no, 1-yes)
    VME_COL_VECTOR = 0,
    // ---- Known to be row vector at compile time (0-no, 1-yes)
    VME_ROW_VECTOR = 0,
    // ---- Known to be symmetric at compile time (0-no, 1-yes)
    VME_SYMMETRIC = 0,
    // ---- Known to be diagonal at compile time (0-no, 1-yes)
    VME_DIAGONAL = 0,
    // ---- If known at compile time, the number of columns (otherwise -1)
    VME_NCOLS = -1,
    // ---- If known at compile time, the number of rows (otherwise -1)
    VME_NROWS = -1,
    // ---- Order for indexing
    // - 0 Column major in memory
    // - 1 Row major in memory
    // - 2 Vector/Symmetric (both Column/Row Major)
    // - 3 Stride vector
    // - 4 Use 2D indexing for this matrix type
    VME_ORDER = 0,
    // ---- BLAS operations can be performed directly on begin() iterator
    // This should be set to 0 if Memory type is not 1
    VME_BLAS = _VM_BLAS_TYPE<eltype>::supported,
    // ---- BLAS operations can be performed even if the stride is not 1
    // If VME_BLAS is 1, this should also be 1.
    VME_BLAS_STRIDE = VME_BLAS,
    // ---- If an option, use BLAS on this matrix type
    // This can be set to 0 for matrices known to be small at compile-time
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    // ---- Whether or not this is a tranpose wrapper
    VME_TRANSPOSE = 0
  };
};    

/// Example Matrix Prototypes
/** Commented prototypes have a common implementation and are not
  * necesarily required 
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap \e Column-Major
  * - Size Specificaiton: \e Run-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e Yes
  * - BLAS: \e Yes \e (Default)
  * - Best Access: \e 1D \e Column-Major
  **/
template<class eltype>
class ExampleMatrix : public YA_Common<eltype, YA_Example> {
 public:
  typedef typename YA_Traits<YA_Example>::ref ref;
  typedef typename YA_Traits<YA_Example>::cref cref;
  typedef typename YA_Traits<YA_Example>::iter iter;
  typedef typename YA_Traits<YA_Example>::citer citer;

  using YA_Common<eltype,YA_Example>::operator();
  using YA_Common<eltype,YA_Example>::operator=;

  /// Assign from string
  YA_Mat(const string &v) : array(NULL), nlength(0) { *this=v; }
  /// Assign from string
  YA_Mat(const char *v) : array(NULL), nlength(0) { *this=v; }

  // Clear any memory associated with matrix
  inline void destruct();
 
  /// Clear the matrix and dynamic memory associated with it
  inline void clear();
  /// Clear any old vector and set up a new one with 'length' elements
  inline void setup(const ya_sizet length);
  /// Clear any old matrix and set up a new one
  inline void setup(const ya_sizet rows, const ya_sizet columns);
  /// Reshape the matrix
  inline void reshape(const ya_sizet nrows, const ya_sizet ncols); 

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const;
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const;
  /// Get the number of elements
  //inline ya_sizet numel() const;

  /// Column-major index into matrix
  inline YA_Traits<YA_Example>::ref at (const ya_sizet i);
  /// Column-major index into matrix
  inline YA_Traits<YA_Example>::cref at (const ya_sizet i) const;

  /// 2D Index into Matrix
  //inline YA_Traits<YA_Example>::ref at2(const ya_sizet row, const ya_sizet col);
  /// 2D Index into Matrix
  //inline YA_Traits<YA_Example>::cref at2
  //       (const ya_sizet row, const ya_sizet col) const;

  /// Row Major Index into Matrix
  //inline YA_Traits<YA_Example>::ref rowmajor_at(const ya_sizet i);
  /// Row Major Index into Matrix
  //inline YA_Traits<YA_Example>::cref rowmajor_at(const ya_sizet i) const;

  // ------ Stride functions for a vector wrapper that has a non-unit stride
  //        for parent
  // First accessible element using stride_at
  //inline ya_sizet stride_start() const;
  // Last accessible element using stride_at
  //inline ya_sizet stride_end() const;
  // Stride for vector
  //inline ya_sizet stride() const;
  // Index according to stride
  //inline ref stride_at(const ya_sizet i);
  // Index according to stride
  //inline cref stride_at(const ya_sizet i) const;

  /// Access first element
  //inline ref front();
  /// Access first element
  //inline cref front() const;
  /// Access last element
  //inline ref back();
  /// Access last element
  //inline cref back() const;

  // -------------------------- Iterators
  /// Return iterator at first element
  //inline iter begin();
  /// Return iterator past last element
  //inline iter end();
  /// Return iterator at first element
  //inline citer begin() const;
  /// Return iterator past last element
  //inline citer end() const;
  /// Return row-major iterator at first element
  //inline typename YA_RowIterT rowbegin();
  /// Return row-major iterator past last element
  //inline typename YA_RowIterT rowend();
  /// Return row-major iterator at first element
  //inline typename YA_ConstRowIterT rowbegin() const;
  /// Return row-major iterator past last element
  //inline typename YA_ConstRowIterT rowend() const;
  // Return a pointer for BLAS access, etc.
  //inline typename YA_CptrT c_ptr();
  // Return a pointer for BLAS access, etc.
  //inline typename YA_ConstCptrT c_ptr() const;

  inline const YA_ExampleT& operator=(const YA_ExampleT &v);

 protected:
  inline void destruct();
};

}
#endif
