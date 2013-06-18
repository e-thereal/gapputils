/***************************************************************************
                                  ya_cols.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix with fixed number of columns stored on heap
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

#ifndef YA_COLS_H
#define YA_COLS_H

YALANAMESPACE

/// Vector/Matrix with fixed number of columns
/** Both vectors and matrices are stored in a 1D dynamically allocated array.
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap \e Column-major
  * - Size Specificaiton: \e Run-time/Compile-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e Yes
  * - BLAS: \e Yes \e (Default)
  * - Best Access: \e 1D \e Column-Major - \e Matrices; \e Any - \e Vectors
  **/
template<class eltype, ya_sizet ncols>
class YA_Cols : public YA_Common<eltype, YA_ColsT> {
 public:
  typedef typename YA_Traits<YA_ColsT>::ref ref;
  typedef typename YA_Traits<YA_ColsT>::cref cref;
  typedef typename YA_Traits<YA_ColsT>::iter iter;
  typedef typename YA_Traits<YA_ColsT>::citer citer;

  using YA_Common<eltype,YA_ColsT>::operator();
  using YA_Common<eltype,YA_ColsT>::operator=;

  /// Empty 0x1 matrix
  YA_Cols()  : array(NULL), nrows(0) {}

  /// Initialize a column vector
  YA_Cols(const ya_sizet len) : nrows(len) { 
    YA_DEBUG_ERROR(ncols==1,"Cannot setup this fixed size matrix as vector.");
    ya_reserve_memory(array,len); }
    
  /// Initialize a matrix
  YA_Cols(const ya_sizet rowsi, const ya_sizet colsi) : nrows(rowsi) { 
    YA_DEBUG_ERROR(ncols==colsi,"Cannot setup this fixed size matrix as vector.");
    ya_reserve_memory(array,this->numel()); 
  }

  /// Copy from another YA_Cols
  YA_Cols(const YA_Cols &v) : nrows(v.nrows) { 
    YA_DEBUG_ERROR(ncols==v.cols(),"Cannot change # of columns in YA_Cols");
    ya_reserve_memory(array,this->numel()); ya_copy_matrix(*this,v); }

  /// Assign from string
  YA_Cols(const std::string &v) : array(NULL), nrows(0) { *this=v; }
  /// Assign from string
  YA_Cols(const char *v) : array(NULL), nrows(0) { *this=v; }

  /// Copy from another matrix
  template <class ya_type2>
  YA_Cols(const ya_type2 &v) {
    YA_DEBUG_ERROR(ncols==v.cols() || ncols==1 && v.rows()==1,
                   "Cannot change # of columns in YA_Cols");
    if (ncols==1) nrows=v.cols()*v.rows(); else nrows=v.rows();
    ya_reserve_memory(array,this->numel()); ya_copy_matrix(*this,v); }

  // Clear any memory associated with matrix
  inline void destruct() { if (array!=NULL) delete[] array; }

  /// Clear the matrix and size
  inline void clear();

  /// Clear any old vector and set up a new one with 'length' elements
  inline void setup(const ya_sizet length);

  /// Clear any old matrix and set up a new one
  inline void setup(const ya_sizet rows, const ya_sizet columns);

  /// Reshape the matrix
  inline void reshape(const ya_sizet rowsi, const ya_sizet colsi) {
    YA_DEBUG_ERROR(rowsi==nrows && colsi==ncols,
                  "Cannot reshape YA_Cols matrix to have different # columns.");
  }

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

  inline const YA_ColsT& operator=(const YA_ColsT &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_ColsT& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  eltype *array;
  ya_sizet nrows;  // size of matrix
};

// Clear any old vector and set up a new one with 'length' elements
template<class eltype, ya_sizet ncols>
void YA_ColsT::setup(const ya_sizet len) {
  YA_DEBUG_ERROR(len!=0,"Attempt to setup matrix with 0 elements");
  YA_DEBUG_ERROR(ncols==1,"Cannot change # of columns in YA_Cols");
  if (len!=nrows) {
    if (array!=NULL) 
      delete[] array;
    nrows=len;
    ya_reserve_memory(array,len);
  }
}

// Clear any old matrix and set up a new one
template<class eltype, ya_sizet ncols>
void YA_ColsT::setup(const ya_sizet rowi, const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi!=0 && columni!=0,"Cannot setup 0 sized matrix.");
  YA_DEBUG_ERROR(ncols==columni || (ncols==1 && rowi==1),
                 "Cannot change # of columns in YA_Cols");
  if (ncols==1)
    setup(rowi*columni);
  else {
    if (rowi!=nrows) {
      if (array!=NULL)
        delete[] array;
      nrows=rowi;
      ya_reserve_memory(array,this->numel());
    }
  }
}

// Clear the matrix
template<class eltype, ya_sizet ncols>
void YA_ColsT::clear() {
  ya_free_memory(array);
  nrows=0;
}

}
#endif
