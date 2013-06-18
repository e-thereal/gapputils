/***************************************************************************
                                  ya_rows.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix with fixed number of rows stored on heap
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

#ifndef YA_ROWS_H
#define YA_ROWS_H

YALANAMESPACE

/// Vector/Matrix with fixed number of rows
/** Both vectors and matrices are stored in a 1D dynamically allocated array.
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap \e Column-major
  * - Size Specificaiton: \e Run-time/Compile-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e Yes
  * - BLAS: \e Yes \e (Default)
  * - Best Access: \e 1D \e Column-Major - \e Matrices; \e Any - \e Vectors
  **/
template<class eltype, ya_sizet nrows>
class YA_Rows : public YA_Common<eltype, YA_RowsT> {
 public:
  typedef typename YA_Traits<YA_RowsT>::ref ref;
  typedef typename YA_Traits<YA_RowsT>::cref cref;
  typedef typename YA_Traits<YA_RowsT>::iter iter;
  typedef typename YA_Traits<YA_RowsT>::citer citer;

  using YA_Common<eltype,YA_RowsT>::operator();
  using YA_Common<eltype,YA_RowsT>::operator=;

  /// Empty 1x0 matrix
  YA_Rows()  : array(NULL), ncols(0) {}

  /// Initialize a column vector
  YA_Rows(const ya_sizet len) : ncols(len) { 
    YA_DEBUG_ERROR(ncols==1,"Cannot setup this fixed size matrix as vector.");
    ya_reserve_memory(array,len); }
    
  /// Initialize a matrix
  YA_Rows(const ya_sizet rowsi, const ya_sizet colsi) : ncols(colsi) { 
    YA_DEBUG_ERROR(nrows==rowsi,"Cannot setup this fixed size matrix as vector.");
    ya_reserve_memory(array,this->numel()); 
  }

  /// Copy from another YA_Rows
  YA_Rows(const YA_Rows &v) : ncols(v.ncols) { 
    YA_DEBUG_ERROR(ncols==v.cols(),"Cannot change # of rows in YA_Rows");
    ya_reserve_memory(array,this->numel()); ya_copy_matrix(*this,v); }

  /// Assign from string
  YA_Rows(const std::string &v) : array(NULL), ncols(0) { *this=v; }
  /// Assign from string
  YA_Rows(const char *v) : array(NULL), ncols(0) { *this=v; }

  /// Copy from another matrix
  template <class ya_type2>
  YA_Rows(const ya_type2 &v) {
    YA_DEBUG_ERROR(nrows==v.rows() || nrows==1 && v.cols()==1,
                   "Cannot change # of rows in YA_Rows");
    if (nrows==1) ncols=v.cols()*v.rows(); else ncols=v.cols();
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
                  "Cannot reshape YA_Rows matrix to have different # rows.");
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

  inline const YA_RowsT& operator=(const YA_RowsT &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_RowsT& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  eltype *array;
  ya_sizet ncols;  // size of matrix
};

// Clear any old vector and set up a new one with 'length' elements
template<class eltype, ya_sizet nrows>
void YA_RowsT::setup(const ya_sizet len) {
  YA_DEBUG_ERROR(len!=0,"Attempt to setup matrix with 0 elements");
  YA_DEBUG_ERROR(nrows==1,"Cannot change # of rows in YA_Rows");
  if (len!=ncols) {
    if (array!=NULL) 
      delete[] array;
    ncols=len;
    ya_reserve_memory(array,len);
  }
}

// Clear any old matrix and set up a new one
template<class eltype, ya_sizet nrows>
void YA_RowsT::setup(const ya_sizet rowi, const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi!=0 && columni!=0,"Cannot setup 0 sized matrix.");
  YA_DEBUG_ERROR(nrows==rowi || (nrows==1 && columni==1),
                 "Cannot change # of rows in YA_Rows");
  if (nrows==1)
    setup(rowi*columni);
  else {
    if (columni!=ncols) {
      if (array!=NULL)
        delete[] array;
      ncols=columni;
      ya_reserve_memory(array,this->numel());
    }
  }
}

// Clear the matrix
template<class eltype, ya_sizet nrows>
void YA_RowsT::clear() {
  ya_free_memory(array);
  ncols=0;
}

}
#endif
