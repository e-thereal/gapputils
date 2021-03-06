/***************************************************************************
                                  ya_matrm.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix as normal array (row major)
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

#ifndef YA_MATRM_H
#define YA_MATRM_H

YALANAMESPACE

/// Vector/Matrix as row-major normal array
/** Both vectors and matrices are stored in a 1D dynamically allocated array.
  * - Storage: \e C++ \e 1-Dimensional \e array
  * - Memory: \e Heap \e Row-Major
  * - Size Specificaiton: \e Run-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e Yes
  * - BLAS: \e Yes \e (Default)
  * - Best Access: \e 1D \e Row-Major
  **/
template<class eltype>
class YA_MatRM : public YA_Common<eltype, YA_MatRMT> {
 public:
  typedef typename YA_Traits<YA_MatRMT>::ref ref;
  typedef typename YA_Traits<YA_MatRMT>::cref cref;
  typedef typename YA_Traits<YA_MatRMT>::iter iter;
  typedef typename YA_Traits<YA_MatRMT>::citer citer;

  using YA_Common<eltype,YA_MatRMT>::operator();
  using YA_Common<eltype,YA_MatRMT>::operator=;

  /// Empty 1x0 matrix
  YA_MatRM()  : array(NULL), ncols(0), nrows(1), nlength(0) {}

  /// Initialize a column vector
  YA_MatRM(const ya_sizet len) : ncols(1), nrows(len), nlength(len)
    { ya_reserve_memory(array,nlength); }
    
  /// Initialize a matrix
  YA_MatRM(const ya_sizet rowsi, const ya_sizet colsi) : ncols(colsi), 
    nrows(rowsi), nlength(rowsi*colsi) { ya_reserve_memory(array,nlength); }

  /// Copy from another YA_MatRM
  YA_MatRM(const YA_MatRM &v) : ncols(v.ncols), nrows(v.nrows), nlength(v.nlength)
    { ya_reserve_memory(array,nlength); ya_copy_matrix(*this,v); }

  /// Assign from string
  YA_MatRM(const std::string &v) : array(NULL), nlength(0) { *this=v; }
  /// Assign from string
  YA_MatRM(const char *v) : array(NULL), nlength(0) { *this=v; }

  /// Copy from another matrix
  template <class ya_type2>
  YA_MatRM(const ya_type2 &v) : ncols(v.cols()), nrows(v.rows()), 
                              nlength(v.rows()*v.cols()) 
    { ya_reserve_memory(array,nlength); ya_copy_matrix(*this,v); }
  
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
    YA_DEBUG_ERROR(rowsi*colsi==nlength,
                "Cannot reshape matrix to have different number of elements.");
    nrows=rowsi; ncols=colsi;
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return nlength; }

  /// Row Major Index into Matrix
  inline ref rowmajor_at(const ya_sizet i) { return array[i]; }
  /// Row Major Index into Matrix
  inline cref rowmajor_at(const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline iter begin() { return array; }
  /// Return iterator past last element
  inline iter end() { return array+numel(); }
  /// Return iterator at first element
  inline citer begin() const { return array; }
  /// Return iterator past last element
  inline citer end() const { return array+numel(); }

  /// Return row-major iterator at first element
  inline iter rowbegin() { return array; }
  /// Return row-major iterator past last element
  inline iter rowend() { return array+numel(); }
  /// Return row-major iterator at first element
  inline citer rowbegin() const { return array; }
  /// Return row-major iterator past last element
  inline citer rowend() const { return array+numel(); }

  /// Return the pointer to the begining of the array
  inline eltype* c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline const eltype* c_ptr() const { return array; }

  inline const YA_MatRMT& operator=(const YA_MatRMT &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_MatRMT& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  eltype *array;
  ya_sizet ncols,nrows;  // size of matrix
  ya_sizet nlength;      // size of vector
};


// Clear any old vector and set up a new one with 'length' elements
template<class eltype>
void YA_MatRMT::setup(const ya_sizet len) {
  YA_DEBUG_ERROR(len!=0,"Attempt to setup matrix with 0 elements");
  ncols=1;
  nrows=len;
  if (len!=numel()) {
    if (array!=NULL) 
      delete[] array;
    nlength=len;
    ya_reserve_memory(array,nlength);
  }
}

// Clear any old matrix and set up a new one
template<class eltype>
void YA_MatRMT::setup(const ya_sizet rowi, const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi!=0 && columni!=0,"Cannot setup 0 sized matrix.");
  nrows=rowi;
  ncols=columni;
  ya_sizet newlength=rowi*columni;
  if (newlength!=numel()) {
    if (array!=NULL)
      delete[] array;
    nrows=rowi;
    ncols=columni;
    nlength=newlength;
    ya_reserve_memory(array,nlength);
  }
}

// Clear the matrix
template<class eltype>
void YA_MatRMT::clear() {
  ya_free_memory(array);
  ncols=0;
  nrows=1;
  nlength=0;
}

}
#endif
