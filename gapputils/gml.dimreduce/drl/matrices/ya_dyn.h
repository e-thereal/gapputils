/***************************************************************************
                                  ya_dyn.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix as dynamic array
  Insert/Access in constant time

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Nov 14 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_DYN_H
#define YA_DYN_H

YALANAMESPACE

/// Vector/Matrix as dynamic array
/** Both vectors and matrices are stored as a 1D STL vector.
  * - Storage: \e STL \e 1-Dimensional \e vector
  * - Memory: \e Heap \e Column-Major
  * - Size Specificaiton: \e Run-time
  * - Element access/set: \e O(1)
  * - memcpy/memcmp/memset: \e No
  * - BLAS: \e No
  * - Best Access: \e 1D \e Column-Major
  **/
template<class eltype>
class YA_Dyn : public YA_Common<eltype, YA_DynT> {
 public:
  typedef typename YA_Traits<YA_DynT>::ref ref;
  typedef typename YA_Traits<YA_DynT>::cref cref;
  typedef typename YA_Traits<YA_DynT>::iter iter;
  typedef typename YA_Traits<YA_DynT>::citer citer;

  using YA_Common<eltype,YA_DynT>::operator();
  using YA_Common<eltype,YA_DynT>::operator=;

  /// Empty 0x1 matrix
  YA_Dyn() : ncols(1), nrows(0) {}

  /// Initialize a column vector
  YA_Dyn(const ya_sizet len) : ncols(1), nrows(len)
    { array.assign(len,eltype()); }
    
  /// Initialize a matrix
  YA_Dyn(const ya_sizet rowsi, const ya_sizet colsi) : ncols(colsi), 
    nrows(rowsi) { array.assign(ncols*nrows,eltype()); }

  /// Copy from another YA_Dyn
  YA_Dyn(const YA_Dyn &v) : ncols(v.ncols), nrows(v.nrows)
    { array=v.array; }

  /// Assign from string
  YA_Dyn(const std::string &v) { *this=v; }
  /// Assign from string
  YA_Dyn(const char *v) { *this=v; }

  /// Copy from another matrix
  template <class ya_type2>
  YA_Dyn(const ya_type2 &v) { *this=v; }
  
  /// Clear the matrix and size
  inline void clear();

  /// Clear any old vector and set up a new one with 'length' elements
  inline void setup(const ya_sizet length);

  /// Clear any old matrix and set up a new one
  inline void setup(const ya_sizet rows, const ya_sizet columns);

  /// Reshape the matrix
  inline void reshape(const ya_sizet rowsi, const ya_sizet colsi) {
    YA_DEBUG_ERROR(rowsi*colsi==numel(),
                "Cannot reshape matrix to have different number of elements.");
    nrows=rowsi; ncols=colsi;
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return array.size(); }

  /// Column-major index into matrix
  inline ref at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline cref at (const ya_sizet i) const 
    { return *(const_cast<eltype*>(&array[i])); }

  /// Access first element
  inline ref front() { return array.front(); }
  /// Access first element
  inline cref front() const { return *(const_cast<eltype*>(array.front())); }
  /// Access last element
  inline ref back() { return array.back(); }
  /// Access last element
  inline cref back() const { return *(const_cast<eltype*>(array.back())); }

  /// Return iterator at first element
  inline iter begin() { return array.begin(); }
  /// Return iterator past last element
  inline iter end() { return array.end(); }
  /// Return iterator at first element
  inline citer begin() const { return array.begin(); }
  /// Return iterator past last element
  inline citer end() const { return array.end(); }

  inline const YA_DynT& operator=(const YA_DynT &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_DynT& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
  /// Add an element to the end of a vector and increase the length by 1
  inline void push_back(const eltype v) {
    YA_DEBUG_ERROR(ncols==1,
                   "Temporarily, can only use push_back on column vector.");
    array.push_back(v); this->nrows++;
  }

  // Helper functions for memory routines
  inline bool comparematrix(const YA_DynT &v) const 
    { return (this->array==v.array); }
  inline void copymatrix(const YA_Dyn &v) { this->array=v.array; }
  inline void zeromatrix() { array.assign(numel(),0); }

 protected:
  std::vector<eltype> array;
  ya_sizet ncols,nrows;  // size of matrix
};


// Clear any old vector and set up a new one with 'length' elements
template<class eltype>
void YA_DynT::setup(const ya_sizet len) {
  ncols=1;
  nrows=len;
  if (len!=numel())
    array.assign(len,eltype());
}

// Clear any old matrix and set up a new one
template<class eltype>
void YA_DynT::setup(const ya_sizet rowi, const ya_sizet columni) {
  nrows=rowi;
  ncols=columni;
  ya_sizet newlength=rowi*columni;
  if (newlength!=numel())
    array.assign(newlength,eltype());
}

// Clear the matrix
template<class eltype>
void YA_DynT::clear() {
  array.clear();
  ncols=1;
  nrows=0;
}

}
#endif
