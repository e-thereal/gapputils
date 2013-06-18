/***************************************************************************
                               vecmatsparse.h
                             -------------------
                               W. Michael Brown

  Sparse Vector/Matrix
  Insert/Access in log(N)

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon May 24 2004
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATSPARSE_H
#define VECMATSPARSE_H
  
#include <map>
#include <algorithm>
//using namespace std;

YALANAMESPACE

// -------------------------- Sparse Matrix Zero Types
template<class eltype>
class _vm_sparse_zero { public: static const eltype zero() { return 0; } }; 
template<class eltype>
class _vm_sparse_inf { public: static const eltype zero() { return VM_INF; }}; 

template <class eltype, class mat_type> class VMSparseReturn;
template<class eltype, class mat_type>
std::ostream & operator<<(std::ostream &out, const VMSparseReturn<eltype, mat_type> &t);

// Allow setting sparse matrix elements with operator() and operator()
template <class eltype, class mat_type>
class VMSparseReturn {
 public:
  VMSparseReturn(mat_type *par,const ya_sizet i);
  operator eltype() const;
  void operator=(const eltype two) const;

  operator VMSparseReturn<eltype,const mat_type>() const {
    return VMSparseReturn<eltype,const mat_type>(parent,index);
  }

  void operator+=(const eltype two) const;
  void operator-=(const eltype two) const;
  void operator*=(const eltype two) const;
  void operator/=(const eltype two) const;
  eltype operator++(int) const;
  eltype operator--(int) const;
  VMSparseReturn<eltype,mat_type>& operator++() const;
  VMSparseReturn<eltype,mat_type>& operator--() const;

  // Write out the value
  friend std::ostream & operator<< <>(std::ostream &out,
                                 const VMSparseReturn<eltype,mat_type> &t);
 private:
   mat_type* const parent;
   const ya_sizet index;
};

/// Sparse vector/matrix
/** Insert and Retrieve occur in log(N) time
  * For a matrix, the size must be set before storing data. **/
template<class eltype, class zerotype>
class VecMatSparse : public YA_Common<eltype,VM_Sparse> {
 public:
   /// Return types for indexing into this matrix
   typedef typename YA_Traits<VM_Sparse>::ref ref;
   typedef typename YA_Traits<VM_Sparse>::cref cref;
  /// Iterator to a matrix element
  typedef typename YA_Traits<VM_Sparse>::iter iter;
  typedef typename YA_Traits<VM_Sparse>::iter citer;

  friend class VMSparseReturn<eltype,VecMatSparse<eltype> >;
  friend class VMSparseReturn<eltype,const VecMatSparse<eltype> >;
  friend std::ostream & operator<< <>(std::ostream &out,
    const VMSparseReturn<eltype,VecMatSparse<eltype> > &t);
  friend std::ostream & operator<< <>(std::ostream &out,
    const VMSparseReturn<eltype,const VecMatSparse<eltype> > &t);

  /// Empty matrix
  VecMatSparse();
  /// Copy matrix
  VecMatSparse(const VecMatSparse &v);
  /// Initialize a vector 
  /** \param zero The default value for "empty" elements. **/
  VecMatSparse(const ya_sizet length);
  /// Initialize a matrix
  /** \param zero The default value for "empty" elements. **/
  VecMatSparse(const ya_sizet rows, const ya_sizet columns);
  template <class ya_type2>
  VecMatSparse(const ya_type2 &);

  /// Return the sparse value for the matrix (usually 0)
  inline const eltype zerovalue() const { return zerotype::zero(); }

  /// Clear any old vector and set up a new one with 'length' elements
  inline void setup(const ya_sizet length);
  /// Clear any old matrix and set up a new one
  inline void setup(const ya_sizet rows, const ya_sizet columns);

  /// Clear the matrix
  inline void clear();
  /// Set each element in the matrix to zero
  inline void zeromatrix();
  /// Test if the matrix is empty
  inline bool empty();

  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }
  
  using YA_Common<eltype,VM_Sparse>::operator=;
  inline const VM_Sparse& operator=(const VM_Sparse &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Sparse& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  inline void copymatrix(const VM_Sparse &v);

  /// Vector access
  inline cref at(const ya_sizet i) const
    { return cref(this,i); }
  inline ref at(const ya_sizet i)
    { return ref(this,i); }
  using YA_Common<eltype,VM_Sparse>::operator();
  /// Access element in matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) {
    return at(col*rows()+row);
  }
  /// Access element in matrix
  inline cref at2(const ya_sizet row, const ya_sizet col) const {
    return at(col*rows()+row);
  }

  /// Write out the non-zero elements of matrix 
  /** - Elements are written column major with 1D index followed by value
    *   followed by a newline delimiter. An
    *   "END\n" is output following the last non-zero element **/
  void write_nonzero(std::ostream &out) const;
  /// Set values in matrix by reading stream
  /** - Stream should contain space delimeted 1D indices followed by values
    * - Reading continues until an "END" token is reached. **/
  int read_nonzero(std::istream &in);

  inline bool comparematrix(const VM_Sparse &v) const;

protected:
  ya_sizet ncolumns,nrows;  // size of matrix
  ya_sizet nlength;         // size of vector

  std::map<ya_sizet,eltype> sparse_array;
  typedef typename std::map<ya_sizet,eltype>::iterator Miter;
  typedef typename std::map<ya_sizet,eltype>::const_iterator MiterC;

  inline void zero(const ya_sizet i);
  inline void zero(const ya_sizet row, const ya_sizet col);
};

}
#endif
