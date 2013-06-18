/***************************************************************************
                                  ya_trans.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix Wrapper to Give Transpose of Another Matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Oct 18 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_TRANS_H
#define YA_TRANS_H

YALANAMESPACE

/// Vector/Matrix Wrapper to Give Transpose of Another Matrix
/** - Storage: \e Parent
  * - Memory: \e Parent
  * - Size Specificaiton: \e Run-time/Compile-time
  * - Element access/set: \e Transpose \e of \e Parent
  * - memcpy/memcmp/memset: \e Transpose \e of \e Parent
  * - BLAS: \e Tranpose of Parent \e (Default)
  * - Best Access: \e Tranpose \e of \e Parent
  **/
template<class eltype, class mat_type>
class YA_Trans : public YA_WrapperCommon<eltype,YA_TransT> {
 public:
  typedef typename YA_Traits<YA_TransT>::ref ref;
  typedef typename YA_Traits<YA_TransT>::cref cref;
  typedef typename YA_Traits<YA_TransT>::iter iter;
  typedef typename YA_Traits<YA_TransT>::citer citer;

  using YA_WrapperCommon<eltype,YA_Trans>::operator();
  using YA_WrapperCommon<eltype,YA_Trans>::operator=;

  /// Assign from string
  YA_Trans(mat_type *v) : parent(v) { }
  /// Assign from string
  YA_Trans(const YA_Trans &v) : parent(v.parent) {} 

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return parent->rows(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->cols(); }
  /// Get the number of elements
  inline ya_sizet numel() const { return parent->numel(); }

  /// Column-major index into matrix
  inline ref at (const ya_sizet i) { return parent->rowmajor_at(i); }
  /// Column-major index into matrix
  inline ref at (const ya_sizet i) const { return parent->rowmajor_at(i); } 

  /// 2D Index into Matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) 
    { return parent->at2(col,row); }
  /// 2D Index into Matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) const
    { return parent->at2(col,row); }

  /// Row Major Index into Matrix
  inline ref rowmajor_at(const ya_sizet i) { return parent->at(i); }
  /// Row Major Index into Matrix
  inline ref rowmajor_at(const ya_sizet i) const { return parent->at(i); }

  // ------ Stride functions for a vector wrapper that has a non-unit stride
  //        for parent
  // First accessible element using stride_at
  inline ya_sizet stride_start() const { return parent->stride_start(); }
  // Last accessible element using stride_at
  inline ya_sizet stride_end() const { return parent->stride_end(); }
  // Stride for vector
  inline ya_sizet stride() const { return parent->stride(); }
  // Index according to stride
  inline ref stride_at(const ya_sizet i) { return parent->stride_at(); }
  // Index according to stride
  inline ref stride_at(const ya_sizet i) const { return parent->stride_at(); }

  /// Access first element
  inline ref front() { return parent->front(); }
  /// Access first element
  inline ref front() const { return parent->front(); }
  /// Access last element
  inline ref back() { return parent->back(); }
  /// Access last element
  inline ref back() const { return parent->back(); }

  // -------------------------- Iterators
  /// Return iterator at first element
  inline iter begin() { return parent->rowbegin(); }
  /// Return iterator past last element
  inline iter end() { return parent->rowend(); }
  /// Return iterator at first element
  inline iter begin() const { return parent->rowbegin(); }
  /// Return iterator past last element
  inline iter end() const { return parent->rowbegin(); }
  /// Return row-major iterator at first element
  inline typename YA_RowIterT rowbegin() { return parent->begin(); }
  /// Return row-major iterator past last element
  inline typename YA_RowIterT rowend() { return parent->end(); }
  /// Return row-major iterator at first element
  inline typename YA_ConstRowIterT rowbegin() const { return parent->begin(); }
  /// Return row-major iterator past last element
  inline typename YA_ConstRowIterT rowend() const { return parent->end(); }
  // Return a pointer for BLAS access, etc.
  inline typename YA_CptrT c_ptr() { return parent->c_ptr(); }
  // Return a pointer for BLAS access, etc.
  inline typename YA_CptrT c_ptr() const { return parent->c_ptr(); }

  inline const YA_TransT& operator=(const YA_TransT &v) { parent=v.parent; }

 protected:
  YA_Trans();
  
  mat_type* const parent;
};

}
#endif
