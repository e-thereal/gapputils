/***************************************************************************
                                vecmattrans.h
                             -------------------
                               W. Michael Brown

  Transpose of another matrix

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

#ifndef VECMATTRANS_H
#define VECMATTRANS_H

YALANAMESPACE

/// Transpose of another matrix
template<class eltype, class mat_type>
class YA_Trans : public YA_WrapperCommon<eltype,YA_TransT> {
 public:
  typedef typename YA_Traits<YA_TransT>::ref ref;
  typedef typename YA_Traits<YA_TransT>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<YA_TransT>::iter iter;
  typedef typename YA_Traits<YA_TransT>::citer citer;

  /// Create a column vector pointing to the specified column in v
  YA_Trans(mat_type *v);
  /// Copy constructor
  YA_Trans(const YA_Trans &v);

  /// True if the matrix is known at compile-time to be a vector
  inline bool vector() const { return parent->vector(); }

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {  return parent->rows(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->cols(); }

  /// Access element in vector
  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_WrapperCommon<eltype,YA_TransT>::operator();
  /// Access element in vector
  inline ref at2(const ya_sizet row, const ya_sizet col) const;
  inline ref at2(const ya_sizet row, const ya_sizet col);
  inline ref rowmajor_at(const ya_sizet i) { return parent->at(i); }
  inline ref rowmajor_at(const ya_sizet i) const { return parent->at(i); }

  inline typename YA_IterT begin() { return parent->rowbegin(); }
  inline typename YA_IterT end() { return parent->rowend(); }
  inline typename YA_ConstIterT begin() const { return parent->rowbegin(); }
  inline typename YA_ConstIterT end() const { return parent->rowend(); }
  inline typename YA_RowIterT rowbegin() { return parent->begin(); }
  inline typename YA_RowIterT rowend() { return parent->end(); }
  inline typename YA_ConstRowIterT rowbegin() const { return parent->begin();}
  inline typename YA_ConstRowIterT rowend() const { return parent->end(); }
  /// Return the pointer to the begining of the array
  inline typename YA_CptrT c_ptr() { return parent->c_ptr(); }
  /// Return the pointer to the begining of the array
  inline typename YA_CptrT c_ptr() const { return parent->c_ptr(); }


  using YA_WrapperCommon<eltype,YA_TransT>::operator=;
  inline const YA_TransT& operator=(const YA_TransT &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_TransT& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  YA_Trans();

  mat_type* const parent;
};

}
#endif
