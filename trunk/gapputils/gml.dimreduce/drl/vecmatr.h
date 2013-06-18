/***************************************************************************
                                  vecmatr.h
                             -------------------
                               W. Michael Brown

  Row vector of matrix

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

#ifndef VECMATR_H
#define VECMATR_H

YALANAMESPACE

/// Row vector of another VecMat matrix
template<class eltype, class mat_type>
class VecMatR : public YA_WrapperCommon<eltype,VM_Row> {
 public:
  typedef typename YA_Traits<VM_Row>::ref ref;
  typedef typename YA_Traits<VM_Row>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Row>::iter iter;
  typedef typename YA_Traits<VM_Row>::citer citer;

  /// Create a column vector pointing to the specified column in v
  VecMatR(mat_type *v, const ya_sizet row_num);
  /// Copy constructor
  VecMatR(const VM_Row &v);

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->cols(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return parent->cols(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  inline ref rowmajor_at (const ya_sizet i) const;
  inline ref rowmajor_at (const ya_sizet i);
  using YA_WrapperCommon<eltype,VM_Row>::operator();

  using YA_WrapperCommon<eltype,VM_Row>::operator=;
  inline const VM_Row& operator=(const VM_Row &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Row& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatR();

  mat_type* const parent;
  const ya_sizet row_num;
};

}
#endif
