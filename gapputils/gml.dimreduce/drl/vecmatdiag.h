/***************************************************************************
                                vecmatdiag.h
                             -------------------
                               W. Michael Brown

  Create diagonal matrix from a vector

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Apr 17 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATDIAG_H
#define VECMATDIAG_H

YALANAMESPACE

/// Create diagonal matrix wrapper from a vector
template<class eltype, class mat_type>
class VecMatDiag : public YA_WrapperCommon<eltype,VM_Diag> {
 public:
  typedef typename YA_Traits<VM_Diag>::ref ref;
  typedef typename YA_Traits<VM_Diag>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Diag>::iter iter;
  typedef typename YA_Traits<VM_Diag>::citer citer;

  /// Create a column vector pointing to the specified column in v
  VecMatDiag(mat_type *v);
  /// Copy constructor
  VecMatDiag(const VecMatDiag &v);

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {  return parent->numel(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->numel(); }

  /// Access element in vector
  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_WrapperCommon<eltype,VM_Diag>::operator();
  /// Access element in vector
  inline ref at2(const ya_sizet row, const ya_sizet col) const;
  inline ref at2(const ya_sizet row, const ya_sizet col);

  using YA_WrapperCommon<eltype,VM_Diag>::operator=;
  inline const VM_Diag& operator=(const VM_Diag &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Diag& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatDiag();

  mat_type* const parent;
  const eltype offdiag;
};

}
#endif
