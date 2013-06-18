/***************************************************************************
                                  vecmats.h
                             -------------------
                               W. Michael Brown

  General Slice Matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Oct 24 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATS_H
#define VECMATS_H

YALANAMESPACE

/// General slice of another matrix
template<class eltype, class mat_type, class rows_type, class cols_type>
class VecMatS : public YA_WrapperCommon<eltype,VM_Slice> {
 public:
  typedef typename YA_Traits<VM_Slice>::ref ref;
  typedef typename YA_Traits<VM_Slice>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Slice>::iter iter;
  typedef typename YA_Traits<VM_Slice>::citer citer;

  /// Create a subset matrix pointing to v using multiple rows and columns
  /** This performs a deep copy of the row and column indices **/
  VecMatS(mat_type *v, const rows_type &rows, const cols_type &cols);
  /// Copy constructor
  VecMatS(const VM_Slice &v);

  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {  return colsi.numel(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return rowsi.numel(); }

  /// Access element in vector
  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_WrapperCommon<eltype, VM_Slice>::operator();
  using YA_WrapperCommon<eltype, VM_Slice>::operator=;
  
  /// Access element in vector
  inline ref at2(const ya_sizet row, const ya_sizet col) const;
  inline ref at2(const ya_sizet row, const ya_sizet col);

  inline const VM_Slice& operator=(const VM_Slice &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Slice& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatS();

  mat_type* const parent;
  const rows_type rowsi;
  const cols_type colsi;
  const ya_sizet nlength;
};

}
#endif
