/***************************************************************************
                                  vecmatc.h
                             -------------------
                               W. Michael Brown

  Column vector of matrix
  This is a slice of another matrix

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Oct 17 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATC_H
#define VECMATC_H

YALANAMESPACE

/// Column vector of another matrix
/** This object acts as a column vector of another VecMat matrix
  * using pointers to data in the base object. Clear
  * will generate assert errors. A VecMatC object cannot be
  * used once the full matrix data has been deleted. **/
template<class eltype, class mat_type>
class VecMatC : public YA_WrapperCommon<eltype,VM_Column> {
 public:
  typedef typename YA_Traits<VM_Column>::ref ref;
  typedef typename YA_Traits<VM_Column>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Column>::iter iter;
  typedef typename YA_Traits<VM_Column>::citer citer;

  /// Create a column vector pointing to the specified column in v
  VecMatC(mat_type *v, const ya_sizet col_num);
  /// Copy constructor
  VecMatC(const VM_Column &v);

  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  inline ref rowmajor_at (const ya_sizet i) const;
  inline ref rowmajor_at (const ya_sizet i);
  using YA_WrapperCommon<eltype,VM_Column>::operator();
  using YA_WrapperCommon<eltype,VM_Column>::operator=;

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->rows(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return 1; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->rows(); }

  inline const VM_Column& operator=(const VM_Column &v) const
    { this->setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Column& operator=(const ya_type2 &v) const
    { this->setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
    
  eltype * get_array() const
    { return &(parent->stv().array[column_num*numel()]); }
  const eltype * get_const_array() const
    { return &(parent->stv().array[column_num*numel()]); }

  #include "vm_friends.h"

  /// Return iterator to begining of matrix
  inline iter begin() { return parent->begin()+column_num*numel(); }
  /// Return iterator to end of matrix
  inline iter end() { return parent->begin()+(column_num+1)*numel(); }
  /// Return iterator to begining of matrix
  inline citer begin() const { return parent->begin()+column_num*numel(); }
  /// Return iterator to end of matrix
  inline citer end() const  { return parent->begin()+(column_num+1)*numel(); }
  /// Return the pointer to the begining of the array
  inline iter c_ptr() { return begin(); }
  /// Return the pointer to the begining of the array
  inline citer c_ptr() const { return begin(); }

 protected:
  // Declare away
  VecMatC();

  mat_type* const parent;
  const ya_sizet column_num;           // size of matrix
};

}
#endif
