/***************************************************************************
                                 vm_reshape.h
                             -------------------
                               W. Michael Brown

  Reshape of a matrix or vector

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

#ifndef VM_RESHAPE_H
#define VM_RESHAPE_H

YALANAMESPACE

/// Reshape of another matrix (new rows and columns for same data)
template<class eltype, class mat_type>
class VecMatReshape : public YA_WrapperCommon<eltype,VM_Reshape> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Reshape>::ref ref;
  typedef typename YA_Traits<VM_Reshape>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Reshape>::iter iter;
  typedef typename YA_Traits<VM_Reshape>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatReshape(mat_type *v, const ya_sizet rowsi, const ya_sizet colsi) :
                                parent(v), nrows(rowsi), ncolumns(colsi) {
    YA_DEBUG_ERROR1(v->numel()==nrows*ncolumns,
                    "Reshape length must match matrix length",*v);
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline ref at (const ya_sizet i) const
     { return parent->at(i); }
  inline ref at (const ya_sizet i) { return parent->at(i); }
  using YA_WrapperCommon<eltype,VM_Reshape>::operator();
  using YA_WrapperCommon<eltype,VM_Reshape>::operator=;

  inline const VM_Reshape& operator=(const VM_Reshape &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Reshape& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

  /// Return iterator to begining of matrix
  inline iter begin() { return parent->begin(); }
  /// Return iterator to end of matrix
  inline iter end() { return parent->end(); }
  /// Return iterator to begining of matrix
  inline citer begin() const { return parent->begin(); }
  /// Return iterator to end of matrix
  inline citer end() const { return parent->end(); }
  /// Return iterator to begining of matrix
  inline iter c_ptr() { return parent->begin(); }
  /// Return iterator to begining of matrix
  inline citer c_ptr() const { return parent->begin(); }
  
 protected:
  // Declare away
  VecMatReshape();

  mat_type* const parent;
  const ya_sizet nrows;
  const ya_sizet ncolumns;
};

}
#endif
