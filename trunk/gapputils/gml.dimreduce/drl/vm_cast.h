/***************************************************************************
                                  vm_cast.h
                             -------------------
                               W. Michael Brown

  Cast one matrix type to another

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Jun 25 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_CAST_H
#define VM_CAST_H

YALANAMESPACE

/// Cast of another matrix (new rows and columns for same data)
template<class eltype, class mat_type>
class VecMatCast : public YA_WrapperCommon<eltype,VM_Cast> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Cast>::ref ref;
  typedef typename YA_Traits<VM_Cast>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Cast>::iter iter;
  typedef typename YA_Traits<VM_Cast>::citer citer;

  /// Construct a matrix with new type
  VecMatCast(mat_type *v) : parent(v) { }

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return parent->cols(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->rows(); }

  /// Access element in vector
  inline ref at (const ya_sizet i) const
    { return static_cast<eltype>(parent->at(i)); }
  inline ref at (const ya_sizet i)
    { return static_cast<eltype>(parent->at(i)); }
  using YA_WrapperCommon<eltype,VM_Cast>::operator();
  using YA_WrapperCommon<eltype,VM_Cast>::operator=;

  inline const VM_Reshape& operator=(const VM_Reshape &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }

 protected:
  // Declare away
  VecMatCast();

  mat_type* const parent;
};

/// Static Class for Casting Matrix Types
template <class newtype>
class vm_cast {
 public:
  /// Static casting to newtype
  template <class eltype, class mat_type>
  inline static const VecMatCast<newtype,mat_type> sc(YA_BaseT &mat)
    { return VecMatCast<newtype,mat_type>(&(mat.stv())); }
  /// Static casting to const newtype
  template <class eltype, class mat_type>
  inline static const VecMatCast<newtype,const mat_type> sc(const YA_BaseT &mat)
    { return VecMatCast<newtype,const mat_type>(&(mat.stv())); }
};

/// Return the scalar value from a 1 x 1 matrix
template <class eltype, class mat_type>
typename YA_Traits<mat_type>::cref scalar(const YA_BaseT &mat)
  { return mat(0); }

}
#endif
