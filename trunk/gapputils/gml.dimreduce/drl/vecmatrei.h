/***************************************************************************
                                vecmatrei.h
                             -------------------
                               W. Michael Brown

   Reindexing of another matrix
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Mar 23 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATREI_H
#define VECMATREI_H

YALANAMESPACE

/// Reindex of another matrix
template<class eltype, class mat_type, class mat_type2>
class VecMatReI : public YA_WrapperCommon<eltype,VM_ReI> {
 public:
  /// Reindex the elements in v based on the indices in ind
  VecMatReI(mat_type* v, const mat_type2 *ind);
  /// Copy constructor
  VecMatReI(const VecMatReI &v);

  typedef typename YA_Traits<VM_ReI>::ref ref;
  typedef typename YA_Traits<VM_ReI>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_ReI>::iter iter;
  typedef typename YA_Traits<VM_ReI>::citer citer;

  /// Get the length of a vector
  inline ya_sizet numel() const { return indices->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {  return indices->cols(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return indices->rows(); }

  /// Access element in vector
  inline ref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_WrapperCommon<eltype,VM_ReI>::operator();
  using YA_WrapperCommon<eltype,VM_ReI>::operator=;
  
  /// Access element in vector
  inline ref at2(const ya_sizet row, const ya_sizet col) const;
  inline ref at2(const ya_sizet row, const ya_sizet col);

  inline const VM_ReI& operator=(const VM_ReI &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_ReI& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatReI();

  mat_type * const parent;
  const mat_type2 * const indices;
};

}
#endif
