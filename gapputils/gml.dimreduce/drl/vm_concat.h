/***************************************************************************
                                 vm_concat.h
                             -------------------
                               W. Michael Brown

  Columnwise Concatenation of two matrices or vectors

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue May 1 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_CONCAT_H
#define VM_CONCAT_H

YALANAMESPACE

/// Columnwise concatenation of two matrices
template<class eltype, class mat_type, class mat_type2>
class VecMatConcat : public YA_WrapperCommon<eltype,VM_Concat> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Concat>::ref ref;
  typedef typename YA_Traits<VM_Concat>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_Concat>::iter iter;
  typedef typename YA_Traits<VM_Concat>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatConcat(mat_type *v, mat_type2 *v2) : parent(v), parent2(v2) {
    #ifdef YA_DEBUG
    YA_DEBUG_ERROR2(v->rows()==v2->rows(),
       "Number of rows must match for concatenation.",*v,*v2);
    YA_DEBUG_ERROR2(v->number_type()==v2->number_type(),
       "The number type must match for concatenation. Try using vm_cast.",
       *v,*v2);
    YA_DEBUG_ERROR(v->rows()*v->cols()==v->numel(),
                   "Cannot concatenate symmetric matrices. Use a wrapper.");
    YA_DEBUG_ERROR(v2->rows()*v2->cols()==v2->numel(),
                   "Cannot concatenate symmetric matrices. Use a wrapper.");
    #endif
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel()+parent2->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {return parent->cols()+parent2->cols();}
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->rows(); }

  /// Access element in vector
  inline ref at(const ya_sizet i) const {
    if (i<parent->numel())
      return parent->at(i);
    else
      return parent2->at(i-parent->numel()); }
  inline ref at(const ya_sizet i) {
    if (i<parent->numel())
      return parent->at(i);
    else
      return parent->at(i-parent->numel()); }

  using YA_WrapperCommon<eltype,VM_Concat>::operator();
  using YA_WrapperCommon<eltype,VM_Concat>::operator=;
  inline const VM_Concat& operator=(const VM_Concat &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_Concat& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatConcat();

  mat_type* const parent;
  mat_type2* const parent2;
};

}
#endif
