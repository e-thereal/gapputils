/***************************************************************************
                                 vm_concatr.h
                             -------------------
                               W. Michael Brown

  Row-wise concatenation of two matrices or vectors

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

#ifndef VM_CONCATR_H
#define VM_CONCATR_H

YALANAMESPACE

/// Row-wise ConcatRenation of two matrices
template<class eltype, class mat_type, class mat_type2>
class VecMatConcatR : public YA_WrapperCommon<eltype,VM_ConcatR> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_ConcatR>::ref ref;
  typedef typename YA_Traits<VM_ConcatR>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_ConcatR>::iter iter;
  typedef typename YA_Traits<VM_ConcatR>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatConcatR(mat_type *v, mat_type2 *v2) : parent(v), parent2(v2) {
    YA_DEBUG_ERROR2(v->cols()==v2->cols(),
                    "Number of columns must match for concatenation",*v,*v2);
    YA_DEBUG_ERROR2(v->number_type()==v2->number_type(),
       "The number type must match for concatenation. Try using vm_cast.",
       *v,*v2);
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return parent->numel()+parent2->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const {return parent->cols();}
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->rows()+parent2->rows(); }

  /// Access element in vector
  inline ref at(const ya_sizet i) const {
    div_t index=div(i,rows());
    return at2(index.rem,index.quot); }
  inline ref at(const ya_sizet i) {
    div_t index=div(i,rows());
    return at2(index.rem,index.quot); }
  inline ref at2(const ya_sizet i, const ya_sizet j) const {
     if (i<parent->rows())
      return parent->at2(i,j);
    else
      return parent2->at2(i-parent->rows(),j); }
  inline ref at2(const ya_sizet i, const ya_sizet j) {
     if (i<parent->rows())
      return parent->at2(i,j);
    else
      return parent2->at2(i-parent->rows(),j); }

  using YA_WrapperCommon<eltype,VM_ConcatR>::operator();
  using YA_WrapperCommon<eltype,VM_ConcatR>::operator=;
  inline const VM_ConcatR& operator=(const VM_ConcatR &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_ConcatR& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatConcatR();

  mat_type* const parent;
  mat_type2* const parent2;
};

}
#endif
