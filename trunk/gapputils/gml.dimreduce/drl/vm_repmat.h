/***************************************************************************
                                 vm_repmat.h
                             -------------------
                               W. Michael Brown

  Tile of a matrix or vector

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sun Apr 8 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_REPMAT_H
#define VM_REPMAT_H

YALANAMESPACE

/// Tiling of another matrix (new rows and columns for same data)
template<class eltype, class mat_type>
class VecMatRepMat : public YA_WrapperCommon<eltype,VM_RepMat> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_RepMat>::ref ref;
  typedef typename YA_Traits<VM_RepMat>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_RepMat>::iter iter;
  typedef typename YA_Traits<VM_RepMat>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatRepMat(mat_type *v, const ya_sizet rowsi, const ya_sizet colsi) :
               parent(v), nrows(rowsi), ncolumns(colsi), nlength(rowsi*colsi) {
    YA_DEBUG_ERROR(v->rows()>0 && v->cols()>0 && rowsi>0 && colsi>0,
                   "Cannot tile a matrix 0 times.");
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline ref at(const ya_sizet i) const {
    ya_sizet row,col;
    oneDtwoD(*this,i,row,col);
    return at2(row,col); }
  inline ref at(const ya_sizet i) {
    ya_sizet row,col;
    oneDtwoD(*this,i,row,col);
    return at2(row,col); }
  inline ref at2(const ya_sizet i, const ya_sizet j) const {
    return parent->at2(i%parent->rows(),j%parent->cols()); }
  inline ref at2(const ya_sizet i, const ya_sizet j) {
    return parent->at2(i%parent->rows(),j%parent->cols()); }

  using YA_WrapperCommon<eltype,VM_RepMat>::operator();
  using YA_WrapperCommon<eltype,VM_RepMat>::operator=;
  inline const VM_RepMat& operator=(const VM_RepMat &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_RepMat& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatRepMat();

  mat_type* const parent;
  const ya_sizet nrows;
  const ya_sizet ncolumns;
  const ya_sizet nlength;
};

/// Row Tiling of another matrix (new rows for same data)
template<class eltype, class mat_type>
class VecMatRowRep : public YA_WrapperCommon<eltype,VM_RowRep> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_RowRep>::ref ref;
  typedef typename YA_Traits<VM_RowRep>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_RowRep>::iter iter;
  typedef typename YA_Traits<VM_RowRep>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatRowRep(mat_type *v, const ya_sizet rowsi) :
               parent(v), nrows(rowsi) {
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return nrows*parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return parent->numel(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline ref at(const ya_sizet i) const {
    ya_sizet row,col;
    oneDtwoD(*this,i,row,col);
    return parent->at(col); }
  inline ref at(const ya_sizet i) {
    ya_sizet row,col;
    oneDtwoD(*this,i,row,col);
    return parent->at(col); }
  inline ref at2(const ya_sizet i, const ya_sizet j) const {
    return parent->at(j); }
  inline ref at2(const ya_sizet i, const ya_sizet j) {
    return parent->at(j); }

  using YA_WrapperCommon<eltype,VM_RowRep>::operator();
  using YA_WrapperCommon<eltype,VM_RowRep>::operator=;
  inline const VM_RowRep& operator=(const VM_RowRep &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_RowRep& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatRowRep();

  mat_type* const parent;
  const ya_sizet nrows;
};

/// Column Tiling of another matrix (new rows for same data)
template<class eltype, class mat_type>
class VecMatColRep : public YA_WrapperCommon<eltype,VM_ColRep> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_ColRep>::ref ref;
  typedef typename YA_Traits<VM_ColRep>::cref cref;
  /// The iterator class for this type of matrix
  typedef typename YA_Traits<VM_ColRep>::iter iter;
  typedef typename YA_Traits<VM_ColRep>::citer citer;

  /// Construct a shallow matrix with new size
  VecMatColRep(mat_type *v, const ya_sizet colsi) :
               parent(v), ncolumns(colsi) {
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return ncolumns*parent->numel(); }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return parent->numel(); }

  /// Access element in vector
  inline ref at(const ya_sizet i) const {
    return parent->at(i%parent->numel()); }
  inline ref at(const ya_sizet i) {
    return parent->at(i%parent->numel()); }
  inline ref at2(const ya_sizet i, const ya_sizet j) const {
    return parent->at(i); }
  inline ref at2(const ya_sizet i, const ya_sizet j) {
    return parent->at(i); }

  using YA_WrapperCommon<eltype,VM_ColRep>::operator();
  using YA_WrapperCommon<eltype,VM_ColRep>::operator=;
  inline const VM_ColRep& operator=(const VM_ColRep &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const VM_ColRep& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif

 protected:
  // Declare away
  VecMatColRep();

  mat_type* const parent;
  const ya_sizet ncolumns;
};

}
#endif
