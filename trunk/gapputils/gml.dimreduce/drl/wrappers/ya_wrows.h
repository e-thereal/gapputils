/***************************************************************************
                                  ya_wrows.h
                             -------------------
                               W. Michael Brown

  Wrapper to create a matrix from a vector of vectors that can be indexed
  with the [] operator.

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sun Jun 8 2008
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_WROWS_H
#define YA_WROWS_H

YALANAMESPACE

/***************************************************************************
  Non-fixed Specialization
 ***************************************************************************/

/// Wrapper to create a matrix from a vector of vectors
/** The type for the vectors that compose the matrix can be most
  * types that support operator[][] for indexing. This is specified by the
  * template parameter mat_type. Examples include double** and 
  * std::vector<std::vector>. If the number of columns or rows is unknown
  * at compile time, nrows or ncols should be -1 respectively
  * - Storage: \e Parent
  * - Memory: \e Parent
  * - Size Specificaiton: \e Run-time/Compile-time
  * - Element access/set: \e Parent
  * - memcpy/memcmp/memset: \e No
  * - BLAS: \e No \e (Default)
  * - Best Access: \e 2D
  **/
template<class eltype, class mat_type, int nrows, int ncols, 
         class reft, class creft>
class YA_WRows<eltype,mat_type,nrows,ncols,reft,creft> : 
public YA_WrapperCommon<eltype, YA_WRowsTU> {
 public:
  typedef typename YA_Traits<YA_WRowsTU>::ref ref;
  typedef typename YA_Traits<YA_WRowsTU>::cref cref;
  typedef typename YA_Traits<YA_WRowsTU>::iter iter;
  typedef typename YA_Traits<YA_WRowsTU>::citer citer;
  typedef typename YA_Traits<YA_WRowsTU>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WRowsTU>::operator();
  using YA_WrapperCommon<eltype,YA_WRowsTU>::operator=;

  /// Initialize a matrix
  YA_WRows(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat), nncols(colsi), nnrows(rowsi), nlength(rowsi*colsi) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_WRows
  YA_WRows(const YA_WRows &v) : array(v.array), nncols(v.cols()), 
                                nnrows(v.rows()), nlength(v.nlength) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Reshape the matrix
  inline void reshape(const ya_sizet rowsi, const ya_sizet colsi) {
    YA_DEBUG_ERROR(rowsi*colsi==nlength,
                "Cannot reshape matrix to have different number of elements.");
    YA_DEBUG_ERROR((ncols==-1 || ncols==colsi)&&(nrows==-1 || nrows==rowsi),
      "Fixed number of columns and/or rows do not match reshape");
    nnrows=rowsi; nncols=colsi;
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return nncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nnrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return nlength; }

  /// Column-major index into matrix
  inline ref at(const ya_sizet i) 
    { ya_sizet row,col; oneDtwoD(*this,i,row,col); return at2(row,col); }
  /// Column-major index into matrix
  inline ref at(const ya_sizet i) const 
    { ya_sizet row,col; oneDtwoD(*this,i,row,col); return at2(row,col); }
  /// 2D index into matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) 
    { return array[row][col]; }
  /// 2D index into matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) const 
    { return array[row][col]; }

  inline const YA_WRowsTU& operator=(const YA_WRowsTU &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WRowsTU& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_WRows();

  mat_type array;
  ya_sizet nncols,nnrows;  // size of matrix
  ya_sizet nlength;      // size of vector
};

/***************************************************************************
  Fixed-Size Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int nrows, int ncols,
         class reft, class creft>
class YA_WRows<eltype,mat_type,nrows,ncols,reft,creft,1> : 
public YA_WrapperCommon<eltype, YA_WRowsTF> {
 public:
  typedef typename YA_Traits<YA_WRowsTF>::ref ref;
  typedef typename YA_Traits<YA_WRowsTF>::cref cref;
  typedef typename YA_Traits<YA_WRowsTF>::iter iter;
  typedef typename YA_Traits<YA_WRowsTF>::citer citer;
  typedef typename YA_Traits<YA_WRowsTF>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WRowsTF>::operator();
  using YA_WrapperCommon<eltype,YA_WRowsTF>::operator=;

  /// Initialize
  YA_WRows(mat_type mat) : array(mat) {}
    
  /// Initialize a matrix
  YA_WRows(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat) {
    YA_DEBUG_ERROR((ncols==colsi)&&(nrows==rowsi),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_WRows
  YA_WRows(const YA_WRows &v) : array(v.array) {
    YA_DEBUG_ERROR((ncols==v.cols())&&(nrows==v.rows()),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return nrows*ncols; }

  /// Column-major index into matrix
  inline ref at(const ya_sizet i) 
    { ya_sizet row,col; oneDtwoD(*this,i,row,col); return at2(row,col); }
  /// Column-major index into matrix
  inline ref at(const ya_sizet i) const 
    { ya_sizet row,col; oneDtwoD(*this,i,row,col); return at2(row,col); }
  /// 2D index into matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) 
    { return array[row][col]; }
  /// 2D index into matrix
  inline ref at2(const ya_sizet row, const ya_sizet col) const 
    { return array[row][col]; }

  inline const YA_WRowsTF& operator=(const YA_WRowsTF &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WRowsTF& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_WRows();

  mat_type array;
};

}
#endif
