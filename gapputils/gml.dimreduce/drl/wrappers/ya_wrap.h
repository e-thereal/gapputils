/***************************************************************************
                                  ya_wrap.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix Wrapper to Create Slices of YALA++ matrices or allow
  for compatability of other matrix types with YALA++

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

#ifndef YA_WRAP_H
#define YA_WRAP_H

YALANAMESPACE

/***************************************************************************
  Non-rowmajor non-fixed Specialization
 ***************************************************************************/


/// Vector/Matrix Wrapper to Create Slices of YALA++ matrices or allow ...
/** ... for compatability of other matrix types with YALA++
  * Wrapper for any matrix type that supports operator[] for indexing.
  * The matrix type is the template parameter mat_type. Examples
  * include double* and std::vector. The order is 0 for column major
  * and 1 for row major. If the number of columns or rows is unknown
  * at compile time, nrows or ncols should be -1 respectively
  * - Storage: \e Parent
  * - Memory: \e Parent
  * - Size Specificaiton: \e Run-time/Compile-time
  * - Element access/set: \e Parent
  * - memcpy/memcmp/memset: \e Parent
  * - BLAS: \e Parent \e (Default)
  * - Best Access: \e Parent
  **/
template<class eltype, class mat_type, int order, int nrows, int ncols, 
         class reft, class creft>
class YA_Wrap<eltype,mat_type,order,nrows,ncols,reft,creft> : 
public YA_WrapperCommon<eltype, YA_WrapC> {
 public:
  typedef typename YA_Traits<YA_WrapC>::ref ref;
  typedef typename YA_Traits<YA_WrapC>::cref cref;
  typedef typename YA_Traits<YA_WrapC>::iter iter;
  typedef typename YA_Traits<YA_WrapC>::citer citer;
  typedef typename YA_Traits<YA_WrapC>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapC>::operator();
  using YA_WrapperCommon<eltype,YA_WrapC>::operator=;

  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len) : array(mat), nncols(1), 
                                              nnrows(len), nlength(len) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat), nncols(colsi), nnrows(rowsi), nlength(rowsi*colsi) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array), nncols(v.cols()), 
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
  inline ref at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline ref at (const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline iter begin() 
    { return YA_TITER<eltype,mat_type,0,YA_WrapC>::begin(*this); }
  /// Return iterator past last element
  inline iter end() 
    { return begin()+numel(); }
  /// Return iterator at first element
  inline iter begin() const 
    { return YA_TITER<eltype,mat_type,0,YA_WrapC>::begin(*this); }
  /// Return iterator past last element
  inline iter end() const
    { return begin()+numel(); }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapC& operator=(const YA_WrapC &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapC& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
  ya_sizet nncols,nnrows;  // size of matrix
  ya_sizet nlength;      // size of vector
};

/***************************************************************************
  Fixed-Size Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int order, int nrows, int ncols,
         class reft, class creft>
class YA_Wrap<eltype,mat_type,order,nrows,ncols,reft,creft,1> : 
public YA_WrapperCommon<eltype, YA_WrapCF> {
 public:
  typedef typename YA_Traits<YA_WrapCF>::ref ref;
  typedef typename YA_Traits<YA_WrapCF>::cref cref;
  typedef typename YA_Traits<YA_WrapCF>::iter iter;
  typedef typename YA_Traits<YA_WrapCF>::citer citer;
  typedef typename YA_Traits<YA_WrapCF>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapCF>::operator();
  using YA_WrapperCommon<eltype,YA_WrapCF>::operator=;

  /// Initialize
  YA_Wrap(mat_type mat) : array(mat) {}
    
  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len) : array(mat) {
    YA_DEBUG_ERROR((ncols==1)&&(nrows==len),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat) {
    YA_DEBUG_ERROR((ncols==colsi)&&(nrows==rowsi),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array) {
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
  inline ref at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline ref at (const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline iter begin() 
    { return YA_TITER<eltype,mat_type,0,YA_WrapCF>::begin(*this); }
  /// Return iterator past last element
  inline iter end() 
    { return begin()+numel(); }
  /// Return iterator at first element
  inline iter begin() const 
    { return YA_TITER<eltype,mat_type,0,YA_WrapCF>::begin(*this); }
  /// Return iterator past last element
  inline iter end() const
    { return begin()+numel(); }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapCF& operator=(const YA_WrapCF &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapCF& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
};

/***************************************************************************
  Rowmajor Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int nrows, int ncols, 
         class reft, class creft>
class YA_Wrap<eltype,mat_type,1,nrows,ncols,reft,creft> : 
public YA_WrapperCommon<eltype, YA_WrapR> {
 public:
  typedef typename YA_Traits<YA_WrapR>::ref ref;
  typedef typename YA_Traits<YA_WrapR>::cref cref;
  typedef typename YA_Traits<YA_WrapR>::iter iter;
  typedef typename YA_Traits<YA_WrapR>::citer citer;
  typedef typename YA_Traits<YA_WrapR>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapR>::operator();
  using YA_WrapperCommon<eltype,YA_WrapR>::operator=;

  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len) : array(mat), nncols(1), 
                                              nnrows(len), nlength(len) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat), nncols(colsi), nnrows(rowsi), nlength(rowsi*colsi) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array), nncols(v.cols()), 
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
  inline ref rowmajor_at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline ref rowmajor_at (const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowbegin() 
    { return YA_TITER<eltype,mat_type,1,YA_WrapRF>::rowbegin(*this); }
  /// Return iterator past last element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowend() 
    { return rowbegin()+numel(); }
  /// Return iterator at first element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowbegin() const 
    { return YA_TITER<eltype,mat_type,1,YA_WrapRF>::rowbegin(*this); }
  /// Return iterator past last element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowend() const
    { return rowbegin()+numel(); }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapR& operator=(const YA_WrapR &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapR& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
  ya_sizet nncols,nnrows;  // size of matrix
  ya_sizet nlength;      // size of vector
};

/***************************************************************************
  Row-Major Fixed-Size Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int nrows, int ncols, 
         class reft, class creft>
class YA_Wrap<eltype,mat_type,1,nrows,ncols,reft,creft,1> : 
public YA_WrapperCommon<eltype, YA_WrapRF> {
 public:
  typedef typename YA_Traits<YA_WrapRF>::ref ref;
  typedef typename YA_Traits<YA_WrapRF>::cref cref;
  typedef typename YA_Traits<YA_WrapRF>::iter iter;
  typedef typename YA_Traits<YA_WrapRF>::citer citer;
  typedef typename YA_Traits<YA_WrapRF>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapRF>::operator();
  using YA_WrapperCommon<eltype,YA_WrapRF>::operator=;

  /// Initialize
  YA_Wrap(mat_type mat) : array(mat) {}
    
  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len) : array(mat) {
    YA_DEBUG_ERROR((ncols==1)&&(nrows==len),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi) : 
    array(mat) {
    YA_DEBUG_ERROR((ncols==colsi)&&(nrows==rowsi),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array) {
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
  inline ref rowmajor_at (const ya_sizet i) { return array[i]; }
  /// Column-major index into matrix
  inline ref rowmajor_at (const ya_sizet i) const { return array[i]; }

  /// Return iterator at first element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowbegin() 
    { return YA_TITER<eltype,mat_type,1,YA_WrapRF>::rowbegin(*this); }
  /// Return iterator past last element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowend() 
    { return rowbegin()+numel(); }
  /// Return iterator at first element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowbegin() const 
    { return YA_TITER<eltype,mat_type,1,YA_WrapRF>::rowbegin(*this); }
  /// Return iterator past last element
  inline typename YA_Traits<YA_WrapRF>::rowiter rowend() const
    { return rowbegin()+numel(); }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapRF& operator=(const YA_WrapRF &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapRF& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
};

/***************************************************************************
  Non-unit stride Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int nrows, int ncols, 
         class reft, class creft>
class YA_Wrap<eltype,mat_type,3,nrows,ncols,reft,creft,0> : 
public YA_WrapperCommon<eltype, YA_WrapS> {
 public:
  typedef typename YA_Traits<YA_WrapS>::ref ref;
  typedef typename YA_Traits<YA_WrapS>::cref cref;
  typedef typename YA_Traits<YA_WrapS>::iter iter;
  typedef typename YA_Traits<YA_WrapS>::citer citer;
  typedef typename YA_Traits<YA_WrapS>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapS>::operator();
  using YA_WrapperCommon<eltype,YA_WrapS>::operator=;

  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len, const ya_sizet stride) : array(mat), nncols(1), 
                                              nnrows(len), nstride(stride) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi, const ya_sizet stride) : 
    array(mat), nncols(colsi), nnrows(rowsi), nstride(stride) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array), nncols(v.cols()), 
                              nnrows(v.rows()), nstride(v.nstride) {
    YA_DEBUG_ERROR((ncols==-1 || nncols==ncols)&&(nrows==-1 || nnrows==nrows),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return nncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nnrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return nncols*nnrows; }

  /// Column-major index into matrix
  inline ref at (const ya_sizet i) { return array[i*nstride]; }
  /// Column-major index into matrix
  inline ref at (const ya_sizet i) const { return array[i*nstride]; }

  // ------ Stride functions for a vector wrapper that has a non-unit stride
  //        for parent
  // First accessible element using stride_at
  inline ya_sizet stride_start() const { return 0; }
  // Last accessible element using stride_at
  inline ya_sizet stride_end() const { return numel()*nstride; }
  // Stride for vector
  inline ya_sizet stride() const { return nstride; }
  // Index according to stride
  inline ref stride_at(const ya_sizet i) { return array[i]; }
  // Index according to stride
  inline cref stride_at(const ya_sizet i) const { return array[i]; }

  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapS& operator=(const YA_WrapS &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapS& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
  ya_sizet nncols,nnrows;  // size of matrix
  ya_sizet nstride;
};

/***************************************************************************
  Non-unit stride Fixed-Size Specialization
 ***************************************************************************/

template<class eltype, class mat_type, int nrows, int ncols, 
         class reft, class creft>
class YA_Wrap<eltype,mat_type,3,nrows,ncols,reft,creft,1> : 
public YA_WrapperCommon<eltype, YA_WrapSF> {
 public:
  typedef typename YA_Traits<YA_WrapSF>::ref ref;
  typedef typename YA_Traits<YA_WrapSF>::cref cref;
  typedef typename YA_Traits<YA_WrapSF>::iter iter;
  typedef typename YA_Traits<YA_WrapSF>::citer citer;
  typedef typename YA_Traits<YA_WrapSF>::cptr cptr;

  using YA_WrapperCommon<eltype,YA_WrapSF>::operator();
  using YA_WrapperCommon<eltype,YA_WrapSF>::operator=;

  /// Initialize
  YA_Wrap(mat_type mat, const ya_sizet stride) : array(mat), nstride(stride){}
    
  /// Initialize a column vector
  YA_Wrap(mat_type mat, const ya_sizet len,
          const ya_sizet stride) : array(mat), nstride(stride) {
    YA_DEBUG_ERROR((ncols==1)&&(nrows==len),
      "Fixed number of columns and/or rows do not match constructor");
  }
    
  /// Initialize a matrix
  YA_Wrap(mat_type mat, const ya_sizet rowsi, const ya_sizet colsi, 
          const ya_sizet stride) : array(mat), nstride(stride) {
    YA_DEBUG_ERROR((ncols==colsi)&&(nrows==rowsi),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Copy from another YA_Wrap
  YA_Wrap(const YA_Wrap &v) : array(v.array), nstride(v.nstride) {
    YA_DEBUG_ERROR((ncols==v.cols())&&(nrows==v.rows()),
      "Fixed number of columns and/or rows do not match constructor");
  }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncols; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }
  /// Get the number of elements
  inline ya_sizet numel() const { return ncols*nrows; }

  /// Column-major index into matrix
  inline ref at (const ya_sizet i) { return array[i*nstride]; }
  /// Column-major index into matrix
  inline ref at (const ya_sizet i) const { return array[i*nstride]; }

  // ------ Stride functions for a vector wrapper that has a non-unit stride
  //        for parent
  // First accessible element using stride_at
  inline ya_sizet stride_start() const { return 0; }
  // Last accessible element using stride_at
  inline ya_sizet stride_end() const { return numel()*nstride; }
  // Stride for vector
  inline ya_sizet stride() const { return nstride; }
  // Index according to stride
  inline ref stride_at(const ya_sizet i) { return array[i]; }
  // Index according to stride
  inline cref stride_at(const ya_sizet i) const { return array[i]; }

  /// Return the pointer to the begining of the array
  inline cptr c_ptr() { return array; }
  /// Return the pointer to the begining of the array
  inline cptr c_ptr() const { return array; }

  inline const YA_WrapSF& operator=(const YA_WrapSF &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #ifdef _MSC_VER
  template<class ya_type2>
  inline const YA_WrapSF& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  #endif
  
 protected:
  YA_Wrap();

  mat_type array;
  ya_sizet nstride;
};

}
#endif
