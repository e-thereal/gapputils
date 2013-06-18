/***************************************************************************
                                  ya_base.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix Base Class

  This file defines functions that are shared by all vector/matrix
  classes, but do not differ in implementation

  Static virtual functions are also declared here, but not implemented
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_BASE_H
#define YA_BASE_H
  
#include <string>
#include <iostream>
#include <fstream>
#include <float.h>
#include <cmath>
#ifdef YA_DEBUG
#include <typeinfo>
#endif
//using namespace std;

YALANAMESPACE

/// Vector Matrix Base Class Template
/** This class cannot be instantiated as object, but defines all
  * common functions for vector/matrix objects **/
template<class eltype, class mat_type>
class YA_Base {
 public:
  // Return types for indexing into this matrix
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref;
  // Iterator types for indexing into this matrix
  typedef typename YA_IterT iter;
  typedef typename YA_ConstIterT citer;
  // Wrapper return types
  typedef const VecMatC<eltype,mat_type> col_slice;
  typedef const VecMatC<eltype,const mat_type> const_col_slice;
  typedef const VecMatR<eltype,mat_type> row_slice;
  typedef const VecMatR<eltype,const mat_type> const_row_slice;
  typedef const YA_Trans<eltype,const mat_type> ttranspose;
  typedef const YA_Trans<eltype,const mat_type> const_transpose;

  ~YA_BaseT() { destruct(); }

  // Allow for static polymorphism
  inline mat_type& stv()
    { return reinterpret_cast<mat_type &>(*this); }
  // Allow for static polymorphism
  inline const mat_type& stv() const
    { return reinterpret_cast<const mat_type &>(*this); }

  /// Clear the matrix and dynamic memory associated with it
  inline void clear() { return stv().clear(); }
  /// Clear any old vector and set up a new one with 'length' elements
  /** Generates error 300 L 19 for insufficient memory **/
  inline void setup(const ya_sizet length) { return stv().setup(length); }
  /// Clear any old matrix and set up a new one
  /** Generates error 300 L 19 for insufficient memory **/
  inline void setup(const ya_sizet rows, const ya_sizet columns)
    { return stv().setup(rows,columns); }
  /// Reshape the matrix
  inline void reshape(const ya_sizet nrows, const ya_sizet ncols) 
    { stv.reshape(nrows,ncols); }

  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return stv().cols(); }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return stv().rows(); }
  /// Get the length of a vector
  inline ya_sizet numel() const { return stv().numel(); }

  /// Access element in matrix (2D)
  inline typename YA_Ref operator()(const ya_sizet row, const ya_sizet col)
    { chkind2(row,col); return stv().at2(row,col); }
  /// Access element in matrix (2D)
  inline typename YA_ConstRef operator()
        (const ya_sizet row, const ya_sizet col) const
    { chkind2(row,col); return stv().at2(row,col); }
  /// Access element in matrix (1D Column Major)
  inline typename YA_Ref operator()(const ya_sizet i)
    { chkind(i); return stv().at(i); }
  /// Access element in matrix (1D Column Major)
  inline typename YA_ConstRef operator()(const ya_sizet i) const
    { chkind(i); return stv().at(i); }
  /// Access element in matrix (1D Column Major)
  inline typename YA_Ref operator[](const ya_sizet i)
    { chkind(i); return stv().at(i); }
  /// Access element in matrix (1D Column Major)
  inline typename YA_ConstRef operator[](const ya_sizet i) const
    { chkind(i); return stv().at(i); }
  /// Access element in row major order
  inline typename YA_Ref row_at(const ya_sizet i)
    { chkind(i); return stv().rowmajor_at(i); }
  /// Access element in row major order
  inline typename YA_ConstRef row_at(const ya_sizet i) const
    { chkind(i); return stv().rowmajor_at(i); }
  /// First accessible element using stride_at
  inline ya_sizet stride_start() const { return stv().stride_start(); }
  /// Last accessible element using stride_at
  inline ya_sizet stride_end() const { return stv().strinde_end(); }
  /// Stride for vector
  inline ya_sizet stride() const { return stv().stride(); }
  /// Index according to stride
  inline typename YA_Ref stride_at(const ya_sizet i) 
    { return stv().stride_at(i); }
  /// Index according to stride
  inline typename YA_ConstRef stride_at(const ya_sizet i) const 
    { return stv().stride_at(i); }

  /// Access last 1D element
  inline typename YA_Ref back() { return stv().back(); }
  /// Access last 1D element
  inline typename YA_ConstRef back() const { return stv().back(); }
  /// Access first 1D element
  inline typename YA_Ref front() { return stv().front(); }
  /// Access first 1D element
  inline typename YA_ConstRef front() const  { return stv().front(); }

  // -------------------------- Iterators
  inline typename YA_IterT begin() { return stv().begin(); }
  inline typename YA_IterT end() { return stv().end(); }
  inline typename YA_ConstIterT begin() const { return stv().begin(); }
  inline typename YA_ConstIterT end() const { return stv().end(); }
  inline typename YA_RowIterT rowbegin() { return stv().begin(); }
  inline typename YA_RowIterT rowend() { return stv().end(); }
  inline typename YA_ConstRowIterT rowbegin() const { return stv().begin(); }
  inline typename YA_ConstRowIterT rowend() const { return stv().end(); }
  inline typename YA_CptrT c_ptr() { return stv().c_ptr(); }
  inline typename YA_ConstCptrT c_ptr() const { return stv().c_ptr(); }
    
  // -------------------------- Column Vector Slice

  /// Return a column vector slice wrapper
  /** Matlab syntax: matrix(:,col) **/
  inline const VM_Column operator() (const std::string colon, const ya_sizet col)
    { return VM_Column(&stv(),col); }
  /// Return a column vector slice wrapper
  /** Matlab syntax: matrix(:,col) **/
  inline const VecMatC<eltype, const mat_type> operator()
                               (const std::string colon, const ya_sizet col) const
    { return VecMatC<eltype,const mat_type>(&stv(),col); }
    

  // -------------------------- Row Vector Slice

  /// Return a row vector slice wrapper
  /** Matlab syntax: matrix(row,:) **/
  inline const VM_Row operator() (const ya_sizet row, const std::string colon)
    { return VM_Row(&stv(),row); }
  /// Return a row vector pointing to this matrix
  /** Matlab syntax: matrix(row,:) **/
  inline const VecMatR<eltype, const mat_type> operator()
                               (const ya_sizet row, const std::string colon) const
    { return VecMatR<eltype,const mat_type>(&stv(),row); }

  
  // -------------------------- Columns Slice
  
  /// Return a wrapper for a slice of columns
  /** Matlab syntax: matrix(:,(1 3 7)) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,mat_type,VM_ZeroToN,mat_type2>
                    operator() (const std::string colon, const YA_BaseT2T &columns)
    { return VecMatS<eltype,mat_type,VM_ZeroToN,mat_type2>(&(stv()),
                     VM_ZeroToN(rows()),columns.stv()); }
  /// Return a wrapper for a slice of columns
  /** Matlab syntax: matrix(:,(1 3 7)) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,const mat_type,VM_ZeroToN,mat_type2>
               operator() (const std::string colon, const YA_BaseT2T &columns) const
    { return VecMatS<eltype,const mat_type,VM_ZeroToN,mat_type2>(&(stv()),
                  VM_ZeroToN(rows()),columns.stv()); }


  // -------------------------- Rows Slice

  /// Return a wrapper for a slice of rows
  /** Matlab syntax: matrix((1 3 7),:) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,mat_type,mat_type2,VM_ZeroToN>
                       operator() (const YA_BaseT2T &rows, const std::string colon)
    { return VecMatS<eltype,mat_type,mat_type2,VM_ZeroToN>(&(stv()),
                                      rows.stv(),VM_ZeroToN(cols())); }
  /// Return a wrapper for a slice of rows
  /** Matlab syntax: matrix((1 3 7),:) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,const mat_type,mat_type2,VM_ZeroToN>
                   operator() (const YA_BaseT2T &rows, const std::string colon) const
    { return VecMatS<eltype,const mat_type,mat_type2,VM_ZeroToN>(&(stv()),
                                           rows.stv(),VM_ZeroToN(cols())); }

  // -------------------------- Slice of Elements on a Single Row

  /// Return a wrapper for a slice of elements on a single row
  /** Matlab syntax: matrix(row,(1 3 7)) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,mat_type,VM_ScalarU,mat_type2>
                     operator() (const ya_sizet row, const YA_BaseT2T &columns)
    { return VecMatS<eltype,mat_type,VM_ScalarU,mat_type2>(&(stv()),
                                              VM_ScalarU(row),columns.stv()); }
  /// Return a wrapper for a slice of elements on a single row
  /** Matlab syntax: matrix(row,(1 3 7)) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,const mat_type,VM_ScalarU,mat_type2>
               operator() (const ya_sizet row, const YA_BaseT2T &columns) const
    { return VecMatS<eltype,const mat_type,VM_ScalarU,mat_type2>(&(stv()),
                                       VM_ScalarU(row),columns.stv()); }


  // -------------------------- Slice of Elements on a Single Column

  /// Return a wrapper for a slice of elements on a single column
  /** Matlab syntax: matrix((1 3 7),col) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,mat_type,mat_type2,VM_ScalarU>
                       operator() (const YA_BaseT2T &rows, const ya_sizet col)
    { return VecMatS<eltype,mat_type,mat_type2,VM_ScalarU>(&(stv()),
                                               rows.stv(),VM_ScalarU(col)); }
  /// Return a wrapper for a slice of elements on a single column
  /** Matlab syntax: matrix((1 3 7),col) **/
  template <class eltype2,class mat_type2>
  inline const VecMatS<eltype,const mat_type,mat_type2,VM_ScalarU>
    operator() (const YA_BaseT2T &rows, const ya_sizet col) const
    { return VecMatS<eltype,const mat_type,mat_type2,VM_ScalarU>(&(stv()),
                                            rows.stv(),VM_ScalarU(col)); }

  // -------------------------- General slice

  /// Return a wrapper for a subset of the rows and columns in the matrix
  /** Matlab syntax: matrix((1 3 7),(1 3 7)) **/
  template <class eltype2,class mat_type2,
            class eltype3,class mat_type3>
  inline const VecMatS<eltype,mat_type,mat_type2,mat_type3> operator()
                                (const YA_BaseT2T &rows,const YA_BaseT3T &vcols)
    { return VecMatS<eltype,mat_type,mat_type2,mat_type3>
                                           (&(stv()),rows.stv(),vcols.stv()); }
  /// Return a subset of the rows and columns in the matrix
  /** Matlab syntax: matrix((1 3 7),(1 3 7)) **/
  template <class eltype2,class mat_type2,class eltype3,class mat_type3>
  inline const VecMatS<eltype,const mat_type,mat_type2,mat_type3>
               operator() (const YA_BaseT2T &rows,const YA_BaseT3T &vcols) const
    { return VecMatS<eltype,const mat_type,mat_type2,mat_type3>
                                          (&(stv()),rows.stv(),vcols.stv()); }

  // -------------------------- Reindexing Slices

  /// Return a wrapper reindexing this
  template <class eltype2, class mat_type2>
  inline const VecMatReI<eltype,mat_type,mat_type2> operator()
                                                   (const YA_BaseT2T &t)
    { return VecMatReI<eltype,mat_type,mat_type2>(&stv(),&t.stv()); }
  /// Return a wrapper reindexing this
  template <class eltype2, class mat_type2>
  inline const VecMatReI<eltype,const mat_type,mat_type2> operator()
                                                     (const YA_BaseT2T &t) const
    { return VecMatReI<eltype,const mat_type,mat_type2>(&stv(),&t.stv()); }


  // -------------------------- Transpose Wrapper

  /// Return a wrapper for the transpose of this
  inline const YA_TransT T() 
    { return YA_Trans<eltype,mat_type>(&stv()); }
  /// Return a wrapper for the transpose of this
  inline const YA_Trans<eltype,const mat_type> T() const
    { return YA_Trans<eltype,const mat_type>(&stv()); }

  // --------------------------- Assignment

  /// Set to a one by one matrix with value (if slice,set all elements to value)
  inline const YA_BaseT& operator=(const eltype value)
    { setup(1,1); (*this)(0)=value; return *this; }    
  /// Set to a matlab formatted string
  /** Generates error 307 L 11 for improperly formatted string **/
  inline const YA_BaseT& operator=(const std::string &value)
    { ya_string_to_mat<eltype>::convert(value,*this); return *this; }
  /// Set to a matlab formatted string
  /** Generates error 307 L 11 for improperly formatted string **/
  inline const YA_BaseT& operator=(const char *value)
    { return (*this)=std::string(value); }

  inline const YA_BaseT& operator=(const YA_BaseT &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }

  template <class ya_type2>
  inline const YA_BaseT& operator=(const ya_type2 &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }

  // following deleted due to "issues" with Visual Studio compilation
  // Allow assignt to a scalar from a one x one matrix
  /// Allow assignt to a scalar from a one x one matrix
  //  inline operator eltype() {
  // YA_DEBUG_ERROR(numel()==1,"Not a 1x1 matrix for scalar convert");
  // return (*this)(0); }

  // ---------------------- Elementwise operations
  /// Element-wise multiplication
  template <class mat_type2>
  const VM_EDMul dot_star(const YA_BaseT2 &two) const
    { return VM_EDMul(this->stv(),two.stv()); }
  /// Element-wise division
  template <class mat_type2>
  const VM_EDDiv dot_div(const YA_BaseT2 &two) const
    { return VM_EDDiv(this->stv(),two.stv()); }

  #ifdef YA_DEBUG
  const char * number_type() const { return typeid(eltype).name(); }
  #endif

 protected:

  YA_Base() {}
  inline void destruct() { stv().destruct(); }

  // ------------------------- Check if indices are within bounds
  void chkind(const ya_sizet i) const {
    #ifdef YA_DEBUG
    YA_INDEX_ERROR(i>=0 && i<numel(),"Index bigger than matrix",
                   stv(),i);
    #endif
  }
  
  void chkind2(const ya_sizet row, const ya_sizet col) const {
    #ifdef YA_DEBUG
    YA_INDEX2_ERROR(row>=0 && col>=0 && row<rows() && col<cols(),
                    "Index bigger than matrix",stv(),row,col);
    #endif
  }
};

}
#endif
