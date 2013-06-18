/***************************************************************************
                                 ya_wrapper.h
                             -------------------
                               W. Michael Brown

  Base class for wrappers
  Allows const modification when parent of wrapper is non-const

  *** Use stv() before any stv functions!

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sun Feb 18 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_WRAPPER_H
#define YA_WRAPPER_H
  
YALANAMESPACE

/// Vector Matrix Slice Common Implementations
template<class eltype, class mat_type>
class YA_Wrapper : public YA_BaseT {
 public:
  using YA_BaseT::operator();
  using YA_BaseT::operator=;
  using YA_BaseT::setup;
  using YA_BaseT::clear;
  using YA_BaseT::operator[];
  using YA_BaseT::row_at;
  using YA_BaseT::stride_at;
  using YA_BaseT::front;
  using YA_BaseT::back;
  using YA_BaseT::begin;
  using YA_BaseT::end;
  using YA_BaseT::rowbegin;
  using YA_BaseT::rowend;
  using YA_BaseT::c_ptr;

  // Allow for static polymorphism
  inline mat_type& stv()
    { return reinterpret_cast<mat_type &>(*this); }
  // Allow for static polymorphism
  inline const mat_type& stv() const
    { return reinterpret_cast<const mat_type &>(*this); }

  inline void clear() const {
    YA_DEBUG_ERROR(this->numel()==0,
      "Attempt to clear a matrix wrapper."); }
  
  inline void setup(const ya_sizet l) const { 
    YA_DEBUG_ERROR((this->rows()==1 || this->cols()==1) && this->numel()==l,
                   "Cannot change size of matrix wrapper."); }
  inline void setup(const ya_sizet i, const ya_sizet j) const {
    YA_DEBUG_ERROR((this->rows()==i && this->cols()==j) ||
                   (this->rows()==j && this->cols()==1) ||
                   (this->cols()==i && this->rows()==1),
                   "Cannot change size of matrix wrapper."); }

  inline typename YA_Ref operator()
        (const ya_sizet row, const ya_sizet col) const
    { this->chkind2(row,col); return stv().at2(row,col); }
  inline typename YA_Ref operator()(const ya_sizet i) const
    { this->chkind(i); return stv().at(i); }
  inline typename YA_Ref operator[](const ya_sizet i) const
    { this->chkind(i); return stv().at(i); }
  inline typename YA_Ref row_at(const ya_sizet i) const
    { this->chkind(i); return stv().rowmajor_at(i); }
  inline typename YA_Ref stride_at(const ya_sizet i) const 
    { return stv().stride_at(i); }
  inline typename YA_Ref back() const { return stv().back(); }
  inline typename YA_Ref front() const  { return stv().front(); }

  inline typename YA_IterT begin() const { return stv().begin(); }
  inline typename YA_IterT end() const { return stv().end(); }
  inline typename YA_IterT rowbegin() const { return stv().begin(); }
  inline typename YA_IterT rowend() const { return stv().end(); }
  inline typename YA_CptrT c_ptr() const { return stv().c_ptr(); }

  inline const YA_WrapperT& operator=(const eltype value) const
    { ya_copy_scalar(*this,value); return *this; }    
  inline const YA_WrapperT& operator=(const std::string &value) const
    { ya_string_to_mat<eltype>::convert(value,*this); return *this; }
  inline const YA_WrapperT& operator=(const char *value) const
    { return (*this)=std::string(value); }
  inline const YA_WrapperT& operator=(const YA_WrapperT &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }
  template <class ya_type2>
  inline const YA_WrapperT& operator=(const ya_type2 &v) const
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }

  inline bool iswrapper() const { return true; }
 protected:
  YA_Wrapper() {}
};

}
#endif
