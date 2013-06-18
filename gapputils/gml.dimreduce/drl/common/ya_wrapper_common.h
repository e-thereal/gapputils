/***************************************************************************
                             ya_wrapper_common.h
                             -------------------
                               W. Michael Brown

  Common Implementations for Matrix Wrappers

  *** Use stv() before any stv functions!

_________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Dec 8 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_WRAPPER_COMMON_H
#define YA_WRAPPER_COMMON_H
  
YALANAMESPACE

/// Common Implementations for Matrices that are not Wrappers
template<class eltype, class mat_type>
class YA_WrapperCommon : public YA_WrapperT {
 public:
  using YA_WrapperT::operator();
  using YA_WrapperT::operator=;
  // Const versions of functions not defined elsewhere
  using YA_WrapperT::setup;
  using YA_WrapperT::clear;

  #include "common/ya_common_fun.h"

  inline typename YA_Ref at (const ya_sizet i) const
    { return _ya_order_help<typename YA_Ref,const mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::at(stv(),i); }
  inline typename YA_Ref at2 (const ya_sizet row, const ya_sizet col) const
    { return _ya_order_help<typename YA_Ref,const mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::at2(stv(),row,col); }
  inline typename YA_Ref rowmajor_at(const ya_sizet i) const
    { return _ya_order_help<typename YA_Ref,const mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::rowmajor_at(stv(),i); }

  inline typename YA_Ref back() const
    { return _ya_order_help<typename YA_Ref,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::back(stv()); }
  inline typename YA_Ref front() const
    { return _ya_order_help<typename YA_Ref,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::front(stv()); }

  inline typename YA_Ref stride_at(const ya_sizet i) const 
    { return stv().at(i); }

  inline typename YA_IterT begin() const 
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::begin(stv()); }
  inline typename YA_IterT end() const
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::begin(stv())+stv().numel(); }
  inline typename YA_RowIterT rowbegin() const 
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::rowbegin(stv()); }
  inline typename YA_RowIterT rowend() const
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::rowbegin(stv())+stv().numel(); }
  inline typename YA_CptrT c_ptr() const { return stv().begin(); }

 protected:
  YA_WrapperCommon() {}
};

}
#endif
