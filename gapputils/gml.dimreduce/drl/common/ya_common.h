/***************************************************************************
                                 ya_common.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix Common Implementations

  This file implements functions from vecmatbase.h that are common to most
  vector/matrix classes, but change in a few
  
  *** Use stv() before any stv functions!

__________________________________________________________________________
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

#ifndef YA_COMMON_H
#define YA_COMMON_H
  
YALANAMESPACE

/// Common Implementations for Matrices that are not Wrappers
template<class eltype, class mat_type>
class YA_Common : public YA_BaseT {
 public:
  using YA_BaseT::operator();
  using YA_BaseT::operator=;

  #include "common/ya_common_fun.h"

  inline typename YA_ConstRef at (const ya_sizet i) const
    { return _ya_order_help<typename YA_ConstRef,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::at(stv(),i); }
  inline typename YA_ConstRef at2
         (const ya_sizet row, const ya_sizet col) const
    { return _ya_order_help<typename YA_ConstRef,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::at2(stv(),row,col); }
  inline typename YA_ConstRef rowmajor_at(const ya_sizet i) const
    { return _ya_order_help<typename YA_ConstRef,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::rowmajor_at(stv(),i); }
  inline typename YA_ConstRef stride_at(const ya_sizet i) const 
    { return stv().at(i); }

  inline typename YA_ConstRef back() const  
    { return _ya_order_help<typename YA_ConstRef,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::back(stv()); }
  inline typename YA_ConstRef front() const  
    { return _ya_order_help<typename YA_ConstRef,const mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::front(stv()); }

  inline typename YA_ConstIterT begin() const 
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::cbegin(stv()); }
  inline typename YA_ConstIterT end() const
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::cbegin(stv())+stv().numel(); }
  inline typename YA_ConstRowIterT rowbegin() const 
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::crowbegin(stv()); }
  inline typename YA_ConstRowIterT rowend() const
    { return _ya_vec_help<const mat_type,
                          YA_TRAIT::VME_VECTOR>::crowbegin(stv())+stv().numel(); }
  inline typename YA_ConstCptrT c_ptr() const { return stv().begin(); }

  inline bool iswrapper() const { return false; }
 protected:
  YA_Common() {}
};

}
#endif
