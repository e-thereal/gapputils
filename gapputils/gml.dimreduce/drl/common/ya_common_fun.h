/***************************************************************************
                               ya_common_fun.h
                             -------------------
                               W. Michael Brown

  Vector/Matrix Common Functions used by classes derived from YA_Base
  or YA_Wrapper

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

  inline mat_type& stv() { return reinterpret_cast<mat_type &>(*this); }
  inline const mat_type& stv() const
    { return reinterpret_cast<const mat_type &>(*this); }

  inline typename YA_Ref at (const ya_sizet i)
    { return _ya_order_help<typename YA_Ref,mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::at(stv(),i); }
  inline typename YA_Ref at2 (const ya_sizet row, const ya_sizet col) 
    { return _ya_order_help<typename YA_Ref,mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::at2(stv(),row,col); }
  inline typename YA_Ref rowmajor_at(const ya_sizet i)
    { return _ya_order_help<typename YA_Ref,mat_type,YA_TRAIT::VME_ORDER,
                            YA_TRAIT::VME_VECTOR>::rowmajor_at(stv(),i); }

  inline typename YA_Ref back()  
    { return _ya_order_help<typename YA_Ref,mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::back(stv()); }
  inline typename YA_Ref front() 
    { return _ya_order_help<typename YA_Ref,mat_type,
              YA_TRAIT::VME_ORDER,YA_TRAIT::VME_VECTOR>::front(stv()); }

  inline ya_sizet stride_start() const { return 0; }
  inline ya_sizet stride_end() const { return stv().numel()-1; }
  inline ya_sizet stride() const { return 1; }
  inline typename YA_Ref stride_at(const ya_sizet i) 
    { return stv().at(i); }

  // Destructor
  inline void destruct() {}

  inline typename YA_IterT begin()
    { return _ya_vec_help<mat_type,
                          YA_TRAIT::VME_VECTOR>::begin(stv()); }
  inline typename YA_IterT end()
    { return _ya_vec_help<mat_type,
                          YA_TRAIT::VME_VECTOR>::begin(stv())+stv().numel(); }
  inline typename YA_RowIterT rowbegin() 
    { return _ya_vec_help<mat_type,
                          YA_TRAIT::VME_VECTOR>::rowbegin(stv()); }
  inline typename YA_RowIterT rowend()
    { return _ya_vec_help<mat_type,
                          YA_TRAIT::VME_VECTOR>::rowbegin(stv())+stv().numel(); }

  inline typename YA_CptrT c_ptr() { return stv().begin(); }
  inline ya_sizet numel() const { return stv().rows()*stv().cols(); }

  // --------------------Static Virtual Functions Not Defined
  inline void clear() { 
    YA_DEBUG_ERROR(0==1,
      "clear() not defined for this type. Developer Error"); }
  inline void setup(const ya_sizet len) {
    YA_DEBUG_ERROR1(stv().numel()==len,
                    "Cannot change size of this matrix type",stv());
  }
  inline void setup(const ya_sizet rowsi, const ya_sizet columnsi) {
    YA_DEBUG_ERROR1((stv().rows()==rowsi && stv().cols()==columnsi) ||
                    (YA_TRAIT::VME_VECTOR && stv().numel()==rowsi*columnsi),
                    "Cannot change size of this matrix type",stv());
  }
  inline void reshape(const ya_sizet rowsi, const ya_sizet colsi) {
    YA_DEBUG_ERROR(rowsi==stv().rows() && colsi==stv().cols(),
                   "Cannot reshape fixed size matrix.");
  }

  inline ya_sizet cols() const { 
    YA_DEBUG_ERROR(0==1,
      "cols() not defined for this type. Developer Error"); return 0; }
  inline ya_sizet rows() const {
    YA_DEBUG_ERROR(0==1,
      "rows() not defined for this type. Developer Error"); return 0; }
