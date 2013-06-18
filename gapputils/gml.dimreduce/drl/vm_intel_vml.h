/***************************************************************************
                               vm_intel_vml.h
                             -------------------
                               W. Michael Brown

  Wrappers for Intel Vector Math Library
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Sep 17 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_INTEL_VML_H
#define VM_INTEL_VML_H

#include "mkl_vml.h"

YALANAMESPACE

// ---------------------------- EXP

template<class ptype1, class ptype2>
inline void _vml_exp(const ya_sizet length, ptype1 in, ptype2 out) {
  YA_DEBUG_ERROR(0==1,"Attempted call to intel vml with unsupported type.");
}

template<>
inline void _vml_exp(const ya_sizet length, const float* in, float* out) {
  vsExp(length,in,out);
}

template<>
inline void _vml_exp(const ya_sizet length, const double* in, double* out) {
  vdExp(length,in,out);
}

// ---------------------------- INV

template<class ptype1, class ptype2>
inline void _vml_inv(const ya_sizet length, ptype1 in, ptype2 out) {
  YA_DEBUG_ERROR(0==1,"Attempted call to intel vml with unsupported type.");
}

template<>
inline void _vml_inv(const ya_sizet length, const float* in, float* out) {
  vsInv(length,in,out);
}

template<>
inline void _vml_inv(const ya_sizet length, const double* in, double* out) {
  vdInv(length,in,out);
}

// ---------------------------- LOG

template<class ptype1, class ptype2>
inline void _vml_log(const ya_sizet length, ptype1 in, ptype2 out) {
  YA_DEBUG_ERROR(0==1,"Attempted call to intel vml with unsupported type.");
}

template<>
inline void _vml_log(const ya_sizet length, const float* in, float* out) {
  vsLn(length,in,out);
}

template<>
inline void _vml_log(const ya_sizet length, const double* in, double* out) {
  vdLn(length,in,out);
}

}
#endif
