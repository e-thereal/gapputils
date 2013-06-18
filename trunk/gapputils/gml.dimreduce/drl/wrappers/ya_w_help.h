/***************************************************************************
                                 ya_w_help.h
                             -------------------
                               W. Michael Brown

  Helper stuff for wrappers

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

#ifndef YA_W_HELP_H
#define YA_W_HELP_H

YALANAMESPACE

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype> struct YA_TITER;
template <class eltype, class itype>
struct YA_TITER<eltype,eltype *,0,itype> {
  typedef eltype* iter;
  typedef const eltype* citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=1 };
};

// Determine the iterator type for wrappers
template <class eltype,class mat_type,int order,class itype>
struct YA_TITER<eltype,mat_type,order,itype> {
  typedef YA_Iter<eltype,itype> iter;
  typedef YA_ConstIter<eltype,itype> citer;
  typedef YA_RowIter<eltype,itype> rowiter;
  typedef YA_ConstRowIter<eltype,itype> crowiter;
  enum { memory=0 };
};


}
#endif
