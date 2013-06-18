/***************************************************************************
                             ya_template_meta.h
                             -------------------
                               W. Michael Brown

  Miscellaneous template helpers

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Sep 19 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_TEMPLATE_META
#define YA_TEMPLATE_META

YALANAMESPACE

template <int one, int two> struct _YA_TOR;
template<> struct _YA_TOR<0,0> { enum { ans=0 }; };
template <int one, int two> struct _YA_TOR { enum { ans=1 }; };

template <int one, int two> struct _YA_TAND;
template<> struct _YA_TAND<1,1> { enum { ans=1 }; };
template <int one, int two> struct _YA_TAND { enum { ans=0 }; };

template <int one> struct _YA_TNOT;
template<> struct _YA_TNOT<0> { enum { ans=1 }; };
template <int one> struct _YA_TNOT { enum { ans=0 }; };

template <class one, class two> struct _YA_TSAME;
template<class one> struct _YA_TSAME<one,one> { enum { ans=1 }; };
template<class one> struct _YA_TSAME<const one,one> { enum { ans=1 }; };
template<class one> struct _YA_TSAME<one,const one> { enum { ans=1 }; };
template<class one, class two> struct _YA_TSAME { enum { ans=0 }; };

// If val is 1, return type 1 otherwise type 2
template<int val, class one, class two> struct _YA_CHOOSETYPE1;
template<class one, class two> struct _YA_CHOOSETYPE1<1,one,two>
  { typedef one ans; };
template<int val, class one, class two> struct _YA_CHOOSETYPE1
  { typedef two ans; };

// If one=1, ans=2 otherwise ans=0
template <int one> struct _YA_TEMPLATE_00_12;
template<> struct _YA_TEMPLATE_00_12<1> { enum { ans=2 }; };
template <int one> struct _YA_TEMPLATE_00_12 { enum { ans=0 }; };

}
#endif
