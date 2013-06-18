/***************************************************************************
                               ya_direct_mem.h
                             -------------------
                               W. Michael Brown

  Definitions for types that use direct memory access
                               
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

#ifndef YA_DIRECT_MEM_H
#define YA_DIRECT_MEM_H

YALANAMESPACE

// -------------------------- Direct Memory Access
template <class eltype, class eltype2> struct ya_direct_mem;
template <class eltype> struct ya_direct_set;
template <class eltype> struct ya_direct_setv;

template <> struct ya_direct_mem<bool,bool> { enum { ans=1 }; };
template <> struct ya_direct_mem<char,char> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned char,unsigned char> { enum { ans=1 }; };
template <> struct ya_direct_mem<int,int> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned,unsigned> { enum { ans=1 }; };
template <> struct ya_direct_mem<short,short> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned short,unsigned short> { enum { ans=1 }; };
template <> struct ya_direct_mem<long,long> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned long,unsigned long> { enum { ans=1 }; };
template <> struct ya_direct_mem<float,float> { enum { ans=1 }; };
template <> struct ya_direct_mem<double,double> { enum { ans=1 }; };
template <> struct ya_direct_mem<long double,long double> { enum { ans=1 }; };

template <> struct ya_direct_mem<const bool,bool> { enum { ans=1 }; };
template <> struct ya_direct_mem<const char,char> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned char,unsigned char> { enum { ans=1 }; };
template <> struct ya_direct_mem<const int,int> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned,unsigned> { enum { ans=1 }; };
template <> struct ya_direct_mem<const short,short> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned short,unsigned short> { enum { ans=1 }; };
template <> struct ya_direct_mem<const long,long> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned long,unsigned long> { enum { ans=1 }; };
template <> struct ya_direct_mem<const float,float> { enum { ans=1 }; };
template <> struct ya_direct_mem<const double,double> { enum { ans=1 }; };
template <> struct ya_direct_mem<const long double,long double> { enum { ans=1 }; };

template <> struct ya_direct_mem<bool,const bool> { enum { ans=1 }; };
template <> struct ya_direct_mem<char,const char> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned char,const unsigned char> { enum { ans=1 }; };
template <> struct ya_direct_mem<int,const int> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned,const unsigned> { enum { ans=1 }; };
template <> struct ya_direct_mem<short,const short> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned short,const unsigned short> { enum { ans=1 }; };
template <> struct ya_direct_mem<long,const long> { enum { ans=1 }; };
template <> struct ya_direct_mem<unsigned long,const unsigned long> { enum { ans=1 }; };
template <> struct ya_direct_mem<float,const float> { enum { ans=1 }; };
template <> struct ya_direct_mem<double,const double> { enum { ans=1 }; };
template <> struct ya_direct_mem<long double,const long double> { enum { ans=1 }; };

template <> struct ya_direct_mem<const bool,const bool> { enum { ans=1 }; };
template <> struct ya_direct_mem<const char,const char> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned char,const unsigned char> { enum { ans=1 }; };
template <> struct ya_direct_mem<const int,const int> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned,const unsigned> { enum { ans=1 }; };
template <> struct ya_direct_mem<const short,const short> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned short,const unsigned short> { enum { ans=1 }; };
template <> struct ya_direct_mem<const long,const long> { enum { ans=1 }; };
template <> struct ya_direct_mem<const unsigned long,const unsigned long> { enum { ans=1 }; };
template <> struct ya_direct_mem<const float,const float> { enum { ans=1 }; };
template <> struct ya_direct_mem<const double,const double> { enum { ans=1 }; };
template <> struct ya_direct_mem<const long double,const long double> { enum { ans=1 }; };

template <class eltype, class eltype2> struct ya_direct_mem { enum { ans=0 }; };

template <> struct ya_direct_set<char> { enum { ans=1 }; };
template <> struct ya_direct_set<unsigned char> { enum { ans=1 }; };
template <> struct ya_direct_set<int> { enum { ans=1 }; };
template <> struct ya_direct_set<unsigned> { enum { ans=1 }; };
template <> struct ya_direct_set<short> { enum { ans=1 }; };
template <> struct ya_direct_set<unsigned short> { enum { ans=1 }; };
template <> struct ya_direct_set<long> { enum { ans=1 }; };
template <> struct ya_direct_set<unsigned long> { enum { ans=1 }; };
#ifdef YA_MEMSET_FLOAT
template <> struct ya_direct_set<float> { enum { ans=1 }; };
template <> struct ya_direct_set<double> { enum { ans=1 }; };
#endif
template <class eltype> struct ya_direct_set { enum { ans=0 }; };

template <> struct ya_direct_setv<char> { enum { ans=1 }; };
template <class eltype> struct ya_direct_setv { enum { ans=0 }; };

template<int order1,int order2> struct _YA_TSAME_ORDER;
template<> struct _YA_TSAME_ORDER<0,0> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<1,1> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<0,2> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<2,0> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<2,2> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<1,2> { enum { ans=1 }; };
template<> struct _YA_TSAME_ORDER<2,1> { enum { ans=1 }; };
template<int order1,int order2> struct _YA_TSAME_ORDER { enum { ans=0 }; };

}
#endif
