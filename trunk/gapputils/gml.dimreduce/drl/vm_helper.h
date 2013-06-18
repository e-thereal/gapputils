/***************************************************************************
                                 vm_helper.h
                             -------------------
                               W. Michael Brown

  Miscellaneous helper functions

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

#ifndef VM_HELPER_H
#define VM_HELPER_H

YALANAMESPACE

// -------------------- Template expressions

template <int one, int two> struct _VM_TEMPLATE_OR_VALUE;
template<> struct _VM_TEMPLATE_OR_VALUE<0,1> { enum { isone=1 }; };
template<> struct _VM_TEMPLATE_OR_VALUE<1,0> { enum { isone=1 }; };
template<> struct _VM_TEMPLATE_OR_VALUE<1,1> { enum { isone=1 }; };
template <int one, int two> struct _VM_TEMPLATE_OR_VALUE { enum { isone=0 }; };

template <int one, int two> struct _VM_TEMPLATE_AND_VALUE;
template<> struct _VM_TEMPLATE_AND_VALUE<1,1> { enum { isone=1 }; };
template <int one, int two> struct _VM_TEMPLATE_AND_VALUE { enum { isone=0 }; };

template <int one> struct _VM_TEMPLATE_VALUE;
template<> struct _VM_TEMPLATE_VALUE<1> { enum { isone=1 }; };
template <int one> struct _VM_TEMPLATE_VALUE { enum { isone=0 }; };

}
#endif
