/***************************************************************************
                                ya_constants.h
                             -------------------
                               W. Michael Brown

  Misc constants

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Aug 10 2005
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef YA_CONSTANTS_H
#define YA_CONSTANTS_H

#include <cmath>
//using namespace std;

YALANAMESPACE

#ifndef PI
///\def PI
#define PI 3.1415926535897932384626433832795
#endif
/// Returns PI
template<class eltype>
inline eltype pi() { return PI; }

#define BIG_DOUBLE std::numeric_limits<double>::max()
#define BIGT std::numeric_limits<eltype>::max()
/// Big number
template<class eltype>
inline eltype big() { return std::numeric_limits<eltype>::max(); }

}

#endif

