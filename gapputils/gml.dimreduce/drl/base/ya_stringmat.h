/***************************************************************************
                               ya_stringmat.h
                             -------------------
                               W. Michael Brown

  Functions for dealing with matrices in strings
 
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

#ifndef YA_STRINGMAT_H
#define YA_STRINGMAT_H

YALANAMESPACE

template <class eltype>
struct ya_string_to_mat {
  template <class ya_type>
  static int convert(const std::string &value, ya_type &mat);
};

}
#endif
