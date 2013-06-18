/***************************************************************************
                                  ya_rand.h
                             -------------------
                               W. Michael Brown

  Functions for random number generation
 
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

#ifndef YA_RAND_H
#define YA_RAND_H

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

YALANAMESPACE

/// Replace any elements in mat with random numbers >=0 and <=1
template<class eltype, class mat_type, class Rng>
void ip_rand(YA_BaseT &mat, Rng &rng);

// Const copy
template<class eltype, class mat_type, class Rng>
void ip_rand(const YA_WrapperT &mat, Rng &rng);

/// Replace any elements in mat with random numbers >0 and <1
template<class eltype, class mat_type, class Rng>
void ip_rande(YA_BaseT &mat, Rng &rng);

// Const copy
template<class eltype, class mat_type, class Rng>
void ip_rande(const YA_WrapperT &mat, Rng &rng);

/// Replace any elements in mat with normal distribution of random numbers
template<class eltype, class mat_type, class Rng>
void ip_randn(YA_BaseT &mat, Rng &rng);

// Const copy
template<class eltype, class mat_type, class Rng>
void ip_randn(const YA_WrapperT &mat, Rng &rng);

/// Replace any elements in mat with integer >=0 and <=max
template<class eltype, class mat_type, class Rng>
void ip_randi(YA_BaseT &mat, const typename Rng::rtype max, Rng &rng);

// Const copy
template<class eltype, class mat_type, class Rng>
void ip_randi(const YA_WrapperT &mat, const typename Rng::rtype max, Rng &rng);

}
#endif
