/**************************************************************************
                            ya_manifold_samples.h
                             -------------------
                                 Haixia Jia

    Sample manifolds for dimensionality reduction

 __________________________________________________________________________
    This file is part of the Dr.L. Library
__________________________________________________________________________

    begin      : Dec 1 2008
    authors    : Haixia Jia
    email      : wmbrown@sandia.gov
 ***************************************************************************/

#ifndef YA_MANIFOLD_SAMPLES_H
#define YA_MANIFOLD_SAMPLES_H

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

YALANAMESPACE

// Generate a japanese flag roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_japaneseflag(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng);

// Generate a my swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_thinswissroll(YA_BaseT &input, int* colors,  ya_sizet sample_size, 
                   Rng &rng);

// Generate a non developable swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_nondevroll(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng);

// Generate a trefoil knot input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_tknot(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng);

// Generate a sphere input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_sphere(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng);

// Generate a torus input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_torus(YA_BaseT &input, int* colors,  ya_sizet sample_size, Rng &rng);

// Generate a swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_swissroll1(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng);

// Generate a box input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_box(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng);

// Generate a swiss roll input matrix
template<class eltype, class mat_type, class Rng>
void ya_manifold_swissroll(YA_BaseT &input, int* colors, ya_sizet sample_size, Rng &rng);

#endif

}
#include "ya_manifold_samples.cpp"
