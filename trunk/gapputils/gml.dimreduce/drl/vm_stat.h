/***************************************************************************
                                  vm_stat.h
                             -------------------
                               W. Michael Brown

  Statistics functions

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Aug 15 2007
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

#ifndef VM_STAT_H
#define VM_STAT_H

YALANAMESPACE

/// Calculate covarience
template<class eltype, class mat_type, class ya_type2>
inline VM_SymMat cov(const YA_BaseT &m, const ya_type2 &t);

/// Compute the correlation coefficient
template<class eltype, class mat_type, class ya_type2>
inline eltype corrcoef(const YA_BaseT &m, const ya_type2 &t);

/// Compute the coefficient of determination
template<class eltype, class mat_type, class ya_type2>
double rsquared(const YA_BaseT &m, const ya_type2 &two);

/// Compute the root mean squared deviation for two matrices (rows are points)
template<class eltype, class mat_type, class ya_type2>
double rmsd(const YA_BaseT &m, const ya_type2 &two);

/// Calculate residual between a distance matrix and distances between new_crds
/** Routine for generating statistics based on the distances between two sets
  * of coordinates. The distances for the first set must be precomputed (i.e.
  * with dist_mat). Distances for the second set of coordinates are calculated
  * for each dimensionality specified in the dims vector (for a dimensionality
  * of 5, the first 5 columns of new_crds is used). Any distances equal to INF
  * in the orginal_distances matrix are not compared.
  *
  * \param original_distances Original distances used for comparison
  * \param new_crds Coordinates from which distances are compared to original
  * \param dims SORTED vector containing dimensionalities to be used for dists
  * \param corrcoef Vector of residual correlation coefficients between dists
  * \param rsquared Vector of residual coefficients of determination 
  **/
template<class eltype, class mat_type, class ya_type2, class ya_type3, 
         class ya_type4, class vm_type5>
void dist_residual(const YA_BaseT &original_distances, const ya_type2 &new_crds,
                   const ya_type3 &dims,ya_type4 &corrcoef,vm_type5 &rsquared);

/// Calculate residual between a distance matrix and distances between new_crds
/** Distances between each point and the landmark points are calculated for
  * new_crds **/
template<class eltype, class mat_type, class ya_type2, class ya_type3, 
         class ya_type4, class vm_type5, class vm_type6>
void dist_residual(const YA_BaseT &original_distances, const ya_type2 &new_crds,
                   const ya_type3 &dims,ya_type4 &corrcoef,vm_type5 &rsquared,
                   const vm_type6 &landmarks);
}

#endif
