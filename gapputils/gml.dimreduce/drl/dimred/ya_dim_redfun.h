/***************************************************************************
                               ya_dim_redfun.h
                             -------------------
                               W. Michael Brown

  Miscellaneous functions for dimensionality reduction

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Tue Feb 19 2008
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

#ifndef YA_DIM_REDFUN_H
#define YA_DIM_REDFUN_H

YALANAMESPACE

/// Return a n x d matrix of residual variances for n points from PCA
/** \param input n x d matrix of samples
  * \param k Local PCA is performed on each point and its k nearest neighbors
  * \param output n x d matrix of residual variances 
  * \param verbose 0 no output, 1 progress meter/timing, 2 also nbor stats **/
template<class eltype, class mat_type, class ya_type2>
void point_pca(const YA_BaseT &input, const ya_sizet k,
               EigenOptions &eigopts, ya_type2 &output, const int verbose);

/// Return a n x d matrix of residual variances for n points from PCA
/** \param dims length D vector of dimensionalities to include in calculation
  * \param input n x D matrix of samples
  * \param k Local PCA is performed on each point and its k nearest neighbors
  * \param output n x D matrix of residual variances 
  * \param verbose 0 no output, 1 progress meter/timing, 2 also nbor stats **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void point_pca(const YA_BaseT &input, const ya_sizet k, const ya_type2 &dims,
               EigenOptions &eigopts, ya_type3 &output, const int verbose);

/// Same as point_pca except epsilon nearest neighbors is used
/** \sa point_pca **/
template<class eltype, class mat_type, class ya_type2>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon,
                  EigenOptions &eigopts, ya_type2 &output, const int verbose);

/// Same as point_pca except epsilon nearest neighbors is used
/** \sa point_pca **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  const ya_type2 &dims, EigenOptions &eigopts, 
                  ya_type3 &output, const int verbose);

/// Return statistics of residual variances for n points from PCA
/** \param input n x d matrix of samples
  * \param k Local PCA is performed on each point and its k nearest neighbors
  * \param vmean length d vector containing mean residual variance for each dim
  * \param vmin length d vector containing min residual variance for each dim
  * \param vmax length d vector containing max residual variance for each dim
  * \param vstd length d vector containing std devation of residual variance
  * \param verbose 0 no output, 1 progress meter/timing, 2 also nbor stats **/
template<class eltype, class mat_type>
void point_pca(const YA_BaseT &input, const ya_sizet k,
               EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin,
               YA_RowD &vmax, YA_RowD &vstd, const int verbose);

/// Return statistics of residual variances for n points from PCA
/** \param input n x d matrix of samples
  * \param k Local PCA is performed on each point and its k nearest neighbors
  * \param dims length D vector containing dimensions for statistics calculation
  * \param vmean length D vector containing mean residual variance for each dim
  * \param vmin length D vector containing min residual variance for each dim
  * \param vmax length D vector containing max residual variance for each dim
  * \param vstd length D vector containing std devation of residual variance
  * \param verbose 0 no output, 1 progress meter/timing, 2 also nbor stats **/
template<class eltype, class mat_type, class ya_type2>
void point_pca(const YA_BaseT &input, const ya_sizet k, const ya_type2 &dims,
               EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin,
               YA_RowD &vmax, YA_RowD &vstd, const int verbose);

/// Same as point_pca except epsilon nearest neighbors is used
/** \sa point_pca **/
template<class eltype, class mat_type>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon,
                  EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin,
                  YA_RowD &vmax, YA_RowD &vstd, const int verbose);

/// Same as point_pca except epsilon nearest neighbors is used
/** \sa point_pca **/
template<class eltype, class mat_type, class ya_type2>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  const ya_type2 &dims, EigenOptions &eigopts, YA_RowD &vmean, 
                  YA_RowD &vmin, YA_RowD &vmax, YA_RowD &vstd, 
                  const int verbose);

}
#endif
