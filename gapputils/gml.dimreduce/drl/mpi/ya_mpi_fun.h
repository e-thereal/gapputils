/***************************************************************************
                                 ya_mpi_fun.h
                             -------------------
                               W. Michael Brown

  Misc MPI Routines

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sat Apr 25 2009
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

#ifndef YA_MPI_FUN_H
#define YA_MPI_FUN_H

YALANAMESPACE

/// Get the rank and number of procs
inline void ya_mpi_info(int &rank, int &num_procs);

/// Divide up N iterations into counts per proc and starting index per proc
template <class ya_type1, class ya_type2>
inline void ya_mpi_counts(const int N, const int num_procs, ya_type1 &counts, 
                           ya_type2 &starts);

#ifdef YA_MPI
/// Concatenate matrices on each proc to large matrix on rank 0 proc
/** \param mat Matrix pieces (most efficient if row major)
  * \param full_mat Full matrix (most efficient if row major)
  * \param C Number of columns
  * \param counts Number of rows in mat on each proc
  * \param starts Starting index within full matrix on each proc **/
template <class eltype, class mat_type, class ya_type2, class ya_type3, 
          class ya_type4>
void ya_mpi_concat_rows(YA_BaseT &mat, ya_type2 &full_mat,
                        const int C, const int me, const int num_procs,
                        ya_type3 &counts, ya_type4 &starts);

/// Return the minimum, maximum, mean, and standard deviation on proc 0
/** \sa vstat **/
template<class eltype, class mat_type, class stype, class ya_type2, 
         class ya_type3>
void mpi_vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean,
               stype &vstd);

/// Return the minimum, maximum, mean, and standard deviation on proc 0
/** \param C number of columns in mat
  * \sa vstat **/
template<class eltype, class mat_type, class stype>
void mpi_vstat(const YA_BaseT &mat, const int C, YA_RowT &vmin, YA_RowT &vmax, 
               YA_Rows<stype,1> &vmean, YA_Rows<stype,1> &vstd);
#endif

}
#endif
