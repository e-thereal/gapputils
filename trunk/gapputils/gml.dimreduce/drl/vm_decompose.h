/***************************************************************************
                                vm_decompose.h
                             -------------------
                               W. Michael Brown

  Routines for matrix decomposition
  
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

/*! \file */

#ifndef VM_DECOMPOSE_H
#define VM_DECOMPOSE_H

#include "vm_lapack_wrapper.h"

YALANAMESPACE

/// Full eigen decomposition assumes symmetric matrix
/** Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the divide and conquer algorithm
  * ...evd. Otherwise the simple driver is utilized **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eig(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values);

/// Full eigen decomposition assumes symmetric matrix
/** Eigen values are returned in ascending order
  * If YA_LAPACK is defined this uses the divide and conquer algorithm
  * ...evd. Otherwise the simple driver is utilized **/
template<class eltype, class mat_type, class ya_type2>
void eig(const YA_BaseT &m, ya_type2 &values);

/// Full eigen decomposition assumes symmetric matrix
/** Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the simple driver which requires less 
  * memory but is slower than eig. Otherwise the internal simple driver 
  * is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigev(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values);

/// Full eigen decomposition assumes symmetric matrix
/** Eigen values are returned in ascending order
  * If YA_LAPACK is defined this uses the simple driver which requires less 
  * memory but is slower than eig. Otherwise the internal simple driver 
  * is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2>
void eigev(const YA_BaseT &m, ya_type2 &values);

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigev(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, char type);

/// Full eigen decomposition assumes symmetric matrix
/** \param tol Convergence parameter for eigen values. Use 0 for default.
  * 
  * The parameter tol describes the convergence of eigenvalues into an
  * interval [a,b] with width: tol + eps *   max( |a|,|b| )
  *
  * Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the RRR driver which is fastest.
  * Otherwise the internal simple driver is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigrrr(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values,
        eltype tol);

/// Full eigen decomposition assumes symmetric matrix
/** \param tol Convergence parameter for eigen values. Use 0 for default.
  * 
  * The parameter tol describes the convergence of eigenvalues into an
  * interval [a,b] with width: tol + eps *   max( |a|,|b| )
  *
  * Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the RRR driver which is fastest.
  * Otherwise the internal simple driver is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2>
void eigrrr(const YA_BaseT &m, ya_type2 &values);

/// Returns largest 'dim' eigenvalues and vectors in descending order
/** \param tol Convergence parameter for eigen values. Use 0 for default.
  * 
  * The parameter tol describes the convergence of eigenvalues into an
  * interval [a,b] with width: tol + eps *   max( |a|,|b| )
  *
  * Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the RRR driver which is fastest.
  * Otherwise the internal simple driver is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigrrr(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, ya_type3 &values,
            eltype tol);

/// Returns largest 'dim' eigenvalues in descending order
/** \param tol Convergence parameter for eigen values. Use 0 for default.
  * 
  * The parameter tol describes the convergence of eigenvalues into an
  * interval [a,b] with width: tol + eps *   max( |a|,|b| )
  *
  * Eigen values and corresponding vectors are returned in ascending order
  * If YA_LAPACK is defined this uses the RRR driver which is fastest.
  * Otherwise the internal simple driver is utilized
  * \sa eig() **/
template<class eltype, class mat_type, class ya_type2>
void eigrrr(const YA_BaseT &m, ya_sizet dim, ya_type2 &values, eltype tol);

/// Calculate the largest 'dim' eigenvalues and eigenvectors in descending order
/** If LAPACK is defined, the ...evx driver is used. Otherwise,
  * the power method is used.
  *
  * \param tol Convergence parameter for eigen values
  * \param max_iters Maximum number of iterations (not used for LAPACK)
  *
  * Generates warning 305 L 9 if vector does not converge. **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                 ya_type3 &values, eltype tol, ya_sizet max_iters);

/// Calculate the largest 'dim' eigenvalues in descending order
/** If LAPACK is defined, the ...evx driver is used. Otherwise,
  * the power method is used.
  *
  * \param tol Convergence parameter for eigen values
  * \param max_iters Maximum number of iterations (not used for LAPACK)
  *
  * Generates warning 305 L 9 if vector does not converge. **/
template<class eltype, class mat_type, class ya_type2>
void powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &values,
                 eltype tol, ya_sizet max_iters);

template<class eltype, class mat_type, class ya_type2, class ya_type3, class Rng>
int _powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                 ya_type3 &values, eltype tol, ya_sizet max_iters, Rng &rng);

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
/** This uses eigsrrr with a default tolerance of 0 **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, ya_type3 &values);

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
/** This uses eigsrrr with a default tolerance of 0 **/
template<class eltype, class mat_type, class ya_type2>
void eigs(const YA_BaseT &m, ya_sizet dim, ya_type2 &values);

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
/** This uses eigsrrr with a default tolerance of 0 **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs_small(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                ya_type3 &values);

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
/** This uses eigsrrr with a default tolerance of 0 **/
template<class eltype, class mat_type, class ya_type2>
void eigs_small(const YA_BaseT &m, ya_sizet dim, ya_type2 &values);

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
          EigenOptions &opts);

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2>
void eigs(const YA_BaseT &m, ya_type2 &values, EigenOptions &opts);

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs_small(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
                EigenOptions &opts);

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2>
void eigs_small(const YA_BaseT &m, ya_type2 &values, EigenOptions &opts);

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void _eigs(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
           EigenOptions &opts, const bool gvectors, const bool small);

/// Cholesky decomposition (A=L * LT)
/** Generates error 304 L 9 if matrix is not positive definite
  * \param write_err If false, no error is output
  * \note This routine assumes the matrix is symmetric **/
template<class eltype, class mat_type, class ya_type2>
int chol(const YA_BaseT &a, ya_type2 &L, bool write_err = true);

/// Cholesky solve A*x=b
/** Generates error 304 L 9 if matrix is not positive definite
  * \param write_err If false, no error is output
  * \note This routine assumes the matrix is symmetric **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
int chol_solve(const YA_BaseT &a, ya_type2 &x, const ya_type3 &b,
               bool write_err = true);

/// LU decomposition
/** ----- Adapted from GSL gsl_linalg_LU_decomp
  * L is stored in the strict lower triangular part of input
  * U is stored in the upper triangular part of the input matrix 
  * P is stored in the permutation p. 
  * signum gives the sign of the permutation, (-1)^n, where n is the
  * number of interchanges in the permutation. **/         
template<class eltype, class mat_type, class ya_type2>
void lu(YA_BaseT &input, ya_type2 &p, int &signum);

/// Solve Ax=b in-place using LU decomposition
/** Backward substitution using C-BLAS
  * \param LU contains the LU decomposition of A
  * \param p is the permutation from the LU decomposition.
  * \param x stores b as input and x as output 
  * \sa lu() **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void lu_solve(const YA_BaseT &LU, const ya_type2 &p, ya_type3 &x);

/// Invert the matrix L*U
/** Invokes LU solve on the identity matrix
  * \param inverse inverse of L*U
  * \sa lu() **/
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void lu_invert(const YA_BaseT &LU, const ya_type2 &p, ya_type3 &inverse);

}
#endif
