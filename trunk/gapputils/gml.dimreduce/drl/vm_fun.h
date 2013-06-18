/***************************************************************************
                                   vm_fun.h
                             -------------------
                W. Michael Brown, Haixia Jia, Jean-Paul Watson

  Vector/Matrix Functions that do not return expressions
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Feb 12 2007
    authors    : W. Michael Brown, Haixia Jia, Jean-Paul Watson
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef VM_FUN_H
#define VM_FUN_H

#include "ya_set.h"
#ifdef YA_PROGMETER
#include "base/ya_timer.h"
#endif

YALANAMESPACE

// ---- return a ya_sizet matrix with random integers >=0 and <=max
template <class Rng>
inline YA_MatI vm_random_int(const ya_sizet rows, const ya_sizet cols,
                             const typename Rng::rtype max, Rng &rng) {
  YA_MatI matrix(rows,cols);
  ip_randi(matrix,max,rng);
  return matrix;
}

// ---- return a double matrix with random numbers >=0 and <=1
template <class Rng>
inline YA_MatD vm_random(const ya_sizet rows, const ya_sizet cols,
                         Rng &rng) {
  YA_MatD matrix(rows,cols);
  ip_rand(matrix,rng);
  return matrix;
}

// ---- return a double matrix with random numbers >0 and <1
template <class Rng>
inline YA_MatD vm_random_exc(const ya_sizet rows, const ya_sizet cols,
                             Rng &rng) {
  YA_MatD matrix(rows,cols);
  ip_rande(matrix,rng);
  return matrix;
}

// ---- return a double matrix with random normal distribution
template <class Rng>
inline YA_MatD vm_random_normal(const ya_sizet rows, const ya_sizet cols,
                                Rng &rng) {
  YA_MatD matrix(rows,cols);
  ip_randn(matrix,rng);
  return matrix;
}

// ---- zeros column vector
inline const VM_ZerosV vmzeros(const ya_sizet length) 
  { return VM_ZerosV(length); }

// ---- zeros matrix
inline const VM_Zeros vmzeros(const ya_sizet rows, const ya_sizet cols) 
  { return VM_Zeros(rows,cols); }

// ---- ones column vector
inline const VM_OnesV vmones(const ya_sizet length) 
  { return VM_OnesV(length); }

// ---- ones matrix
inline const VM_Ones vmones(const ya_sizet rows, const ya_sizet cols) 
  { return VM_Ones(rows,cols); }

// ---- identity matrix
inline const VM_EyeD vmeye(const ya_sizet rows, const ya_sizet cols) 
  { return VM_EyeD(rows,cols); }

// ---- identity matrix
inline const VM_EyeD vmeye(const ya_sizet rows) 
  { return VM_EyeD(rows); }

// ---- count from zero by 1
inline const VM_ZeroToN vmcount(const ya_sizet length) 
  { return VM_ZeroToN(length); }

// ---- count from start to end by 1
inline const VM_CountU vmcount(const ya_sizet start, const char *s, 
                               const ya_sizet end) {
  return VM_CountU(start,":",end);
}

// ---- count from start to end by i
inline const VM_SeriesU vmcount(const ya_sizet start, const char *s, 
                                const ya_sizet i, const char *s2, 
                                const ya_sizet end) {
  return VM_SeriesU(start,":",i,":",end);
}

// ---- linspace of length 100
inline const VMLinSpace<double> vmlinspace(const double start, 
                                           const double end) {
  return VMLinSpace<double>(start,end);
}

// ---- linspace of specified length
inline const VMLinSpace<double> vmlinspace(const double start, const double end, 
                                           const ya_sizet length) {
  return VMLinSpace<double>(start,end,length);
}

// ---- copy the lower triangle of the matrix b into the lower triangle of a
template<class eltype, class mat_type, class mat_type2>
inline void copy_lower_triangle(YA_BaseT &a, const YA_BaseT2 &b);
  
// ---- copy the lower triangle of the matrix b into the lower triangle of a
template<class eltype, class mat_type, class mat_type2>
inline void copy_lower_triangle(const YA_WrapperT &a, const YA_BaseT2 &b);
  
// ---- length
template<class eltype, class mat_type>
inline ya_sizet length(const YA_BaseT &m);

// ---- size
template<class eltype, class mat_type>
inline ya_sizet size(const YA_BaseT &m, const ya_sizet i);

// ---- cumprod
template<class eltype, class mat_type>
inline YA_MatT cumprod(const YA_BaseT &m);

// ---- cumsum
template<class eltype, class mat_type>
inline YA_MatT cumsum(const YA_BaseT &m);

// ---- norm
template<class eltype, class mat_type>
inline eltype norm(const YA_BaseT &m);

// ---- trace
template<class eltype, class mat_type>
inline eltype trace(const YA_BaseT &m);

// ---- dot product
template<class eltype, class mat_type, class mat_type2>
inline eltype dot(const YA_BaseT &one, const YA_BaseT2 &two);

// ---- inverse
template<class eltype, class mat_type, class ya_type2>
inline void inv(const YA_BaseT &m, ya_type2 &ans);

// ---- inverse
template<class eltype, class mat_type>
inline YA_MatT inv(const YA_BaseT &m);

// ---- Unique elements
template <class eltype, class mat_type>
inline YA_DynT unique(const YA_BaseT &one);

// ---- Unique elements
template <class eltype, class mat_type>
inline VecMatReI<eltype,const mat_type,YA_DynI> unique(const YA_BaseT &one,
                                                            YA_DynI &i);

// ---- Unique elements
template <class eltype, class mat_type, class ya_type2>
VecMatReI<eltype,const mat_type,YA_DynI> unique(const YA_BaseT &one,
                                               YA_DynI &ui,ya_type2 &uj);

// ---- Set difference (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setdiff(const YA_BaseT &one, const YA_BaseT2T &two);

// ---- Set difference (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline VecMatReI<eltype,const mat_type,YA_DynI> 
  setdiff(const YA_BaseT &one, const YA_BaseT2T &two, YA_DynI &i);

// ---- Set union (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setunion(const YA_BaseT &one, const YA_BaseT2T &two);

// ---- Set union (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setunion(const YA_BaseT &one, const YA_BaseT2T &two,
                          YA_DynI &ia, YA_DynI &ib);

// ---- Set intersection (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT intersect(const YA_BaseT &one, const YA_BaseT2T &two);

// ---- Set intersection (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT intersect(const YA_BaseT &one, const YA_BaseT2T &two,
                           YA_DynI &ia, YA_DynI &ib);

// ---- Set xor (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setxor(const YA_BaseT &one, const YA_BaseT2T &two);

// ---- Set xor (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setxor(const YA_BaseT &one, const YA_BaseT2T &two,
                        YA_DynI &ia, YA_DynI &ib);


// ---- Random permutation
template <class eltype, class mat_type, class Rng>
void randperm(const ya_sizet n, YA_BaseT& m, Rng &rng);

// ---- Random permutation
template <class eltype, class mat_type, class Rng>
void randperm(const ya_sizet n, const YA_WrapperT& m, Rng &rng);

// ---- Random permutation
template <class Rng>
inline YA_RowI randperm(const ya_sizet n, Rng &rng);

// ---- Primes
template <class Rng>
inline YA_MatI primes(const ya_sizet n, Rng &rng);

// ---- Factor
template <class Rng>
inline YA_DynI factor(const ya_sizet n, Rng &rng);

/// Nearest neighbors between each point in input to all other points in input
/** \param input m x n matrix containing m points for which nbors are found
  * \param k number of neighbors to find
  * \param output m x k matrix storing the indices into neighbors
  * \param dists m x k matrix storing distances to neighbors 
  * \param verbose if >0 output progress meter/time, if >1 also nbor stats **/
template<class eltype, class mat_type,
         class ya_type2, class ya_type3>
void kneighbors(const YA_BaseT &input, const ya_sizet k, ya_type2 &output,
                ya_type3 &dists, const int verbose);

/// Find all the neighbors within epsilon distance;
/** \param m m x n matrix containing m points for which nbors are found
  * \param epsilon distance threshold
  * \param output Set of m arbitrary length vectors storing neighbor indices
  * \param dists Set of m arbitrary length vectors storing nbor distances
  * \param verbose if >0 output progress meter/time, if >1 also nbor stats **/
template<class eltype, class mat_type>
void eneighbors(const YA_BaseT &input, const eltype epsilon,                   
    std::vector<YA_DynI> &output, std::vector<YA_DynT> &dists,
                const int verbose);

/// Find nearest neighbors in the matrix nbors closest to each point in input
/** \param input m x n matrix containing m points for which nbors are found
  * \param nbors r x n matrix containing r points to be considered as neighbors
  * \param k number of neighbors to find
  * \param output m x k matrix storing the indices into neighbors
  * \param dists m x k matrix storing distances to neighbors
  * \param verbose if >0 output progress meter/time, if >1 also nbor stats **/
template<class eltype, class mat_type,class ya_type2, class ya_type3,
         class ya_type4>
void kneighbors(const YA_BaseT &input, const ya_type2 &nbors, const ya_sizet k, 
                ya_type3 &output, ya_type4 &dists, const int verbose);

/// Find all neighbors in nbors within epsilon distance to each point in input
/** \param input m x n matrix containing m points for which nbors are found
  * \param nbors r x n matrix containing r points to be considered as neighbors
  * \param epsilon distance threshold
  * \param output set of m vectors storing the indices into neighbors
  * \param dists set of m vectors storing distances to neighbors
  * \param verbose if >0 output progress meter/time, if >1 also nbor stats **/
template<class eltype, class mat_type, class ya_type2>
void eneighbors(const YA_BaseT &input, const ya_type2 &nbors, 
                const eltype epsilon, std::vector<YA_DynI> &output,
                std::vector<YA_DynT> &dists, const int verbose);

// ---- 1D Plot (using GNU plot)
template<class eltype, class mat_type>
int plot(const YA_BaseT &in, const std::string title, const char style,
         const std::string xlabel = std::string(), const std::string ylabel = std::string());

// ---- 2D color Plot (using GNU plot)
int plot(int * colors, const YA_MatD &output);

// ---- 2D Plot (using GNU plot)
template<class eltype, class mat_type, class eltype2, class mat_type2>
int plot(const YA_BaseT &xin, const YA_BaseT2T &yin, const std::string titles,
         const char cstyle, const std::string xlabel = std::string(),
         const std::string ylabel = std::string());

}

// ---- Max for matrix
template<class eltype, class mat_type,
         class eltype2, class mat_type2,
         class eltype3, class mat_type3>
inline void maxi(const YA_BaseT &m, YA_BaseT2T &ans, YA_BaseT3T &ind);

// ---- Max for vector
template<class eltype, class mat_type>
inline void max(const YA_BaseT &m, eltype &ans, ya_sizet &ind);

// --- Min for matrix
template <class eltype, class mat_type>
YA_RowT min(const YA_BaseT &m);

// --- Min for matrix with indices
template <class eltype, class mat_type,class ya_type2>
YA_RowT min(const YA_BaseT &m, ya_type2 &indices);

// ---- Sort
template<class eltype, class mat_type>
inline YA_MatT sort(const YA_BaseT &m);

// ---- Sort
template<class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_MatT sort(const YA_BaseT &m, YA_BaseT2T &ans);

// ---- Sort and return indices
template<class eltype, class mat_type, class eltype2, class mat_type2,
         class ya_type2>
inline void sorti(const YA_BaseT &m, ya_type2 &ans, YA_BaseT2T &indices);

#endif
