/***************************************************************************
                               ya_temporary.h
                             -------------------
                               W. Michael Brown

  Return types for temporary creation
                               
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

#ifndef YA_TEMPORARY_H
#define YA_TEMPORARY_H

YALANAMESPACE

template <int mem> struct _YA_TMEM_TEMP;
template <> struct _YA_TMEM_TEMP<2> { enum { ans=2 }; };
template <> struct _YA_TMEM_TEMP<4> { enum { ans=4 }; };
template <int mem> struct _YA_TMEM_TEMP { enum { ans=0 }; };
  
#define YA_COPY_TYPE _ya_temp_type<eltype,_YA_TMEM_TEMP<YA_TRAIT::VME_MEMORY>::ans,YA_TRAIT::VME_NROWS,YA_TRAIT::VME_NCOLS>::copy_type
              
template<class eltype,int mem,int nrows,int ncols> class _ya_temp_type;
// Symmetric matrix
template<class eltype,int nrows,int ncols> class _ya_temp_type<eltype,2,nrows,ncols> { public:
  typedef VM_SymMat copy_type; };
// Sparse matrix
template<class eltype,int nrows,int ncols> class _ya_temp_type<eltype,4,nrows,ncols> { public:
  typedef VecMatSparse<eltype> copy_type; };
// Regular matrix
template<class eltype> class _ya_temp_type<eltype,0,-1,-1> { public:
  typedef YA_MatT copy_type; };
// Scalar
template<class eltype> class _ya_temp_type<eltype,0,1,1> { public:
  typedef VM_Scalar copy_type; };
// Row Vector (Unknown size)
template<class eltype> class _ya_temp_type<eltype,0,1,-1> { public:
  typedef YA_RowT copy_type; };
// Column Vector (Unknown size)
template<class eltype> class _ya_temp_type<eltype,0,-1,1> { public:
  typedef YA_VecT copy_type; };
// Row Vector (Known size)
template<class eltype,int ncols> class _ya_temp_type<eltype,0,1,ncols> { public:
  typedef YA_Fix<eltype,1,ncols> copy_type; };
// Column Vector (Known size)
template<class eltype,int nrows> class _ya_temp_type<eltype,0,nrows,1> { public:
  typedef YA_Fix<eltype,nrows,1> copy_type; };
// Matrix (Known size)
template<class eltype,int mem,int nrows,int ncols> class _ya_temp_type { public:
  typedef YA_Fix<eltype,nrows,ncols> copy_type; };

}
#endif
