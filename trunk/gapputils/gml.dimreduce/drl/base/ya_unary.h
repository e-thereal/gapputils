/***************************************************************************
                                  ya_unary.h
                             -------------------
                               W. Michael Brown

  Unary operations not using template expressions
 
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

#ifndef YA_UNARY_H
#define YA_UNARY_H

/*! \file */

using namespace yala;

template<class eltype, class mat_type>
inline void operator++(YA_BaseT &mat,int);

template<class eltype, class mat_type>
inline void operator++(const YA_WrapperT &mat,int);

template<class eltype, class mat_type>
inline YA_BaseT& operator++(YA_BaseT &mat);

template<class eltype, class mat_type>
inline const YA_WrapperT& operator++(const YA_WrapperT &mat);

template<class eltype, class mat_type>
inline void operator--(YA_BaseT &mat,int);

template<class eltype, class mat_type>
inline void operator--(const YA_WrapperT &mat,int);

template<class eltype, class mat_type>
inline YA_BaseT& operator--(YA_BaseT &mat);

template<class eltype, class mat_type>
inline const YA_WrapperT& operator--(const YA_WrapperT &mat);

#endif
