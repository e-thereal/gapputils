/***************************************************************************
                                 ya_binary.h
                             -------------------
                               W. Michael Brown

  Binary operations not using template expressions
 
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

#ifndef YA_BINARY_H
#define YA_BINARY_H

/*! \file */

using namespace yala;

template<class eltype, class mat_type>
YA_BaseT& operator+=(YA_BaseT &mat, const eltype num);

template<class eltype, class mat_type>
const YA_WrapperT& operator+=(const YA_WrapperT &mat, const eltype num);

template<class eltype, class mat_type>
YA_BaseT& operator-=(YA_BaseT &mat, const eltype num);

template<class eltype, class mat_type>
const YA_WrapperT& operator-=(const YA_WrapperT &mat, const eltype num);

template<class eltype, class mat_type>
YA_BaseT& operator*=(YA_BaseT &mat, const eltype num);

template<class eltype, class mat_type>
const YA_WrapperT& operator*=(const YA_WrapperT &mat, const eltype num);

template<class eltype, class mat_type>
YA_BaseT& operator/=(YA_BaseT &mat, const eltype num);

template<class eltype, class mat_type>
const YA_WrapperT& operator/=(const YA_WrapperT &mat, const eltype num);

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator+=(YA_BaseT &mat, const ya_type2 &two);

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator+=(const YA_WrapperT &mat, const ya_type2 &two);

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator-=(YA_BaseT &mat, const ya_type2 &two);

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator-=(const YA_WrapperT &mat, const ya_type2 &two);

template<class eltype, class mat_type, class ya_type2>
YA_BaseT& operator*=(YA_BaseT &mat, const ya_type2 &two);

template<class eltype, class mat_type, class ya_type2>
const YA_WrapperT& operator*=(const YA_WrapperT &mat, const ya_type2 &two);


#endif
