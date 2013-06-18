/***************************************************************************
                                 ya_types.h
                             -------------------
                               W. Michael Brown

  Definitions and types for YALA++ classes

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Oct 27 2006
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

#ifndef YA_TYPES_H
#define YA_TYPES_H

YALANAMESPACE

///\var typedef int ya_sizet;
typedef int ya_sizet;

// -------------------------- Traits
template <class mat_type> class YA_Traits;
#define YA_TRAIT YA_Traits<mat_type>
#define YA_TRAIT1 YA_Traits<mat_type1>
#define YA_TRAIT2 YA_Traits<mat_type2>
#define YA_TRAIT3 YA_Traits<mat_type3>
#define YA_TRAIT4 YA_Traits<mat_type4>
#define YATRAIT YA_Traits<ya_type>
#define YATRAIT1 YA_Traits<ya_type1>
#define YATRAIT2 YA_Traits<ya_type2>
#define YATRAIT3 YA_Traits<ya_type3>
#define YATRAIT4 YA_Traits<ya_type4>

// --------------------------- Matrix Iterators
template <class eltype, class mat_type> class YA_Iter;
template <class eltype, class mat_type> class YA_ConstIter;
template <class eltype, class mat_type> class YA_RowIter;
template <class eltype, class mat_type> class YA_ConstRowIter;
#define YA_IterT YA_Traits<mat_type>::iter
#define YA_ConstIterT YA_Traits<mat_type>::citer
#define YA_RowIterT YA_Traits<mat_type>::rowiter
#define YA_ConstRowIterT YA_Traits<mat_type>::crowiter
#define YA_CptrT YA_Traits<mat_type>::cptr
#define YA_ConstCptrT YA_Traits<mat_type>::const_cptr
#define YA_Ref YA_Traits<mat_type>::ref
#define YA_ConstRef YA_Traits<mat_type>::cref

// --------------------------- Base Classes
template <class eltype, class mat_type> class YA_Base;
template <class eltype, class mat_type> class YA_Wrapper;
template <class eltype, class mat_type> class YA_Common;
///\def YA_BaseT
#define YA_BaseT YA_Base<eltype,mat_type>
///\def YA_BaseT2
#define YA_BaseT2 YA_Base<eltype,mat_type2>
///\def YA_BaseT2T
#define YA_BaseT2T YA_Base<eltype2,mat_type2>
///\def YA_BaseT3
#define YA_BaseT3 YA_Base<eltype,mat_type3>
///\def YA_BaseT3T
#define YA_BaseT3T YA_Base<eltype3,mat_type3>
///\def YA_BaseT4
#define YA_BaseT4 YA_Base<eltype,mat_type3>
///\def YA_WrapperT
#define YA_WrapperT YA_Wrapper<eltype,mat_type>
///\def YA_WrapperT2
#define YA_WrapperT2 YA_Wrapper<eltype,mat_type2>
///\def YA_WrapperT3
#define YA_WrapperT3 YA_Wrapper<eltype,mat_type3>
///\def YA_WrapperT4
#define YA_WrapperT4 YA_Wrapper<eltype,mat_type4>

// --------------------------- Matrices
template <class eltype> class YA_Mat;
///\def Template
#define YA_MatT YA_Mat<eltype>
///\def Double Precision
#define YA_MatD YA_Mat<double>
///\def Single Precision
#define YA_MatS YA_Mat<float>
///\def Double Complex
#define YA_MatZ YA_Mat<complex<double> >
///\def Single Complex
#define YA_MatC YA_Mat<complex<float> >
///\def Integer
#define YA_MatI YA_Mat<ya_sizet>
///\def Unsigned Int
#define YA_MatU YA_Mat<unsigned>

template <class eltype> class YA_MatRM;
///\def Template
#define YA_MatRMT YA_MatRM<eltype>
///\def Double Precision
#define YA_MatRMD YA_MatRM<double>
///\def Single Precision
#define YA_MatRMS YA_MatRM<float>
///\def Double Complex
#define YA_MatRMZ YA_MatRM<complex<double> >
///\def Single Complex
#define YA_MatRMC YA_MatRM<complex<float> >
///\def Integer
#define YA_MatRMI YA_MatRM<ya_sizet>
///\def Unsigned Int
#define YA_MatRMU YA_MatRM<unsigned>

template <class eltype, ya_sizet nrows, ya_sizet ncols> class YA_Fix;
///\def Template
#define YA_FixT YA_Fix<eltype,nrows,ncols>

template <class eltype, ya_sizet nrows> class YA_Cols;
///\def Template
#define YA_ColsT YA_Cols<eltype,ncols>
///\def Template
#define YA_ColT YA_Cols<eltype,1>
#define YA_VecT YA_Cols<eltype,1>
///\def Double Precision
#define YA_ColD YA_Cols<double,1>
#define YA_VecD YA_Cols<double,1>
///\def Single Precision
#define YA_ColS YA_Cols<float,1>
#define YA_VecS YA_Cols<float,1>
///\def Double Complex
#define YA_ColZ YA_Cols<complex<double>,1>
#define YA_VecZ YA_Cols<complex<double>,1>
///\def Single Complex
#define YA_ColC YA_Cols<complex<float>,1>
#define YA_VecC YA_Cols<complex<float>,1>
///\def Integer
#define YA_ColI YA_Cols<ya_sizet,1>
#define YA_VecI YA_Cols<ya_sizet,1>
///\def Unsigned Int
#define YA_ColU YA_Cols<unsigned,1>
#define YA_VecU YA_Cols<unsigned,1>

template <class eltype, ya_sizet nrows> class YA_Rows;
///\def Template
#define YA_RowsT YA_Rows<eltype,nrows>
///\def Template
#define YA_RowT YA_Rows<eltype,1>
///\def Double Precision
#define YA_RowD YA_Rows<double,1>
///\def Single Precision
#define YA_RowS YA_Rows<float,1>
///\def Double Complex
#define YA_RowZ YA_Rows<complex<double>,1>
///\def Single Complex
#define YA_RowC YA_Rows<complex<float>,1>
///\def Integer
#define YA_RowI YA_Rows<ya_sizet,1>
///\def Unsigned Int
#define YA_RowU YA_Rows<unsigned,1>

template <class eltype> class YA_Dyn;
///\def Template
#define YA_DynT YA_Dyn<eltype>
///\def Double Precision
#define YA_DynD YA_Dyn<double>
///\def Single Precision
#define YA_DynS YA_Dyn<float>
///\def Double Complex
#define YA_DynZ YA_Dyn<complex<double> >
///\def Single Complex
#define YA_DynC YA_Dyn<complex<float> >
///\def Integer
#define YA_DynI YA_Dyn<ya_sizet>
///\def Unsigned Int
#define YA_DynU YA_Dyn<unsigned>

// --------------------------- Wrappers
template <class eltype, class mat_type> class YA_Trans;
#define YA_TransT YA_Trans<eltype,mat_type>

template <class eltype, class mat_type, int order=0, int nrows=-1, int ncols=-1,
          class reft=eltype&, class creft=const eltype&, int fixt=0> class YA_Wrap;
///\def Template
#define YA_WrapT YA_Wrap<eltype,mat_type,order,nrows,ncols,reft,creft,fixt>
#define YA_WrapC YA_Wrap<eltype,mat_type,order,nrows,ncols,reft,creft>
#define YA_WrapCF YA_Wrap<eltype,mat_type,order,nrows,ncols,reft,creft,1>
#define YA_WrapR YA_Wrap<eltype,mat_type,1,nrows,ncols,reft,creft>
#define YA_WrapRF YA_Wrap<eltype,mat_type,1,nrows,ncols,reft,creft,1>
#define YA_WrapS YA_Wrap<eltype,mat_type,3,nrows,ncols,reft,creft>
#define YA_WrapSF YA_Wrap<eltype,mat_type,3,nrows,ncols,reft,creft,1>

template <class eltype, class mat_type, int nrows=-1, int ncols=-1,
          class reft=eltype&, class creft=const eltype&, int fixt=0> class YA_WRows;
///\def Template
#define YA_WRowsT YA_WRows<eltype,mat_type,nrows,ncols,reft,creft,fixt>
#define YA_WRowsTU YA_WRows<eltype,mat_type,nrows,ncols,reft,creft>
#define YA_WRowsTF YA_WRows<eltype,mat_type,nrows,ncols,reft,creft,1>

// Column major c-array
#define YA_WRAP(N) YA_Wrap<N,N*>
#define YA_WRAP_RM(N) YA_Wrap<N,N*,1>
#define YA_VWRAP(N) YA_Wrap<N,std::vector<double> &>
#define YA_VWRAP_RM(N) YA_Wrap<N,std::vector<double> &>

}

#endif
