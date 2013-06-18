/***************************************************************************
                                 vm_types.h
                             -------------------
                               W. Michael Brown

  Friends for VecMat classes

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Feb 16 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

  template <class ab, class ac> friend class YA_Base;
  template <class ab, class ac> friend class YA_Common;
  template <class ac> friend class VecMat;
  template <class ac, ya_sizet cn> friend class YA_Rows;
  template <class ac, ya_sizet cn> friend class YA_Cols;
  template <class ac, ya_sizet cn, ya_sizet ct> friend class YA_Fix;
  template <class ac> friend class VecMatSym;
  template <class ac, ya_sizet cn> friend class VecMatSymF;
  template <class ab, class ac> friend class VecMatSparse;
  template <class ac> friend class YA_Dyn;
  template <class ac> friend class VecMatScalar;
  template <class ac> friend class VecMatCount;
  template <class ac> friend class VecMatCountV;
  template <class ac, int cn> friend class VMCountFromI;
  template <class ac, int cn> friend class VecMatConstV;

  template <class ac, class mt> friend class VecMatC;
  template <class ac, class mt> friend class VecMatR;
  template <class ac, class mt> friend class YA_Trans;
  template <class ac, class mt> friend class VecMatDiag;
  template <class ac, class mt, class rt,class ct> friend class VecMatS;
