/***************************************************************************
                                 ya_dimred.h
                             -------------------
                               W. Michael Brown

  Include for Dimensionality Reduction Routines

 __________________________________________________________________________
    This file is part of the Dr.L. Library
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

#ifndef VM_DIMRED_H
#define VM_DIMRED_H

#define DRL_VER "0.2.466"

#ifdef YA_PROGMETER
#include "base/ya_timer.h"
#endif

#include "vm.h"
#include "namegraph.h"
#include "ya_dim_reduce.h"
#include "ya_pca_reduce.h"
#include "ya_mds_reduce.h"
#include "ya_lle_reduce.h"
#include "ya_iso_reduce.h"
#include "ya_ate_reduce.h"
#include "ya_dim_redfun.h"
#include "namegraph.cpp"
#include "ya_dim_reduce.cpp"
#include "ya_pca_reduce.cpp"
#include "ya_mds_reduce.cpp"
#include "ya_lle_reduce.cpp"
#include "ya_iso_reduce.cpp"
#include "ya_ate_reduce.cpp"
#include "ya_dim_redfun.cpp"

#endif
