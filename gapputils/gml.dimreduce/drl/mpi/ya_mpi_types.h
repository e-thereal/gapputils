/***************************************************************************
                                ya_mpi_types.h
                             -------------------
                               W. Michael Brown

  MPI Definitions and types for YALA++ classes

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Apr 24 2009
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

#ifndef YA_MPI_TYPES_H
#define YA_MPI_TYPES_H

#ifdef YA_MPI

#include "mpi.h"

YALANAMESPACE

template <class eltype> struct ya_mpi;
template <> struct ya_mpi<double> 
  { inline static MPI_Datatype datatype() { return MPI_DOUBLE; } };
template <> struct ya_mpi<float>
  { inline static MPI_Datatype datatype() { return MPI_FLOAT; } };
template <> struct ya_mpi<ya_sizet>
  { inline static MPI_Datatype datatype() { return MPI_INT; } };

}
#endif

#endif
