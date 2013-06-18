/***************************************************************************
                               ya_mds_reduce.cpp
                             -------------------
                               W. Michael Brown

  Multidimensional Scaling

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

#ifndef YA_MDS_REDUCE_CPP
#define YA_MDS_REDUCE_CPP

YALANAMESPACE

template <class eltype>
template <class ya_type1, class ya_type2>
int YAMDSReduce<eltype>::find_map(const ya_type1 &input, 
                   ya_type2 &output, ya_sizet dim, 
                   EigenOptions &eigopts) {
  YA_DEBUG_ERROR(input.rows()==input.cols(),"Input matrix must be square");

  eigopts.dim(dim);
  VM_SymMat m=square(input);
  this->double_center(m);

  eigs(m,eigen_vectors,this->eigen_values,eigopts);

  // Calculate the projection
  output=eigen_vectors*diag(sqrt(this->eigen_values(vmcount(dim))));
  return 0;
}

}
#endif
