/***************************************************************************
                                     vm.h
                             -------------------
                               W. Michael Brown

  Definitions and types for VecMat classes
  
  This is the include file for all vector/matrix classes
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Dec 6 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_LIB_H
#define VM_LIB_H

#ifdef _MSC_VER
#pragma warning( disable: 4522 4812 4503 4996 )
#endif

#include <string>
//using namespace std;

#define YALANAMESPACE namespace yala {
#define YALA_MAJOR_VERSION "0.2"
#define YALA_VERSION "0.2.466"


#include "base/ya_constants.h"
#include "base/ya_types.h"
#include "mpi/ya_mpi_types.h"
#include "vm_helper.h"
#include "vm_types.h"
#include "base/ya_template_meta.h"

// ------------------- Traits
#include "common/ya_base_traits.h"
#include "common/ya_wrapper_traits.h"
#include "matrices/ya_mat_traits.h"
#include "matrices/ya_matrm_traits.h"
#include "matrices/ya_fix_traits.h"
#include "matrices/ya_cols_traits.h"
#include "matrices/ya_rows_traits.h"
#include "matrices/ya_dyn_traits.h"
#include "wrappers/ya_wrap_traits.h"
#include "wrappers/ya_wrows_traits.h"
#include "wrappers/ya_trans_traits.h"

#include "math_random.h"

#include "base/ya_misc.h"
#include "base/ya_debug.h"
#include "base/ya_error.h"
#include "base/ya_help.h"
#include "iters/ya_iter_misc.h"
#include "iters/ya_iter.h"
#include "iters/ya_rowiter.h"
#include "blas/ya_cblas.h"
#include "memory/ya_loop.h"
#include "memory/ya_direct_mem.h"
#include "memory/ya_copy_matrix.h"
#include "memory/ya_compare_matrix.h"
#include "memory/ya_zero_matrix.h"
#include "memory/ya_allocate.h"
#include "memory/ya_temporary.h"
#include "base/ya_stringmat.h"
#include "base/ya_io.h"

#ifdef VM_INTEL_VML
#include "vm_intel_vml.h"
#endif

#include "vm_unary_expr.h"
#include "vm_binary_expr.h"
#include "vm_matrix_expr.h"
#include "common/ya_base.h"
#include "common/ya_wrapper.h"
#include "common/ya_common.h"
#include "common/ya_wrapper_common.h"
#include "matrices/ya_mat.h"
#include "matrices/ya_matrm.h"
#include "matrices/ya_fix.h"
#include "matrices/ya_cols.h"
#include "matrices/ya_rows.h"
#include "matrices/ya_dyn.h"
#include "wrappers/ya_wrap.h"
#include "wrappers/ya_wrows.h"
#include "wrappers/ya_trans.h"
#include "vecmatopts.h"
#include "vecmatr.h"
#include "vecmatsym.h"
#include "vecmatsymf.h"
#include "vecmatc.h"
#include "vecmats.h"
#include "vecmatdiag.h"
#include "vecmatrei.h"
#include "vm_reshape.h"
#include "vm_cast.h"
#include "vm_repmat.h"
#include "vm_concat.h"
#include "vm_concatr.h"
#include "vecmatsparse.h"
#include "vecmatscalar.h"
#include "vecmatconst.h"
#include "base/ya_unary.h"
#include "base/ya_binary.h"
#include "base/ya_rand.h"
#include "base/ya_fun.h"
#include "base/ya_fun_wrap.h"
#include "mpi/ya_mpi_fun.h"
#include "vm_fun.h"
#include "vm_cblas_op5.h"
#include "vm_stat.h"
#include "vm_decompose.h"

#include "base/ya_stringmat.cpp"
#include "base/ya_io.cpp"
#include "solve/ya_int_eig.h"
#include "vecmatopts.cpp"
#include "vecmatr.cpp"
#include "vecmatsym.cpp"
#include "vecmatsymf.cpp"
#include "vecmatc.cpp"
#include "vecmats.cpp"
#include "vecmatdiag.cpp"
#include "vecmatrei.cpp"
#include "vecmatsparse.cpp"
#include "vecmatscalar.cpp"
#include "base/ya_rand.cpp"
#include "base/ya_unary.cpp"
#include "base/ya_binary.cpp"
#include "base/ya_fun.cpp"
#include "base/ya_fun_wrap.cpp"
#include "mpi/ya_mpi_fun.cpp"
#include "solve/ya_int_eig.cpp"
#include "vm_decompose.cpp"
#include "vm_fun.cpp"
#include "vm_stat.cpp"

#endif
