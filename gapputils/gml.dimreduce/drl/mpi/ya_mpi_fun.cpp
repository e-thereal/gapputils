/***************************************************************************
                                ya_mpi_fun.cpp
                             -------------------
                               W. Michael Brown

  Misc MPI Routines

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sat Apr 25 2009
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

#ifndef YA_MPI_FUN_CPP
#define YA_MPI_FUN_CPP

YALANAMESPACE

// Get the rank and number of procs
inline void ya_mpi_info(int &rank, int &num_procs) {
  #ifndef YA_MPI
  rank=0;
  num_procs=1;
  #else
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  #endif
}  

// Divide up N iterations into counts per proc and starting index per proc
template <class ya_type1, class ya_type2>
inline void ya_mpi_counts(const int N, const int num_procs, ya_type1 &counts, 
                           ya_type2 &starts) {
  counts.setup(num_procs);
  setall(counts,N/num_procs);
  ya_sizet extra=N%num_procs;
  if (extra>0)
    counts(vmcount(extra))+=1;
  starts=cumsum(counts)-counts;
}  

#ifdef YA_MPI
// Concatenate matrices on each proc to large matrix on rank 0 proc
template <class eltype, class mat_type, class ya_type2, class ya_type3, 
          class ya_type4>
void ya_mpi_concat_rows(YA_BaseT &mat, ya_type2 &full_mat,
                        const int C, const int me, const int num_procs,
                        ya_type3 &counts, ya_type4 &starts) {
  YA_DEBUG_ERROR(me>0 || mat.numel()>0,
                 "Cannot run ya_mpi_concat_rows on matrix with no elements.");
  
  // Handle the empty piece case
  if (counts[me]==0) {
    eltype *one=NULL, *two=NULL;
    MPI_Gatherv(one,counts[me],ya_mpi<eltype>::datatype(),two,counts.c_ptr(),
                starts.c_ptr(),ya_mpi<eltype>::datatype(),0,MPI_COMM_WORLD);
    return;
  }
                          
  // Need row-major c-style matrix for pieces 
  if (YA_TRAIT::VME_MEMORY!=1) {
    YA_MatRMT piece_t(mat);
    ya_mpi_concat_rows(piece_t,full_mat,C,me,num_procs,counts,starts);
    return;
  }
  // Need row-major c-style matrix for full
  int R=starts.back()+counts.back();
  if (YATRAIT2::VME_MEMORY!=1) {
    YA_MatRMT temp_full;
    ya_mpi_concat_rows(mat,temp_full,C,me,num_procs,counts,starts);
    if (me==0)
      full_mat=temp_full;
    return;
  }
  
  YA_ColI mcounts=counts*C;
  YA_ColI mstarts=starts*C;
  if (me==0)
    full_mat.setup(R,C);
  
  // Need piece to be row-major
  if (YA_TRAIT::VME_ORDER!=1 && (mat.rows()>1 || mat.cols()>1))
    ip_transpose(mat);
  MPI_Gatherv(mat.c_ptr(),mcounts[me],ya_mpi<eltype>::datatype(),
              full_mat.c_ptr(),mcounts.c_ptr(),mstarts.c_ptr(),
              ya_mpi<eltype>::datatype(),0,MPI_COMM_WORLD);
  // If full matrix is not row-major, transpose
  if (YATRAIT2::VME_ORDER!=1 && C>1 && me==0) {
    full_mat.reshape(C,R);
    ip_transpose(full_mat);
  }
}                                    

/// Return the minimum, maximum, mean, and standard deviation
template<class eltype, class mat_type, class stype>
void mpi_vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean,
               stype &vstd) {
  ya_sizet myN=mat.numel();
  ya_sizet N;
  MPI_Allreduce(&myN,&N,1,ya_mpi<ya_sizet>::datatype(),MPI_SUM,MPI_COMM_WORLD);

  if (mat.numel()>0) {
    vstat(mat,vmin,vmax,vmean);
    vmean*=mat.numel()/static_cast<stype>(N);
  } else {
    vmin=std::numeric_limits<eltype>::max();
    vmax=std::numeric_limits<eltype>::min();
    vmean=0;
    vstd=0;
  }

  stype sbuffer;
  MPI_Allreduce(&vmean,&sbuffer,1,ya_mpi<stype>::datatype(),MPI_SUM,
                MPI_COMM_WORLD);
  vmean=sbuffer;
  eltype ebuffer;
  MPI_Allreduce(&vmin,&ebuffer,1,ya_mpi<eltype>::datatype(),MPI_MIN,
                MPI_COMM_WORLD);
  vmin=ebuffer;
  MPI_Allreduce(&vmax,&ebuffer,1,ya_mpi<eltype>::datatype(),MPI_MAX,
                MPI_COMM_WORLD);
  vmax=ebuffer;

  if (mat.numel()>0) {
    YA_Mat<stype> sm=vm_cast<stype>::sc(mat)-vmean;
    sm=sm.dot_star(sm);
    vstd=total(sm)/static_cast<stype>(N);  
  }
  MPI_Allreduce(&vstd,&sbuffer,1,ya_mpi<stype>::datatype(),MPI_SUM,
                MPI_COMM_WORLD);
  vstd=sqrt(sbuffer);
}

/// Return the minimum, maximum, mean, and standard deviation
/** \sa vstat **/
template<class eltype, class mat_type, class stype>
void mpi_vstat(const YA_BaseT &mat, const int C, YA_RowT &vmin, YA_RowT &vmax, 
               YA_Rows<stype,1> &vmean, YA_Rows<stype,1> &vstd) {
  ya_sizet myN=mat.rows();
  ya_sizet N;
  MPI_Allreduce(&myN,&N,1,ya_mpi<ya_sizet>::datatype(),MPI_SUM,MPI_COMM_WORLD);

  if (mat.numel()>0) {
    vstat(mat,vmin,vmax,vmean);
    vmean*=mat.rows()/static_cast<stype>(N);
  } else {
    vmin.setup(C); vmax.setup(C); vmean.setup(C); vstd.setup(C);
    setall(vmin,std::numeric_limits<eltype>::max());
    setall(vmax,std::numeric_limits<eltype>::min());
    setall(vmean,static_cast<stype>(0));
    setall(vstd,static_cast<stype>(0));
  }
  
  YA_Rows<stype,1> sbuffer(1,C);
  MPI_Allreduce(vmean.c_ptr(),sbuffer.c_ptr(),C,ya_mpi<stype>::datatype(),
                MPI_SUM,MPI_COMM_WORLD);
  vmean=sbuffer;
  YA_RowT ebuffer(1,C);
  MPI_Allreduce(vmin.c_ptr(),ebuffer.c_ptr(),C,ya_mpi<eltype>::datatype(),
                MPI_MIN,MPI_COMM_WORLD);
  vmin=ebuffer;
  MPI_Allreduce(vmax.c_ptr(),ebuffer.c_ptr(),C,ya_mpi<eltype>::datatype(),
                MPI_MAX,MPI_COMM_WORLD);
  vmax=ebuffer;

  if (mat.numel()>0) {
    YA_Mat<stype> sm=vm_cast<stype>::sc(mat)-rowrep(vmean,mat.rows());
    sm=sm.dot_star(sm);
    vstd=sum(sm)/static_cast<stype>(N);  
  }
  MPI_Allreduce(vstd.c_ptr(),sbuffer.c_ptr(),C,ya_mpi<stype>::datatype(),
                MPI_SUM,MPI_COMM_WORLD);
  vstd=sqrt(sbuffer);
}

#endif

}
#endif
