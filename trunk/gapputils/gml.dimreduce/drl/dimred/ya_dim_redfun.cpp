/***************************************************************************
                              ya_dim_redfun.cpp
                             -------------------
                               W. Michael Brown

  Miscellaneous functions for dimensionality reduction

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Tue Feb 19 2008
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_DIM_REDFUN_CPP
#define YA_DIM_REDFUN_CPP

#ifdef YA_MPI
#include "mpi.h"
#endif

YALANAMESPACE

template<class eltype, class mat_type, class ya_type2>
void _point_pca(const YA_BaseT &input, const ya_sizet k,
                EigenOptions &eigopts, ya_type2 &output, const bool residual,
                const int verbose_out, const int me, const int num_procs, 
                const YA_ColI &counts, const YA_ColI &starts) {
  YA_DEBUG_ERROR(k>=0 && k<input.rows(),
                 "More neighbors than datapoints for point_pca");

  int verbose=verbose_out;
  #ifdef YA_MPI
    if (me!=0)
      verbose=0;
    else if (num_procs>1 && verbose>1)
      verbose=1;
    #ifdef YA_PROGMETER
    if (verbose)
      cout << "Performing point-PCA on " << num_procs << " processors.\n";
    #endif
  #endif

  YA_MatI neighbors;
  YA_MatD dists;
  if (num_procs==1) {
    kneighbors(input,k,neighbors,dists,verbose);
    neighbors=YA_MatI(concat(vm_cast<ya_sizet>::sc(vmcount(input.rows()).T()),
                      neighbors));
  } else if (counts[me]>0)
    kneighbors(input(vmcount(starts[me],":",starts[me]+counts[me]-1),":"),
               input,k+1,neighbors,dists,verbose);

  #ifdef YA_MPI
  if (num_procs>1 && verbose_out>1) {
    #ifdef YA_PROGMETER
    YA_TimeKeeper mtk;
    if (verbose>0) {
      cerr << "Waiting on other procs...";
      mtk.start();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbose>0) {
      mtk.end();
      cerr << "Done. " << mtk << endl;
    } 
    #endif
    if (me==0) cerr << "Neighbor Distance Stats:\n";
    eltype dmin, dmax, dmean, dstd;
    mpi_vstat(dists(":",vmcount(k)+1),dmin,dmax,dmean,dstd);
    if (me==0)
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
  }
  #endif
  dists.clear();
  output.setup(counts[me],input.cols());

  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Performing Point PCA:",70,counts[0],false);
  }
  #endif

  #pragma omp parallel
  {
  YA_RowT column_means;
  YA_MatT input_cen;
  VM_SymMat covmat;
  YA_RowT eigens;
  #pragma omp for
  for (ya_sizet i=0; i<counts[me]; i++) {
    // Column center the matrix
    column_means=sum(input(neighbors(i,":"),":")/static_cast<eltype>(k+1));
    input_cen=input(neighbors(i,":"),":")-rowrep(column_means,k+1);
    covmat=input_cen.T()*input_cen;

    eigs(covmat,eigens,eigopts);
    output(i,":")=eigens;
    #ifdef YA_PROGMETER
    if (verbose)
      pm.iterate();
    #endif
  }
  }
  neighbors.clear();
  
  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  YA_TimeKeeper mtk;
  if (num_procs>1)
    if (verbose>0) {
      cerr << "Waiting on other procs...";
      mtk.start();
    }
  #endif
  #ifdef YA_MPI
  if (num_procs>1)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
  #ifdef YA_PROGMETER
  if (num_procs>1 && verbose>0) {
    mtk.end();
    cerr << "Done. " << mtk << endl;
  } 
  #endif

  if (residual && counts[me]>0) {                        
    YA_VecT totals=sum(output.T());
    output=output.dot_div(colrep(totals,output.cols()));
    ya_sizet iC=output.cols();
    for (ya_sizet i=1; i<iC-1; i++)
      output(":",i)+=output(":",i-1);
    output(":",iC-1)=1.0;
    output=1.0-output;
  }
}

template<class eltype, class mat_type, class vmtype2>
void _point_pca_ep(const YA_BaseT &input, const eltype epsilon,
                   EigenOptions &eigopts, vmtype2 &output, 
                   const bool residual, const int verbose_out, const int me, 
                   const int num_procs, const YA_ColI &counts, 
                   const YA_ColI &starts) {
  YA_DEBUG_ERROR(epsilon>0,
                 "Epsilon must be greater than 0 for point_pca");

  int verbose=verbose_out;
  #ifdef YA_MPI
    if (me!=0)
      verbose=0;
    else if (num_procs>1 && verbose>1)
      verbose=1;
    #ifdef YA_PROGMETER
    if (verbose)
      cout << "Performing point-PCA on " << num_procs << " processors.\n";
    #endif
  #endif

  std::vector<YA_DynI> neighbors;
  std::vector<YA_Dyn<eltype> > dists;
  if (num_procs==1) {
    eneighbors(input,epsilon,neighbors,dists,verbose);
    int iR=input.rows();
    for (ya_sizet i=0; i<iR; i++)
      neighbors[i].push_back(i);
  } else if (counts[me]>0)
    eneighbors(input(vmcount(starts[me],":",starts[me]+counts[me]-1),":"),
               input,epsilon,neighbors,dists,verbose);

  dists.clear();
  #ifdef YA_MPI
  if (num_procs>1 && verbose_out>1) {
    #ifdef YA_PROGMETER
    YA_TimeKeeper mtk;
    if (verbose>0) {
      cerr << "Waiting on other procs...";
      mtk.start();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbose>0) {
      mtk.end();
      cerr << "Done. " << mtk << endl;
    }
    #endif 
    if (me==0) cerr << "Neighbor Counts Stats:\n";
    ya_sizet dmin, dmax;
    eltype dmean, dstd;
    ya_sizet n=neighbors.size();
    YA_ColI ncounts;
    if (n>0)
      ncounts.setup(n);
    for (ya_sizet i=0; i<n; i++)
      ncounts(i)=neighbors[i].numel()-1;
    mpi_vstat(ncounts,dmin,dmax,dmean,dstd);
    if (me==0)
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
  }
  #endif

  output.setup(counts[me],input.cols());

  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Performing Point PCA:",70,counts[0],false);
  }
  #endif

  #pragma omp parallel
  {
  YA_RowT column_means;
  YA_MatT input_cen;
  VM_SymMat covmat;
  YA_RowT eigens;
  ya_sizet NN=counts[me];
  #pragma omp for
  for (ya_sizet i=0; i<NN; i++) {
    ya_sizet k=neighbors[i].numel();
    column_means=sum(input(neighbors[i],":")/static_cast<eltype>(k));
    input_cen=input(neighbors[i],":")-rowrep(column_means,k);
    covmat=input_cen.T()*input_cen;

    eigs(covmat,eigens,eigopts);
    output(i,":")=eigens;
    #ifdef YA_PROGMETER
    if (verbose)
      pm.iterate();
    #endif
  }
  }
  neighbors.clear();

  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  YA_TimeKeeper mtk;
  if (num_procs>1 && verbose) {
    cerr << "Waiting on other procs...";
    mtk.start();
  }
  #endif
  #ifdef YA_MPI
  if (num_procs>1)
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
  #ifdef YA_PROGMETER
  if (num_procs>1 && verbose>0) {
    mtk.end();
    cerr << "Done. " << mtk << endl;
  } 
  #endif

  if (residual && counts[me]>0) {
    YA_VecT totals=sum(output.T());
    output=output.dot_div(colrep(totals,output.cols()));
    ya_sizet iC=output.cols();
    for (ya_sizet i=1; i<iC-1; i++)
      output(":",i)+=output(":",i-1);
    output(":",iC-1)=1.0;
    output=1.0-output;
  }
}

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void point_pca(const YA_BaseT &input, const ya_sizet k, const ya_type2 &dims,
               EigenOptions &eigopts, ya_type3 &output,const int verbose_out) {
  int me, num_procs;
  ya_mpi_info(me,num_procs);
  eigopts.dim(input.cols());

  YA_ColI counts, starts;
  ya_mpi_counts(input.rows(),num_procs,counts,starts);
  
  _point_pca(input,k,eigopts,output,true,verbose_out,me,num_procs,counts,
             starts);
  
  if (num_procs>1) {
    YA_MatRMT piece_t;
    #ifdef YA_MPI
    if (counts[me]>0)
      piece_t=output(":",dims-1);
    ya_mpi_concat_rows(piece_t,output,dims.numel(),me,num_procs,counts,starts);
    #endif
  } else if (input.cols()!=dims.numel())
    output=YA_MatT(output(":",dims-1));    
}

template<class eltype, class mat_type, class ya_type2>
void point_pca(const YA_BaseT &input, const ya_sizet k,
               EigenOptions &eigopts, ya_type2 &output,const int verbose_out) {
  point_pca(input,k,vmcount(input.cols())+1,eigopts,output,verbose_out);
}

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  const ya_type2 &dims, EigenOptions &eigopts, 
                  ya_type3 &output, const int verbose_out) {
  YA_DEBUG_ERROR(epsilon>0,
                 "Epsilon must be greater than 0 for point_pca");

  int me, num_procs;
  ya_mpi_info(me,num_procs);

  YA_ColI counts, starts;
  ya_mpi_counts(input.rows(),num_procs,counts,starts);
  
  eigopts.dim(input.cols());
  
  _point_pca_ep(input,epsilon,eigopts,output,true,verbose_out,me,num_procs,
                counts,starts);

  if (num_procs>1) {
    YA_MatRMT piece_t;
    #ifdef YA_MPI
    if (counts[me]>0)
      piece_t=output(":",dims-1);
    ya_mpi_concat_rows(piece_t,output,dims.numel(),me,num_procs,counts,starts);
    #endif
  } else if (input.cols()!=dims.numel())
    output=YA_MatT(output(":",dims-1));       
}

template<class eltype, class mat_type, class ya_type2>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  EigenOptions &eigopts, 
                  ya_type2 &output, const int verbose_out) {
  point_pca_ep(input,epsilon,vmcount(input.cols())+1,eigopts,output,
               verbose_out);
}

template<class eltype, class mat_type, class ya_type2>
void point_pca(const YA_BaseT &input, const ya_sizet k, const ya_type2 &dims,
               EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin,
               YA_RowD &vmax, YA_RowD &vstd, const int verbose) {
  int me, num_procs;
  ya_mpi_info(me,num_procs);
  eigopts.dim(input.cols());

  YA_ColI counts, starts;
  ya_mpi_counts(input.rows(),num_procs,counts,starts);
  
  YA_MatT output;
  _point_pca(input,k,eigopts,output,true,verbose,me,num_procs,counts,starts);
  
  if (input.cols()!=dims.numel() && counts[me]>0)
    output=YA_MatT(output(":",dims-1));    

  #ifdef YA_MPI
  if (num_procs>1)
    mpi_vstat(output,dims.numel(),vmin,vmax,vmean,vstd);
  else
  #endif
    vstat(output,vmin,vmax,vmean,vstd);
}

template<class eltype, class mat_type>
void point_pca(const YA_BaseT &input, const ya_sizet k,
               EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin,
               YA_RowD &vmax, YA_RowD &vstd, const int verbose) {
  point_pca(input,k,vmcount(input.cols())+1,eigopts,vmean,vmin,vmax,vstd,
            verbose);
}

template<class eltype, class mat_type, class ya_type2>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  const ya_type2 &dims, EigenOptions &eigopts, YA_RowD &vmean, 
                  YA_RowD &vmin, YA_RowD &vmax, YA_RowD &vstd, 
                  const int verbose) {
  YA_DEBUG_ERROR(epsilon>0,
                 "Epsilon must be greater than 0 for point_pca");

  int me, num_procs;
  ya_mpi_info(me,num_procs);

  YA_ColI counts, starts;
  ya_mpi_counts(input.rows(),num_procs,counts,starts);
  
  eigopts.dim(input.cols());
  
  YA_MatT output;
  _point_pca_ep(input,epsilon,eigopts,output,true,verbose,me,num_procs,
                counts,starts);

  if (input.cols()!=dims.numel() && counts[me]>0)
    output=YA_MatT(output(":",dims-1));    

  #ifdef YA_MPI
  if (num_procs>1)
    mpi_vstat(output,dims.numel(),vmin,vmax,vmean,vstd);
  else
  #endif
    vstat(output,vmin,vmax,vmean,vstd);
}

template<class eltype, class mat_type>
void point_pca_ep(const YA_BaseT &input, const eltype epsilon, 
                  EigenOptions &eigopts, YA_RowD &vmean, YA_RowD &vmin, 
                  YA_RowD &vmax, YA_RowD &vstd, const int verbose) {
  point_pca_ep(input,epsilon,vmcount(input.cols())+1,eigopts,vmean,vmin,vmax,
               vstd,verbose);
}

}
#endif
