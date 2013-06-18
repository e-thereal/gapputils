/***************************************************************************
                                 vm_stat.cpp
                             -------------------
                               W. Michael Brown

  Statistics functions

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Aug 15 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_STAT_CPP
#define VM_STAT_CPP

YALANAMESPACE

// ---- Covarience
template<class eltype, class mat_type, class ya_type2>
inline VM_SymMat cov(const YA_BaseT &m, const ya_type2 &t) {
  YA_DEBUG_ERROR(m.numel()==t.numel() && m.numel()>1,
                 "Number of elements is different in m and t");
  YA_MatT mat(m.numel(),2);
  mat(":",0)=m-total(m)/m.rows();
  mat(":",1)=t-total(t)/t.rows();
  return VM_SymMat( (mat.T()*mat)/static_cast<eltype>(mat.rows()-1) );
}

// ---- Correlation Coefficients
template<class eltype, class mat_type, class ya_type2>
inline eltype corrcoef(const YA_BaseT &m, const ya_type2 &t) {
  VecMatSymF<double,2> mat(cov(m,t));
  return mat(1)/sqrt(mat(0)*mat(2));
}

// ---- Compute the squared correlation coefficient
template<class eltype, class mat_type, class ya_type2>
double rsquared(const YA_BaseT &m, const ya_type2 &two) {
  YA_DEBUG_ERROR(length(m)==length(two),
                 "Number of elements in vectors are different");

  double yave=total(m)/length(m);
  double num=total(square(two-m));
  double dem=total(square(m-yave));
  return 1.0-num/dem;
}

// Compute the root mean squared deviation for two matrices (rows are points)
template<class eltype, class mat_type, class ya_type2>
double rmsd(const YA_BaseT &one, const ya_type2 &two) {
  YA_DEBUG_ERROR(one.cols()==two.cols() && one.rows()==two.rows(),
                 "Matrix dimensions do not match");

  YA_MatT temp=(two-one);
  return sqrt(total(temp^2.0)/one.rows());
}

// Calculate residual between a distance matrix and distances between new_crds
// dims must be sorted
template<class eltype, class mat_type, class ya_type2, class ya_type3, 
         class ya_type4, class vm_type5>
void dist_residual(const YA_BaseT &original_distances, const ya_type2 &new_crds,
                   const ya_type3 &dims, ya_type4 &corrcoef_vec,
                   vm_type5 &rsquared_vec) {
  ya_sizet len=new_crds.rows()*new_crds.rows();
  YA_DEBUG_ERROR(original_distances.numel()==len,
                 "Cannot compute dist residual for different sized matrices.");

  corrcoef_vec.setup(dims.numel());
  rsquared_vec.setup(dims.numel());
  YA_MatT new_dists=VM_SymMat(dist2_mat(new_crds(":",0)));

  YA_VecI no_inf=find(original_distances!=VM_INF);

  ya_sizet dim=1;
  //#pragma omp parallel
  {
  ya_sizet dN=dims.numel();
  for (ya_sizet i=0; i<dN; i++) {
    while(dim<dims(i)) {
      //#pragma omp for schedule(static,1)
      ya_sizet jN=new_dists.rows();
      for (ya_sizet j=0; j<jN; j++)
        for (ya_sizet k=j+1; k<jN; k++) {
          double nd=new_crds(j,dim)-new_crds(k,dim);
          nd*=nd;
          new_dists(j,k)+=nd;
          new_dists(k,j)+=nd;
        }
       dim++;
    }
    //#pragma omp single
    {
    YA_VecT new_no_inf=sqrt(new_dists(no_inf));
    corrcoef_vec(i)=1.0-pow(corrcoef(original_distances(no_inf),new_no_inf),
                            2.0);
    rsquared_vec(i)=1.0-rsquared(original_distances(no_inf),new_no_inf);
    }
  }
  }
}

/// Calculate residual between a distance matrix and distances between new_crds
/** Distances between each point and the landmark points are calculated for
  * new_crds **/
template<class eltype, class mat_type, class ya_type2, class ya_type3, 
         class ya_type4, class vm_type5, class vm_type6>
void dist_residual(const YA_BaseT &original_distances, const ya_type2 &new_crds,
                   const ya_type3 &dims,ya_type4 &corrcoef_vec,
                   vm_type5 &rsquared_vec, const vm_type6 &landmarks) {
  ya_sizet len=new_crds.rows()*landmarks.numel();
  YA_DEBUG_ERROR(original_distances.numel()==len,
                 "Cannot compute dist residual for different sized matrices.");

  corrcoef_vec.setup(dims.numel());
  rsquared_vec.setup(dims.numel());
  YA_MatT new_dists(vmzeros(landmarks.numel(),new_crds.rows()));

  YA_VecI no_inf=find(original_distances!=VM_INF);

  ya_sizet dim=0;
  ya_sizet dN=dims.numel();
  ya_sizet nR=new_dists.rows();
  ya_sizet nC=new_dists.cols();
  for (ya_sizet i=0; i<dN; i++) {
    while(dim<dims(i)) {
      for (ya_sizet j=0; j<nR; j++)
        for (ya_sizet k=0; k<nC; k++) {
          double nd=new_crds(landmarks(j),dim)-new_crds(k,dim);
          nd*=nd;
          new_dists(j,k)+=nd;
        }
       dim++;
    }
    YA_VecT new_no_inf=sqrt(new_dists(no_inf));
    corrcoef_vec(i)=1.0-pow(corrcoef(original_distances(no_inf),new_no_inf),
                            2.0);
    rsquared_vec(i)=1.0-rsquared(original_distances(no_inf),new_no_inf);
  }
}

}
#endif
