/***************************************************************************
                              vm_decompose.cpp
                             -------------------
                               W. Michael Brown

  Routines for matrix decomposition

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

#ifndef VMDECOMPOSE_CPP
#define VMDECOMPOSE_CPP

YALANAMESPACE

// ------------------ Divide and Conquer

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eig(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    vectors=m;
    long info;
    syevd('V','L',vectors,values,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

template<class eltype, class mat_type, class ya_type2>
void eig(const YA_BaseT &m, ya_type2 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors(m);
    long info;
    syevd('N','L',vectors,values,info);
  } else 
  #endif
  {
    YA_MatT vectors;
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

// ------------------ Simple Driver

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigev(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    vectors=m;
    long info;
    syev('V','L',vectors,values,info);
  } else 
  #endif
  {
    YA_MatT vectors;
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

template<class eltype, class mat_type, class ya_type2>
void eigev(const YA_BaseT &m, ya_type2 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors(m);
    long info;
    syev('N','L',vectors,values,info);
  } else 
  #endif
  {
    YA_MatT vectors;
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

// ------------------ RRR Driver

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigrrr(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values,
            eltype tol) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    YA_Cols<long,1> isuppz;
    long info,mout,il,iu;
    eltype vl,vu;
    YA_MatT input(m);
    syevr('V','A','L',input,vl,vu,il,iu,tol,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

template<class eltype, class mat_type, class ya_type2>
void eigrrr(const YA_BaseT &m, ya_type2 &values,eltype tol) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors;
    YA_Cols<long,1> isuppz;
    long info,mout,il,iu;
    eltype vl,vu;
    YA_MatT input(m);
    syevr('N','A','L',input,vl,vu,il,iu,tol,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    YA_MatT vectors;
    VM_SymMat work(m);
    ya_simple_eig(work,vectors,values);
  }
}

template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigrrr(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, ya_type3 &values,
            eltype tol) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    YA_Cols<long,1> isuppz;
    long info,mout;
    long iu=m.rows();
    long il=m.rows()-dim+1;
    eltype vl,vu;
    YA_MatT input(m);
    syevr('V','I','L',input,vl,vu,il,iu,tol,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectorst;
    YA_VecT valuest;
    ya_simple_eig(work,vectorst,valuest);
    vectors=vectorst(":",vmcount(m.rows()-dim,":",m.rows()-1));
    values=valuest(vmcount(m.rows()-dim,":",m.rows()-1));
  }
}

template<class eltype, class mat_type, class ya_type2>
void eigrrr(const YA_BaseT &m, ya_sizet dim, ya_type2 &values,eltype tol) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors;
    YA_Cols<long,1> isuppz;
    long info,mout;
    eltype vl,vu;
    long iu=m.rows();
    long il=m.rows()-dim+1;
    YA_MatT input(m);
    syevr('N','I','L',input,vl,vu,il,iu,tol,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectors;
    YA_VecT valuest;
    ya_simple_eig(work,vectors,valuest);
    values=valuest(vmcount(m.rows()-dim,":",m.rows()-1));
  }
}

// ------------------ Power decomposition
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                 ya_type3 &values, eltype tol, ya_sizet max_iters) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    YA_Cols<long,1> ifail(m.rows());
    long info, mout;
    long il=m.rows()-dim+1;
     long iu=m.rows();
    eltype vl,vu;
    YA_MatT input(m);
    syevx('V','I','L',input,vl,vu,il,iu,tol,mout,values,vectors,ifail,info);
    ip_fliplr(vectors);
    ip_flipcm(values);
  } else 
  #endif
  {
    MathRandom<MathMersenneTwister> rng;
    _powereigens(m,dim,vectors,values,tol,max_iters,rng);
  }
}

template<class eltype, class mat_type, class ya_type2>
void powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &values, 
                 eltype tol, ya_sizet max_iters) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors;
    YA_Cols<long,1> ifail(m.rows());
    long info, mout;
    long il=m.rows()-dim+1;
     long iu=m.rows();
    eltype vl,vu;
    YA_MatT input(m);
    syevx('N','I','L',input,vl,vu,il,iu,tol,mout,values,vectors,ifail,info);
    ip_fliplr(vectors);
    ip_flipcm(values);
  } else 
  #endif
  {
    YA_MatT vectors;
    MathRandom<MathMersenneTwister> rng;
    _powereigens(m,dim,vectors,values,tol,max_iters,rng);
  }
}

template<class eltype, class mat_type, class ya_type2, class ya_type3, 
         class Rng>
int _powereigens(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                 ya_type3 &values, eltype tol, ya_sizet max_iters,
                 Rng &rng) {
  YA_DEBUG_ERROR(dim>0,"Dimensions for powereigens must be non-zero.");
  YA_DEBUG_ERROR(m.rows()==m.cols(),
                 "Matrix for powereigens must be square.");
  YA_DEBUG_ERROR(dim<=m.rows(),"powereigens dimensions greater than matrix.");
                    
  int err_flag = 0;
  vectors.setup(m.rows(),dim);
  values.setup(dim);
  if (tol==0.0)
    tol=1e-8;
  eltype pe_eps=1.0-tol;
  YA_MatT uhat(m.rows());
  for (ya_sizet i=0; i<dim; i++) {
    ip_rand(uhat,rng);
    uhat/=l2_norm(uhat);
    ya_sizet k;
    eltype converge=0;
    eltype lambda=0;
    for (k=0; k<max_iters; k++) {
      vectors(":",i)=uhat;
      // Othogonalize against previous eigenvectors
      for (ya_sizet j=0; j<i; j++)
        vectors(":",i)-=vectors(":",j)*dot(vectors(":",i),vectors(":",j));
      uhat=m*vectors(":",i);
      lambda=norm(uhat);
      uhat/=lambda;

      converge=dot(uhat,vectors(":",i));
      if (converge>pe_eps)
        break;
    }
    if (k==max_iters) {
      std::ostringstream owarn;
      owarn << "Eigen vector did not converge to within " << tol << " with " 
            << max_iters << " iterations of the power method. Rather, "
            << "convergence to: " << 1.0-converge;
      ya_addwarn(305,9,"powereigens",owarn.str());
      err_flag=305;
    }
    vectors(":",i)=uhat;
    values(i)=lambda;
  }
  return err_flag;
}

// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, ya_type3 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    eigrrr(m,dim,vectors,values,0.0);
    ip_fliplr(vectors);
    ip_flipcm(values);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectorst;
    YA_VecT valuest;
    ya_simple_eighl(work,vectorst,valuest);
    vectors=vectorst(":",vmcount(dim));
    values=valuest(vmcount(dim));
  }    
}    

// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2>
void eigs(const YA_BaseT &m, ya_sizet dim, ya_type2 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    eigrrr(m,dim,values,0.0);
    ip_flipcm(values);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectorst;
    YA_VecT valuest;
    ya_simple_eighl(work,vectorst,valuest);
    values=valuest(vmcount(dim));
  }    
}

// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs_small(const YA_BaseT &m, ya_sizet dim, ya_type2 &vectors, 
                ya_type3 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    YA_Cols<long,1> isuppz;
    long info,mout;
    long iu=dim;
    long il=1;
    eltype vl,vu;
    YA_MatT input(m);
    syevr('V','I','L',input,vl,vu,il,iu,0.0,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectorst;
    YA_VecT valuest;
    ya_simple_eig(work,vectorst,valuest);
    vectors=vectorst(":",vmcount(dim));
    values=valuest(vmcount(dim));
  }    
}

// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2>
void eigs_small(const YA_BaseT &m, ya_sizet dim, ya_type2 &values) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS) {
    YA_MatT vectors;
    YA_Cols<long,1> isuppz;
    long info,mout;
    long iu=dim;
    long il=1;
    eltype vl,vu;
    YA_MatT input(m);
    syevr('V','I','L',input,vl,vu,il,iu,0.0,mout,values,vectors,isuppz,info);
  } else 
  #endif
  {
    VM_SymMat work(m);
    YA_MatT vectorst;
    YA_VecT valuest;
    ya_simple_eig(work,vectorst,valuest);
    values=valuest(vmcount(dim));
  }    
}

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
          EigenOptions &opts) {
  _eigs(m,vectors,values,opts,true,false);
}

/// Eigen decomposition for largest 'dim' eigenvalues (in descending order)
template<class eltype, class mat_type, class ya_type2>
void eigs(const YA_BaseT &m, ya_type2 &values, EigenOptions &opts) {
  YA_MatT vectors;
  _eigs(m,vectors,values,opts,false,false);
}

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void eigs_small(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
                EigenOptions &opts) {
  _eigs(m,vectors,values,opts,true,true);
}

/// Eigen decomposition for smallest 'dim' eigenvalues (in ascending order)
template<class eltype, class mat_type, class ya_type2>
void eigs_small(const YA_BaseT &m, ya_type2 &values, EigenOptions &opts) {
  YA_MatT vectors;
  _eigs(m,vectors,values,opts,false,true);
}

// Eigen decomposition
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void _eigs(const YA_BaseT &m, ya_type2 &vectors, ya_type3 &values, 
           EigenOptions &opts, const bool gvectors, const bool small) {
  #ifdef YA_LAPACK
  if (YATRAIT2::VME_BLAS && YATRAIT3::VME_BLAS) {
    if (opts.method()==VMEIGRRR) {
      YA_Cols<long,1> isuppz;
      long info,mout,iu,il;
      if (small) {
        iu=opts.dim();
        il=1;
      } else {
        iu=m.rows();
        il=m.rows()-opts.dim()+1;
      }
      eltype vl,vu;
      YA_MatT input(m);
      if (gvectors)
        syevr('V','I','L',input,vl,vu,il,iu,opts.tol(),mout,values,vectors,
              isuppz,info);
      else
        syevr('N','I','L',input,vl,vu,il,iu,opts.tol(),mout,values,vectors,
              isuppz,info);
      if (!small) {
        if (gvectors)
          ip_fliplr(vectors);
        ip_flipcm(values);
      }
      return;
    } else if (opts.method()==VMEIGPOWER) {
      if (small) {
        YA_Cols<long,1> ifail(m.rows());
        long info, mout;
        long il=1;
        long iu=opts.dim();
        eltype vl,vu;
        YA_MatT input(m);
        if (gvectors)
          syevx('V','I','L',input,vl,vu,il,iu,opts.tol(),mout,values,vectors,
                ifail,info);
        else
          syevx('N','I','L',input,vl,vu,il,iu,opts.tol(),mout,values,vectors,
                ifail,info);
      } else {
        if (gvectors)
          powereigens(m,opts.dim(),vectors,values,opts.tol(),opts.max_iters());
        else
          powereigens(m,opts.dim(),values,opts.tol(),opts.max_iters());
      }
      return;
    } else {
      if (gvectors) {
        YA_MatT vectorst;
        YA_VecT valuest;
        if (opts.method()==VMEIGDQ)
          eig(m,vectorst,valuest);
        else
          eigev(m,vectorst,valuest);
        if (small) {
          vectors=vectorst(":",vmcount(opts.dim()));
          values=valuest(vmcount(opts.dim()));
        } else {
          vectors.setup(m.rows(),opts.dim());
          values.setup(opts.dim());
          ya_sizet oD=opts.dim();
          ya_sizet mR=m.rows();
          for (ya_sizet i=1; i<=oD; i++) {
            vectors(":",i-1)=vectorst(":",mR-i);
            values(i-1)=valuest(mR-i);
          }
        }
      } else {
        YA_VecT valuest;
        if (opts.method()==VMEIGDQ)
          eig(m,valuest);
        else
          eigev(m,valuest);
        if (small)
          values=valuest(vmcount(opts.dim()));
        else {
          ya_sizet oD=opts.dim();
          ya_sizet mR=m.rows();
          values.setup(oD); 
          for (ya_sizet i=1; i<=oD; i++)
            values(i-1)=valuest(mR-i);
        }
      }
    }
    return;
  }
  #endif
    
  if (opts.method()==VMEIGPOWER && !small) {
    if (gvectors)
      powereigens(m,opts.dim(),vectors,values,opts.tol(),opts.max_iters());
    else
      powereigens(m,opts.dim(),values,opts.tol(),opts.max_iters());
    return;
  }
  
  if (gvectors) {
    if (small)
      eigs_small(m,opts.dim(),vectors,values);
    else {
      VM_SymMat work(m);
      YA_MatT vectorst;
      YA_VecT valuest;
      ya_simple_eighl(work,vectorst,valuest);
      vectors=vectorst(":",vmcount(opts.dim()));
      values=valuest(vmcount(opts.dim()));
    }
  } else {
    if (small)
      eigs_small(m,opts.dim(),values);
    else {
      VM_SymMat work(m);
      YA_VecT valuest;
      ya_simple_eighl(work,vectors,valuest);
      values=valuest(vmcount(opts.dim()));
    }
  }  
  return;
}

// Cholesky decomposition (A=L * LT)
template<class eltype, class mat_type,
         class ya_type2>
int chol(const YA_BaseT &a, ya_type2 &l, bool write_err) {
  l=vmzeros(a.rows(),a.rows());
  ya_sizet aR=a.rows();
  for (ya_sizet i=0; i<aR; i++)
    for (ya_sizet j=i; j<aR; j++) {
      eltype sum=a(i,j);
      for (ya_sizet k=i-1; k>=0; k--)
        sum-=l(i,k)*l(j,k);
      if (i==j) {
        if (sum<=0.0) {
          if (write_err)
            ya_addwarn(304,9,"VM_Decompose",
               "Unable to perform Cholesky decomposition on a matrix that is not positive definite.");
          return 304;
        }
        l(i,j)=sqrt(sum);
      } else
        l(j,i)=sum/l(i,i);
    }
  return 0;
}

// Cholesky solve A*x=b
template<class eltype, class mat_type,
         class ya_type2, class ya_type3>
int chol_solve(const YA_BaseT &a, ya_type2 &x, const ya_type3 &b, bool write_err) {
  YA_DEBUG_ERROR2(b.cols()==1 && b.rows()==a.rows(),
                  "Matrix dimensions do not match for chol_solve",a,b);

  YA_MatT l;
  int err_flag=chol(a,l,write_err);
  x.setup(a.rows());
  ya_sizet aR=a.rows();
  for (ya_sizet i=0; i<aR; i++) {
    eltype sum=b(i);
    for (ya_sizet k=i-1; k>=0; k--)
      sum-=l(i,k)*x(k);
    x(i)=sum/l(i,i);
  }
  for (ya_sizet i=aR-1; i>=0; i--) {
    eltype sum=x(i);
    for (ya_sizet k=i+1; k<aR; k++)
      sum-=l(k,i)*x(k);
    x(i)=sum/l(i,i);
  }
  return err_flag;
}

/// LU decomposition
template<class eltype, class mat_type, class ya_type2>
void lu(YA_BaseT &input, ya_type2 &p, int &signum) {
  YA_DEBUG_ERROR1(input.rows()==input.cols(),
                  "Matrix input for LU not square",input);
  
  signum=1;
  ya_sizet iR=input.rows();
  p=transpose(vmcount(iR));
  for (ya_sizet j=0; j<iR-1; j++) {
    eltype ajj, max = fabs(input(j,j));
  ya_sizet i_pivot=j;
  
  for (ya_sizet i=j+1; i<iR; i++) {
    eltype aij=fabs(input(i,j));
    if (aij>max) {
    max=aij;
    i_pivot=i;
      }
    }
  
  if (i_pivot!=j) {
    ya_swap(input(j,":"),input(i_pivot,":"));
    ya_sizet temp=p(j); p(j)=p(i_pivot); p(i_pivot)=temp;
    signum=-signum;
  }
  
  ajj=input(j,j);
  if (ajj!=0.0)
    for (ya_sizet i=j+1; i<iR; i++) {
        eltype aij=input(i,j)/ajj;
        input(i,j)=aij;
        for (ya_sizet k=j+1; k<iR; k++)
      input(i,k)=input(i,k)-aij*input(j,k);
    }
  }
}

/// Solve Ax=b in-place using LU decomposition
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void lu_solve(const YA_BaseT &LU, const ya_type2 &p, ya_type3 &x) {
  YA_DEBUG_ERROR(LU.rows()==LU.cols() && LU.rows()==p.numel() && 
                 LU.rows()==x.numel(),
                 "Matrix dimensions do not match for lu_solve");
  
  x=YA_VecT(x(p));

  /* Solve for c using forward-substitution, L c = P b */
  cblas_trsv(CblasLower,CblasUnit,LU,x);
  /* Perform back-substitution, U x = c */
  cblas_trsv(CblasUpper,CblasNonUnit,LU,x);
}

/// Invert the matrix L*U
template<class eltype, class mat_type, class ya_type2, class ya_type3>
void lu_invert(const YA_BaseT &LU, const ya_type2 &p, ya_type3 &inverse) {
  inverse=vmeye(LU.rows());
  ya_sizet luR=LU.rows();
  for (ya_sizet i = 0; i < luR; i++)
  lu_solve(LU, p, inverse(":",i));
}

}
#endif
