/***************************************************************************
                                ya_int_eig.cpp
                             -------------------
                               W. Michael Brown

  Internal functions for eigen decomposition
 
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

#ifndef YA_INT_EIG_S_H
#define YA_INT_EIG_S_H

YALANAMESPACE

template<class eltype, class ya_type1, class ya_type2>
inline void ya_simple_eig(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values) {
  ya_simple_eig_s<eltype,ya_type1,
                  ya_type2,VMExprOpLT<eltype> >(mat,vectors,values);
}
  
template<class eltype, class ya_type1, class ya_type2>
inline void ya_simple_eighl(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values) {
  ya_simple_eig_s<eltype,ya_type1,
                  ya_type2,VMExprOpGT<eltype> >(mat,vectors,values);
}
  
template<class eltype, class ya_type1, class ya_type2, class sortop>
inline void ya_simple_eig_s(VM_SymMat &mat,ya_type1 &vectors,ya_type2 &values) {
  const ya_sizet nR=mat.rows();

  vectors.setup(nR,nR);
  values.setup(nR);

  eltype *eivectors=vectors.c_ptr();
  eltype *eivalues=values.c_ptr();
  eltype *array=mat.c_ptr();

  // Setup variables
  ya_sizet i,j,l,m,lm,im;
  ya_sizet il, ilq, ilr, imq, imr, ind, iq, k, km, ll, lq, mm, mq;
  double am, anorm, anrmx, cosx, cosx2, sincs, sinx, sinx2, thr, x, y;
  double theshold = DBL_EPSILON;

  // The matrix that will hold the results is initially = I.
  iq=0;
  for (i=0; i<nR; i++)
    for (j=0; j<nR; j++) {
      if (i==j)
        eivectors[iq]=1;
      else
        eivectors[iq]=0;
      iq++;
    }

  // Initial and final norms (anorm & anrmx).
  anorm=0.0;
  iq=0;
  for (i=0; i<nR; i++)
    for (j=0; j<=i; j++) {
      if (j!=i) anorm+=array[iq]*array[iq];
      iq++;
    }

  if (anorm>0.0) {
    anorm=sqrt(2.0*anorm);
    anrmx=theshold*anorm/nR;

    // Compute threshold and initialise flag.
    thr=anorm;
    while (thr>anrmx) { // Compare threshold with final norm
      thr/=nR;
      ind=1;
       while (ind) {
        ind=0;
        l=0;
        while (l != nR-1) { // Test for l beyond penultimate column
          lq=l*(l+1)/2;
          ll=l+lq;
          m=l+1;
          ilq=nR*l;
          while (m != nR) { // Test for m beyond last column
            // Compute sin & cos.
            mq=m*(m+1)/2;
            lm=l+mq;
            if (fabs(static_cast<double>(array[lm]))>=thr) {
              ind=1;
              mm=m+mq;
              x=0.5*(array[ll]-array[mm]);
               y=-array[lm]/sqrt(array[lm]*array[lm]+x*x);
              if (x<0.0)
                y=-y;
              sinx=y/sqrt(2.0*(1.0+(sqrt(1.0-y*y))));
              sinx2=sinx*sinx;
              cosx=sqrt(1.0-sinx2);
              cosx2=cosx*cosx;
              sincs=sinx*cosx;

              // Rotate l & m columns.

              imq=nR*m;
              for (i=0; i<nR; i++) {
                iq=i*(i+1)/2;
                if (i!=l && i!=m) {
                  if (i<m) im=i+mq;
                    else im=m+iq;
                  if (i<l) il=i+lq;
                    else il=l+iq;
                  x=array[il]*cosx-array[im]*sinx;
                  array[im]=array[il]*sinx+array[im]*cosx;
                  array[il]=x;
                }
                ilr=ilq+i;
                imr=imq+i;
                x = (*(eivectors+ilr)*cosx)-(*(eivectors+imr)*sinx);
                *(eivectors+imr) = (*(eivectors+ilr)*sinx) +
                                   (*(eivectors+imr)*cosx);
                *(eivectors+ilr) = x;
              }
              x=2.0*array[lm]*sincs;
              y=array[ll]*cosx2+array[mm]*sinx2-x;
              x=array[ll]*sinx2+array[mm]*cosx2+x;
              array[lm]=(array[ll]-array[mm])*sincs+
                          array[lm]*(cosx2-sinx2);
              array[ll]=y;
              array[mm]=x;
            }
            m++;
          }
          l++;
        }
      }
    }
  }

  // Sort eigenvalues & eigenvectors in order of descending eigenvalue.
  k=0;
  for (i=0; i<nR-1; i++) {
    im=i;
    km=k;
    am=array[k];
    l=0;
    for (j=0; j<nR; j++) {
      if (j>i && sortop::op(array[l],am)) {
        im=j;
        km=l;
        am=array[l];
      }
      l+=j+2;
    }
    if (im!=i) {
      array[km]=array[k];
      array[k]=am;
      l=nR*i;
      m=nR*im;
      for (j=0; j<nR; j++) {
        am=*(eivectors+l);
        *(eivectors+(l++)) = *(eivectors+m);
        *(eivectors+(m++)) = am;
      }
    }
    k+=i+2;
  }

  // place sorted eigenvalues into the matrix_vector structure
  for (j=0, k=0; j<nR; j++) {
    eivalues[j]=array[k];
    k+=j+2;
  }
}


}
#endif
