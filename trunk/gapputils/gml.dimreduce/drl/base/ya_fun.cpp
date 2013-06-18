/***************************************************************************
                                  ya_fun.cpp
                             -------------------
                               W. Michael Brown

  Misc functions
 
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

#ifndef YA_FUN_S_H
#define YA_FUN_S_H

YALANAMESPACE

// ---- Loop helpers
template <class ref, class eltype> struct _ya_ufor_total {
  static inline void op(ref r, eltype &v) { v+=r; } 
};
template <class ref, class eltype> struct _ya_ufor_product {
  static inline void op(ref r, eltype &v) { v*=r; } 
};
template <class ref, class eltype> struct _ya_ufor_maximum {
  static inline void op(ref r, eltype &v) { if (r>v) v=r; } 
};
template <class ref, class eltype> struct _ya_ufor_minimum {
  static inline void op(ref r, eltype &v) { if (r<v) v=r; } 
};
template <class ref> struct _ya_ufor_find {
  static inline bool op(ref r) { return r!=0; } 
};
template <class eltype, class stype> struct _ya_vstat_struct 
  { eltype vmin,vmax; stype vmean,vstd; };
template <class ref, class eltype, class stype> struct _ya_ufor_vstat {
  static inline void op(ref r, _ya_vstat_struct<eltype,stype> &v) {
    if (r<v.vmin) v.vmin=r;
    if (r>v.vmax) v.vmax=r;
    v.vmean+=r;
  }
};
template <class ref, class eltype, class stype> struct _ya_ufor_vstat2 {
  static inline void op(ref r, _ya_vstat_struct<eltype,stype> &v)
    { stype t=r-v.vmean; v.vstd+=t*t; }
};

template<class eltype, class mat_type>
inline typename YA_COPY_TYPE copy(const YA_BaseT &mat) {
  return typename YA_COPY_TYPE(mat);
}

template<class eltype, class mat_type>
inline void zero(YA_BaseT &mat) {
  ya_zero_matrix(mat);
}

template<class eltype, class mat_type>
inline void zero(const YA_WrapperT &mat) {
  ya_zero_matrix(mat);
}

template<class eltype, class mat_type>
inline void setall(YA_BaseT &mat, const eltype value) { 
  ya_copy_scalar(mat,value);
}

template<class eltype, class mat_type>
inline void setall(const YA_WrapperT &mat, const eltype value) { 
  ya_copy_scalar(mat,value);
}

template<class eltype, class mat_type, class ya_type2>
inline bool identical(const YA_BaseT &one, const ya_type2 &two) {
  return ya_compare_matrix(one,two); 
}

template<class eltype, class mat_type, class ya_type2>
inline void ya_swap(YA_BaseT &one, ya_type2 &two) {
  typename YA_COPY_TYPE temp(one);
  one=two;
  two=temp;
}

template<class eltype, class mat_type, class ya_type2>
inline void ya_swap(const YA_WrapperT &one, ya_type2 &two) {
  typename YA_COPY_TYPE temp(one);
  one=two;
  two=temp;
}

template<class eltype, class mat_type>
inline void ip_flipcm(YA_BaseT &mat) {
  ya_sizet diff=length(mat)-1;
  ya_sizet i=0;
  while (diff>i) {
    eltype temp=mat(i); mat(i)=mat(diff); mat(diff)=temp;
    i++;
    diff--;
  }
}

template<class eltype, class mat_type>
inline void ip_flipcm(const YA_WrapperT &mat) {
  ya_sizet diff=length(mat)-1;
  ya_sizet i=0;
  while (diff>i) {
    eltype temp=mat(i); mat(i)=mat(diff); mat(diff)=temp;
    i++;
    diff--;
  }
}

template<class eltype, class mat_type>
inline void ip_fliplr(YA_BaseT &mat) {
  ya_sizet diff=mat.cols()-1;
  ya_sizet i=0;
  while (diff>i) {
    ya_swap(mat(":",i),mat(":",diff));
    i++;
    diff--;
  }
}
  
template<class eltype, class mat_type>
inline void ip_fliplr(const YA_WrapperT &mat) {
  ya_sizet diff=mat.cols()-1;
  ya_sizet i=0;
  while (diff>i) {
    ya_swap(mat(":",i),mat(":",diff));
    i++;
    diff--;
  }
}

template<class eltype, class mat_type>
inline void ip_transpose(YA_BaseT &mat) {
  if (mat.cols()==mat.rows()) {
    ya_sizet R=mat.rows();
    for (ya_sizet i=0;i<R;i++) 
      for (ya_sizet j=i;j<R;j++) {        
        eltype temp = mat(j-i,j); 
        mat(j-i,j) = mat(j,j-i); 
        mat(j,j-i) = temp; 
      }
  } else
    mat=copy(mat.T());
}  

template<class eltype, class mat_type>
inline void ip_transpose(const YA_WrapperT &mat) {
  if (mat.cols()==mat.rows()) {
    ya_sizet R=mat.rows();
    for (int i=0;i<R;i++) 
      for (int j=i;j<R;j++) {        
        eltype temp = mat(j-i,j); 
        mat(j-i,j) = mat(j,j-i); 
        mat(j,j-i) = temp; 
      }
  } else
    mat=copy(mat.T());
}  

// ----------

template<class eltype, class mat_type>
eltype l2_norm(const YA_BaseT &mat) {
  return sqrt(total(square(mat)));
}

template<class eltype, class mat_type>
eltype total(const YA_BaseT &mat) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute total() on empty matrix");
  eltype sumi=0;
  ya_no_order_loop<_ya_ufor_total<typename mat_type::cref,
                                  eltype> >::lp(mat,sumi);
  return sumi;
}

template<class eltype, class mat_type>
eltype product(const YA_BaseT &mat) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute product() on empty matrix");
  eltype prod=1;
  ya_no_order_loop<_ya_ufor_product<typename mat_type::cref,
                                    eltype> >::lp(mat,prod);
  return prod;
}

template<class eltype, class mat_type>
eltype minimum(const YA_BaseT &mat) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute minimum() on empty matrix");
  eltype min=mat(0);
  ya_no_order_loop<_ya_ufor_minimum<typename mat_type::cref,
                                    eltype> >::lp(mat,min);
  return min;
}

template<class eltype, class mat_type>
eltype maximum(const YA_BaseT &mat) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute maximum() on empty matrix");
  eltype max=mat(0);
  ya_no_order_loop<_ya_ufor_maximum<typename mat_type::cref,
                                    eltype> >::lp(mat,max);
  return max;
}

/// Return the minimum, maximum, and mean
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute vstat() on empty matrix");
  _ya_vstat_struct<eltype,stype> ss;
  ss.vmin=mat(0);
  ss.vmax=mat(0);
  ss.vmean=0.0;
  ya_no_order_loop<_ya_ufor_vstat<typename mat_type::cref,eltype,
                                  stype> >::lp(mat,ss);
  vmin=ss.vmin;
  vmax=ss.vmax;
  vmean=ss.vmean/mat.numel();
}

/// Return the minimum, maximum, mean, and standard deviation
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, eltype &vmin, eltype &vmax, stype &vmean,
           stype &vstd) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute vstat() on empty matrix");
  _ya_vstat_struct<eltype,stype> ss;
  ss.vmin=mat(0);
  ss.vmax=mat(0);
  ss.vmean=0.0;
  ya_no_order_loop<_ya_ufor_vstat<typename mat_type::cref,eltype,
                                  stype> >::lp(mat,ss);
  ss.vmean/=mat.numel();
  ss.vstd=0.0;
  ya_no_order_loop<_ya_ufor_vstat2<typename mat_type::cref,eltype,
                                   stype> >::lp(mat,ss);
  vmin=ss.vmin;
  vmax=ss.vmax;
  vmean=ss.vmean;
  vstd=sqrt(ss.vstd/mat.numel());
}

/// Return the minimum, maximum, mean of each column
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, YA_RowT &vmin, YA_RowT &vmax, 
           YA_Rows<stype,1> &vmean) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute vstat() on empty matrix");

  _ya_vstat_struct<eltype,stype> ss;
  ya_sizet C=mat.cols();
  ya_sizet R=mat.rows();
  vmin.setup(C);
  vmax.setup(C);
  vmean.setup(C);

  for (ya_sizet i=0; i<C; i++) {
    ss.vmin=mat(0,i);
    ss.vmax=mat(0,i);
    ss.vmean=0.0;
    ya_no_order_loop<_ya_ufor_vstat<typename mat_type::cref,eltype,
                                     stype> >::lp(mat(":",i),ss);
    vmin(i)=ss.vmin;
    vmax(i)=ss.vmax;
    vmean(i)=ss.vmean/R;
  }
}

/// Return the minimum, maximum, mean, and standard deviation of each column
template<class eltype, class mat_type, class stype>
void vstat(const YA_BaseT &mat, YA_RowT &vmin, YA_RowT &vmax, 
           YA_Rows<stype,1> &vmean, YA_Rows<stype,1> &vstd) {
  YA_DEBUG_ERROR(mat.numel()>0,"Cannot compute vstat() on empty matrix");

  _ya_vstat_struct<eltype,stype> ss;
  ya_sizet C=mat.cols();
  ya_sizet R=mat.rows();
  vmin.setup(C);
  vmax.setup(C);
  vmean.setup(C);
  vstd.setup(C);

  for (ya_sizet i=0; i<C; i++) {
    ss.vmin=mat(0,i);
    ss.vmax=mat(0,i);
    ss.vmean=0.0;
    ya_no_order_loop<_ya_ufor_vstat<typename mat_type::cref,eltype,
                                     stype> >::lp(mat(":",i),ss);
    ss.vmean/=R;
    ss.vstd=0.0;
    ya_no_order_loop<_ya_ufor_vstat2<typename mat_type::cref,eltype,
                                      stype> >::lp(mat(":",i),ss);
    vmin(i)=ss.vmin;
    vmax(i)=ss.vmax;
    vmean(i)=ss.vmean;
    vstd(i)=sqrt(ss.vstd/R);
  }
}

template <class eltype, class stype> struct _ya_stat_struct 
  { eltype vmin,vmax; stype vmean; };
template <class ref, class eltype, class stype> struct _ya_ufor_stat {
  static inline void op(ref r, _ya_stat_struct<eltype,stype> &v) {
    if (r<v.vmin) r=v.vmin;
    if (r>v.vmax) r=v.vmax;
    v.vmean+=r;
  }
};


template<class eltype, class mat_type, class ya_type2>
eltype dist2(const YA_BaseT &one, const ya_type2 &two) {
  return total(square(one-two));
}

/// Return a vector with indices to non-zero elements in the matrix
template<class eltype, class mat_type>
YA_DynI find(const YA_BaseT &mat) {
  YA_DynI findv;
  ya_no_order_loop<_ya_ufor_find<typename mat_type::cref> >::find(mat,
                                                                  findv);
  return findv;
}

}
#endif
