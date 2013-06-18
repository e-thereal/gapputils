/***************************************************************************
                                  vm_fun.cpp
                             -------------------
                W. Michael Brown, Haixia Jia, Jean-Paul Watson

  Vector/Matrix Functions that do not return expressions
  
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Feb 12 2007
    authors    : W. Michael Brown, Haixia Jia, Jean-Paul Watson
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VM_FUN_CPP
#define VM_FUN_CPP

#ifdef CUDA
#include "cuda_knn.h"
#endif

YALANAMESPACE

// ---- copy the lower triangle of the matrix b into the lower triangle of a
template<class eltype, class mat_type, class mat_type2>
inline void copy_lower_triangle(YA_BaseT &a, const YA_BaseT2 &b) {
  YA_DEBUG_ERROR(a.rows()==b.rows() && a.rows()==a.cols() && 
                 b.rows()==b.cols(),
                 "Matrix dimensions don't allow lower triangle copy.");
  if (YA_TRAIT::VME_SYMMETRIC) {
    if (YA_TRAIT2::VME_SYMMETRIC) {
      a=b;
      return;
    }
    ya_sizet symi=0, bR=b.rows();
    for (ya_sizet i=0; i<bR; i++)
      for (ya_sizet j=0; j<=i; j++) {
        a(symi)=b(i,j);
        symi++;
      }
    return;
  } else if (YA_TRAIT2::VME_SYMMETRIC) {
    ya_sizet symi=0, bR=b.rows();
    for (ya_sizet i=0; i<bR; i++)
      for (ya_sizet j=0; j<=i; j++) {
        a(i,j)=b(symi);
        symi++;
      }
    return;
  }
  ya_sizet bR=b.rows();
  for (ya_sizet i=0; i<bR; i++)
    for (ya_sizet j=0; j<=i; j++)
      a(i,j)=b(i,j);
}
  
// ---- copy the lower triangle of the matrix b into the lower triangle of a
template<class eltype, class mat_type, class mat_type2>
inline void copy_lower_triangle(const YA_WrapperT &a, const YA_BaseT2 &b) {
  YA_DEBUG_ERROR(a.rows()==b.rows() && a.rows()==a.cols() && 
                 b.rows()==b.cols(),
                 "Matrix dimensions don't allow lower triangle copy.");
  if (YA_TRAIT::VME_SYMMETRIC) {
    if (YA_TRAIT2::VME_SYMMETRIC) {
      a=b;
      return;
    }
    ya_sizet symi=0, bR=b.rows();
    for (ya_sizet i=0; i<bR; i++)
      for (ya_sizet j=0; j<=i; j++) {
        a(symi)=b(i,j);
        symi++;
      }
    return;
  } else if (YA_TRAIT2::VME_SYMMETRIC) {
    ya_sizet symi=0, bR=b.rows();
    for (ya_sizet i=0; i<bR; i++)
      for (ya_sizet j=0; j<=i; j++) {
        a(i,j)=b(symi);
        symi++;
      }
    return;
  }
  ya_sizet bR=b.rows();
  for (ya_sizet i=0; i<bR; i++)
    for (ya_sizet j=0; j<=i; j++)
      a(i,j)=b(i,j);
}
  
// ---- length
template<class eltype, class mat_type>
inline ya_sizet length(const YA_BaseT &m) {
  return m.numel();
}
// ---- size
template<class eltype, class mat_type>
inline ya_sizet size(const YA_BaseT &m, const ya_sizet i) {
  YA_DEBUG_ERROR(i==0 || i==1,"Second parameter to size must be 0 or 1");
  if (i==0)
    return m.rows();
  return m.cols();
}

// ---- cumsum
template<class eltype, class mat_type>
inline YA_MatT cumsum(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()!=0,"Cannot do cumprod on a empty matrix.");
  YA_MatT ans(m.rows(),m.cols());
  if (m.rows()==1) {
    ans(0)=m(0);
    ya_sizet mC=m.cols();
    for (ya_sizet i=1; i<mC; i++)
      ans(i)=ans(i-1)+m(i);
    return ans;
  }
  ya_sizet mC=m.cols(), mR=m.rows();  
  for (ya_sizet i=0; i<mC; i++) {
    ans(0,i)=m(0,i);
    for (ya_sizet j=1; j<mR; j++)
      ans(j,i)=ans(j-1,i)+m(j,i);
  }
  return ans;
}

// ---- cumprod
template<class eltype, class mat_type>
inline YA_MatT cumprod(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.numel()!=0,"Cannot do cumprod on a empty matrix.");
  YA_MatT ans(m.rows(),m.cols());
  ya_sizet mC=m.cols();
  if (m.rows()==1) {
    ans(0)=m(0);
    for (ya_sizet i=1; i<mC; i++)
      ans(i)=ans(i-1)*m(i);
    return ans;
  }
    
  ya_sizet mR=m.rows();
  for (ya_sizet i=0; i<mC; i++) {
    ans(0,i)=m(0,i);
    for (ya_sizet j=1; j<mR; j++)
      ans(j,i)=ans(j-1,i)*m(j,i);
  }
  return ans;
}

// ---- norm
template<int blas_type>
struct _vm_norm_helper {
  template<class eltype, class mat_type>
  inline static eltype op(const YA_BaseT &m) {
    double nor=0;
    ya_sizet mN=m.numel();
    for (ya_sizet i=0; i<mN; i++)
      nor+=m(i)*m(i);
    return sqrt(nor);
  }
};

#ifdef YA_BLAS
template<>
struct _vm_norm_helper<1> {
  template<class eltype, class mat_type>
  inline static eltype op(const YA_BaseT &m) {
    return cblas_nrm2(m);
  }
};
#endif
    
template<class eltype, class mat_type>
eltype norm(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.rows()==1 || m.cols()==1,
                 "No matrix norm implemented yet");
  return _vm_norm_helper<YA_TRAIT::VME_USE_BLAS>::op(m);
}
// ---- trace
template<class eltype, class mat_type>
eltype trace(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.rows()==m.cols(),"Matrix for trace must be square");

  eltype sum=0;
  ya_sizet mR=m.rows();
  for (ya_sizet i=0; i<mR; i++)
    sum+=m(i,i);
  return sum;
}

// Compute dot product without BLAS
template<int,int>
struct _vm_dot_product {
  template<class eltype, class mat_type, class ya_type2>
  static inline eltype op(const YA_BaseT &one, const ya_type2 &two) {
    eltype ans=0;
    ya_sizet oN=one.numel();
    for (ya_sizet i=0; i<oN; i++)
      ans+=one(i)*two(i);
    return ans;
  }
};
  
#ifdef YA_BLAS
// Compute dot product without BLAS
template<>
struct _vm_dot_product<1,1> {
  template<class eltype, class mat_type, class ya_type2>
  static inline eltype op(const YA_BaseT &one, const ya_type2 &two) {
    return cblas_dot(one,two);
  }
};
#endif  

// ---- dot product
// ---- dot product
template<class eltype, class mat_type, class mat_type2>
eltype dot(const YA_BaseT &one, const YA_BaseT2 &two) {
  YA_DEBUG_ERROR(one.numel()==two.numel(),
                 "Vector lengths do not match for dot product.");
  return _vm_dot_product<YA_TRAIT::VME_USE_BLAS,
                         YA_TRAIT2::VME_USE_BLAS>::op(one,two);
}

// ---- inverse
template<class eltype, class mat_type, class ya_type2>
inline void inv(const YA_BaseT &m, ya_type2 &ans) {
  YA_MatT LU(m);
  YA_ColI p;
  int signum;
  lu(LU,p,signum);
  lu_invert(LU,p,ans);
}

// ---- inverse
template<class eltype, class mat_type>
inline YA_MatT inv(const YA_BaseT &m) {
  YA_MatT ans;
  inv(m,ans);
  return ans;
}

// ---- Unique elements
template <class eltype, class mat_type>
YA_DynT unique(const YA_BaseT &one) {
  YA_DynT ui;
  YA_MatI sorti;
  sort_indices_cm(one,sorti);
  ya_sizet i=0;
  while (i<sorti.numel()) {
    ui.push_back(one(sorti(i)));
    i++;
    while (i<sorti.numel()) {
      if (one(sorti(i))!=one(sorti(i-1)))
        break;
      i++;
    }
  }
  return ui;
}

// ---- Unique elements
template <class eltype, class mat_type>
VecMatReI<eltype,const mat_type,YA_DynI> unique(const YA_BaseT &one,
                                               YA_DynI &ui) {
  YA_MatI indices;
  sort_indices_cm(one,indices);
  
  ya_sizet i=0;
  while (i<one.numel()) {
    ui.push_back(indices(i));
    i++;
    while (i<one.numel()) {
      if (one(indices(i))!=one(indices(i-1)))
        break;
      i++;
    }
  }
  return VecMatReI<eltype,const mat_type,YA_DynI>
           (&one.stv(),&ui);
}

// ---- Unique elements
template <class eltype, class mat_type, class ya_type2>
VecMatReI<eltype,const mat_type,YA_DynI> unique(const YA_BaseT &one,
                                               YA_DynI &ui,ya_type2 &uj) {
  YA_MatI indices;
  sort_indices_cm(one,indices);
  uj.setup(one.numel());
  
  ya_sizet i=0;
  while (i<one.numel()) {
    ui.push_back(indices(i));
    uj(indices(i))=ui.numel()-1;
    i++;
    while (i<one.numel()) {
      if (one(indices(i))!=one(indices(i-1)))
        break;
      uj(indices(i))=ui.numel()-1;
      i++;
    }
  }
  return VecMatReI<eltype,const mat_type,YA_DynI>
           (&one.stv(),&ui);
}

// ---- Set difference (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT setdiff(const YA_BaseT &one, const YA_BaseT2T &two) {
  if (two.numel()==0)
    return unique(one);
    
  YA_DynT ui;
  if (one.numel()==0)
    return ui;
  
  YA_MatI indices, indices2;
  sort_indices_cm(one,indices);
  sort_indices_cm(two,indices2);
  ya_sizet i=0;
  ya_sizet j=0;
  while (1) {
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(one(indices(i)));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices(i))!=one(indices(i-1)))
          ui.push_back(one(indices(i)));
        i++;
      }
      return ui;
    }
    while (i<one.numel()) {
      if (one(indices(i))<two(indices2(j))) {
        ui.push_back(one(indices(i)));
        // no duplicates
        i++;
        while (i<one.numel()) {
          if (one(indices(i))!=one(indices(i-1)))
            break;
          i++;
        }
      } else
        break;
    }
    if (i==one.numel())
      return ui;
    while (j<two.numel()) {
      if (one(indices(i))<two(indices2(j)))
        break;
      else if (one(indices(i))==two(indices2(j))) {
        // no duplicates
        i++;
        while (i<one.numel()) {
          if (one(indices(i))!=one(indices(i-1)))
            break;
          i++;
        }
        if (i==one.numel())
          return ui;
        j++;
        while (j<two.numel()) {
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          j++;
        }
      } else
        j++;
    }
  }
}

// ---- Set difference (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
VecMatReI<eltype,const mat_type,YA_DynI> setdiff(const YA_BaseT &one,
                                                      const YA_BaseT2T &two,
                                                      YA_DynI &ui) {
  if (two.numel()==0)
    return unique(one,ui);

  if (one.numel()==0)
    return VecMatReI<eltype,const mat_type,YA_DynI>
                  (&one.stv(),&ui);
  
  YA_MatI indices, indices2;
  sort_indices_cm(one,indices);
  sort_indices_cm(two,indices2);
  ya_sizet i=0;
  ya_sizet j=0;
  while (1) {
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(indices(i));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices(i))!=one(indices(i-1)))
          ui.push_back(indices(i));
        i++;
      }
      return VecMatReI<eltype,const mat_type,YA_DynI>
                  (&one.stv(),&ui);
    }
    while (i<one.numel()) {
      if (one(indices(i))<two(indices2(j))) {
        ui.push_back(indices(i));
        // no duplicates
        i++;
        while (i<one.numel()) {
          if (one(indices(i))!=one(indices(i-1)))
            break;
          i++;
        }
      } else
        break;
    }
    if (i==one.numel())
      return VecMatReI<eltype,const mat_type,YA_DynI>
                        (&one.stv(),&ui);
    while (j<two.numel()) {
      if (one(indices(i))<two(indices2(j)))
        break;
      else if (one(indices(i))==two(indices2(j))) {
        // no duplicates
        i++;
        while (i<one.numel()) {
          if (one(indices(i))!=one(indices(i-1)))
            break;
          i++;
        }
        if (i==one.numel())
          return VecMatReI<eltype,const mat_type,YA_DynI>
                        (&one.stv(),&ui);;
        j++;
        while (j<two.numel()) {
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          j++;
        }
      } else
        j++;
    }
  }
}

// ---- Set union (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT setunion(const YA_BaseT &one, const YA_BaseT2T &two) {
  if (two.numel()==0)
    return unique(one);
  if (one.numel()==0)
    return unique(two);

  YA_DynT ui;
  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);

  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel()) {
      if (j==0) {
        ui.push_back(two(indices2(j)));
        j++;
      }
      while (j<two.numel()) {
        if (two(indices2(j))!=two(indices2(j-1))) {
          ui.push_back(two(indices2(j)));
        }
        j++;
      }
      return ui;
    }
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        ui.push_back(two(indices2(j)));
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(one(indices1(i)));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices1(i))!=one(indices1(i-1))) {
          ui.push_back(one(indices1(i)));
        }
        i++;
      }
      return ui;
    }
  }
}

// ---- Set union (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT setunion(const YA_BaseT &one, const YA_BaseT2T &two, YA_DynI &ia,
                   YA_DynI &ib) {
  if (two.numel()==0)
    return YA_DynT(unique(one,ia));

  if (one.numel()==0)
    return YA_DynT(unique(two,ib));
  
  YA_DynT ui;
  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);

  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel()) {
      if (j==0) {
        ui.push_back(two(indices2(j)));
        ib.push_back(indices2(j));
        j++;
      }
      while (j<two.numel()) {
        if (two(indices2(j))!=two(indices2(j-1))) {
          ui.push_back(two(indices2(j)));
          ib.push_back(indices2(j));
        }
        j++;
      }
      return ui;
    }
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        ui.push_back(two(indices2(j)));
        ib.push_back(indices2(j));
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices1(i))!=one(indices1(i-1))) {
          ui.push_back(one(indices1(i)));
          ia.push_back(indices1(i));
        }
        i++;
      }
      return ui;
    }
  }
}

// ---- Set intersection (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT intersect(const YA_BaseT &one, const YA_BaseT2T &two) {
  YA_DynT ui;
  if (two.numel()==0)
    return ui;

  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);
  
  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel())
      return ui;
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel())
      return ui;
  }
}

// ---- Set intersection (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT intersect(const YA_BaseT &one, const YA_BaseT2T &two, YA_DynI &ia,
                    YA_DynI &ib) {
  YA_DynT ui;

  if (two.numel()==0)
    return ui;

  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);
  
  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        ib.push_back(indices2(j));
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel())
      return ui;
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel())
      return ui;
  }
}

// ---- Set xor (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
YA_DynT setxor(const YA_BaseT &one, const YA_BaseT2T &two) {
  if (two.numel()==0)
    return unique(one);
  if (one.numel()==0)
    return unique(two);

  YA_DynT ui;
  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);
  
  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel()) {
      if (j==0) {
        ui.push_back(two(indices2(j)));
        j++;
      }
      while (j<two.numel()) {
        if (two(indices2(j))!=two(indices2(j-1))) {
          ui.push_back(two(indices2(j)));
        }
        j++;
      }
      return ui;
    }
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        ui.push_back(two(indices2(j)));
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(one(indices1(i)));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices1(i))!=one(indices1(i-1))) {
          ui.push_back(one(indices1(i)));
        }
        i++;
      }
      return ui;
    }
  }
}

// ---- Set xor (input vectors not necessarily sorted)
template <class eltype, class mat_type, class eltype2, class mat_type2>
inline YA_DynT setxor(const YA_BaseT &one, const YA_BaseT2T &two,
                        YA_DynI &ia, YA_DynI &ib) {
  if (two.numel()==0)
    return YA_DynT(unique(one,ia));
  if (one.numel()==0)
    return YA_DynT(unique(two,ib));
    
  YA_DynT ui;
  YA_MatI indices1;
  YA_MatI indices2;
  sort_indices_cm(one,indices1);
  sort_indices_cm(two,indices2);
  
  ya_sizet i=0, j=0;
  while (1) {
    while (i<one.numel()) {
      if (one(indices1(i))<two(indices2(j))) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        // no duplicates
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
      } else
        break;
    }
    while (i<one.numel() && j<two.numel()) {
      if (one(indices1(i))==two(indices2(j))) {
        i++;
        while (i<one.numel())
          if (one(indices1(i))!=one(indices1(i-1)))
            break;
          else
            i++;
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (i==one.numel()) {
      if (j==0) {
        ui.push_back(two(indices2(j)));
        ib.push_back(indices2(j));
        j++;
      }
      while (j<two.numel()) {
        if (two(indices2(j))!=two(indices2(j-1))) {
          ui.push_back(two(indices2(j)));
          ib.push_back(indices2(j));
        }
        j++;
      }
      return ui;
    }
    while (j<two.numel()) {
      if (one(indices1(i))>two(indices2(j))) {
        ui.push_back(two(indices2(j)));
        ib.push_back(indices2(j));
        // no duplicates
        j++;
        while (j<two.numel())
          if (two(indices2(j))!=two(indices2(j-1)))
            break;
          else
            j++;
      } else
        break;
    }
    if (j==two.numel()) {
      if (i==0) {
        ui.push_back(one(indices1(i)));
        ia.push_back(indices1(i));
        i++;
      }
      while (i<one.numel()) {
        if (one(indices1(i))!=one(indices1(i-1))) {
          ui.push_back(one(indices1(i)));
          ia.push_back(indices1(i));
        }
        i++;
      }
      return ui;
    }
  }
}

// ---- Primes
inline YA_MatI primes(const ya_sizet n) {
  YA_MatI ans;
  if (n<2) {
    ans=0;
    return ans;
  }
  ans=vmcount(1,":",2,":",n);
  ans(0)=2;
  for (ya_sizet k=3; k<=static_cast<ya_sizet>(sqrt(static_cast<double>(n))); 
       k=k+2)
    if (ans((k-1)/2)>0)
      ans(vmcount( (k*k-1)/2, ":", k, ":", length(ans)-1 ))=0;
  return YA_MatI(ans(find(ans>0)));
}

// ---- Factor
inline YA_DynI factor(const ya_sizet num) {
  YA_DynI ans;
  if (num<4) {
    ans=num;
    return ans;
  }
  
  ya_sizet n=num;
  YA_MatI pnums=primes(static_cast<ya_sizet>(sqrt(static_cast<double>(n))));
  while (n>1) {
    YA_DynI d = find( (n%pnums) == 0);
    if (length(d)==0) {
      ans.push_back(n);
      break;
    }
    pnums=YA_MatI(pnums(d));
    ya_sizet pN=pnums.numel();
    for (ya_sizet i=0; i<pN; i++)
      ans.push_back(pnums(i));
    n = n/product(pnums);
  }
  ipsort(ans);
  return ans;
}

// ---- Random permutation
template <class eltype, class mat_type, class Rng>
void randperm(const ya_sizet n, YA_BaseT& m, Rng &rng) {
  m=vmcount(n);
  ya_sizet half=n/2;
  for (ya_sizet i=0; i<half; i++) {
    ya_sizet ii=rng.rand(n-1); ya_sizet jj=rng.rand(n-1);
    eltype t=m(ii); m(ii)=m(jj); m(jj)=t;
  }
}

// ---- Random permutation
template <class eltype, class mat_type, class Rng>
void randperm(const ya_sizet n, const YA_WrapperT& m, Rng &rng) {
  m=vmcount(n);
  ya_sizet half=n/2;
  for (ya_sizet i=0; i<half; i++) {
    ya_sizet ii=rng.rand(n-1); ya_sizet jj=rng.rand(n-1);
    eltype t=m(ii); m(ii)=m(jj); m(jj)=t;
  }
}

// ---- Random permutation
template <class Rng>
YA_RowI randperm(const ya_sizet n, Rng &rng) {
  YA_RowI ans;
  randperm(n,ans,rng);
  return ans;
}

// ---- K-nearest neighbors
template<class eltype, class mat_type,
         class ya_type2, class ya_type3>
void kneighbors(const YA_BaseT &in, const ya_sizet k, ya_type2 &neighbors,
                ya_type3 &dists, const int verbose) {
  YA_DEBUG_ERROR(k>0,"Cannot compute 0 nearest neighbors.");
  YA_DEBUG_ERROR(k<in.rows(),"More neighbors than data for kneighbors");

  neighbors.setup(in.rows(),k);
  dists.setup(in.rows(),k);

  #ifdef CUDA
  #ifdef YA_PROGMETER
  YA_TimeKeeper cuda_tk;
  if (verbose)
    cuda_tk.start();
  #endif
  YA_Mat<float> in_float(vm_cast<float>::sc(in));
  YA_Mat<float> dists_float(in.rows(),k);
  cuda_kneighbors<float>(in_float.c_ptr(),k,neighbors.c_ptr(),
                          dists_float.c_ptr(),in.cols(),in.rows());
  dists=vm_cast<eltype>::sc(dists_float);
  #ifdef YA_PROGMETER
  if (verbose) {
    cuda_tk.end();
    cerr << "Done. " << cuda_tk << ".\n";
  }
  #endif
  return;
  // YA_MatI ind(neighbors);
  #endif
          
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    #ifdef YA_OMP
    ya_sizet ptot=(in.rows()*in.rows()+in.rows())/2;
    #else
    ya_sizet ptot=in.rows();
    #endif
    pm.start("Calculating k-neighbors:",70,ptot,false);
  }
  #endif
  
  neighbors.setup(in.rows(),k);
  dists.setup(in.rows(),k);

  std::vector<std::multimap<double, ya_sizet> > distances(in.rows());
  
  ya_sizet lrows=in.rows();

  #pragma omp parallel
  {
  #pragma omp for
  for (ya_sizet i=0; i<lrows; i++) {
    #ifdef YA_OMP // Remove race conditions efficiently with full loop
    for (ya_sizet j=0; j<lrows; j++) {
      if (i==j) continue;
    #else
    for (ya_sizet j=i+1; j<lrows; j++) {
    #endif
      eltype dist=0;
      ya_sizet iC=in.cols();
      for (ya_sizet c=0; c<iC; c++) {
        eltype v=in(i,c)-in(j,c);
        dist+=v*v;
      }

      // Keep track of the smallest k distances and indices
      if (static_cast<ya_sizet>(distances[i].size())<k)
        distances[i].insert(std::pair<double, ya_sizet>(dist,j));
      else if (dist<distances[i].rbegin()->first) {
        distances[i].erase(--distances[i].end());
        distances[i].insert(std::pair<double, ya_sizet>(dist,j));
      }
      #ifndef YA_OMP
      if (static_cast<ya_sizet>(distances[j].size())<k)
        distances[j].insert(std::pair<double, ya_sizet>(dist,i));
      else if (dist<distances[j].rbegin()->first) {
        distances[j].erase(--distances[j].end());
        distances[j].insert(std::pair<double, ya_sizet>(dist,i));
      }
      #endif
    }
    #ifdef YA_PROGMETER
    if (verbose) {
      #ifndef YA_OMP
      pm.iterate();
      #else
      pm.iterate(in.rows()-i);
      #endif
    }
    #endif
  }
  
  // Move the indices into the matrix
  #pragma omp for schedule(static)
  for (ya_sizet i=0; i<lrows; i++) {
    ya_sizet j=0;
    for (std::multimap<double,ya_sizet>::iterator m=distances[i].begin();
         m!=distances[i].end(); m++) {
      neighbors(i,j)=m->second;
      dists(i,j)=sqrt(m->first);
      j++;
    }
  }
  }
  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  
    if (verbose>1) {
      cerr << "Neighbor Distance Stats:\n";
      eltype dmin, dmax, dmean, dstd;
      vstat(dists,dmin,dmax,dmean,dstd);
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
    }
  }
  #endif
  
  #ifdef CUDA
  //cout << "Neighbor differences: " << maximum(neighbors-ind) << " "
  //     << minimum(neighbors-ind) << endl;
  //cout << "Distance differences: " << maximum(vm_cast<float>::sc(dists)-dists_float) << " "
  //     << minimum(vm_cast<float>::sc(dists)-dists_float) << endl;
  #endif
}

template<class eltype, class mat_type>
void eneighbors(const YA_BaseT &input, const eltype epsilon,                   
    std::vector<YA_DynI> &output, std::vector<YA_DynT> &dists,
                const int verbose) {
  YA_DEBUG_ERROR(epsilon>0,"Cannot compute 0 nearest neighbors");
                  
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    #ifdef YA_OMP
    ya_sizet ptot=(input.rows()*input.rows()+input.rows())/2;
    #else
    ya_sizet ptot=input.rows();
    #endif
    pm.start("Calculating e-neighbors:",70,ptot,false);
  }
  #endif
                  
  ya_sizet nrows = input.rows();
  output.resize(nrows);
  dists.resize(nrows);

  double eps2=epsilon*epsilon;                  
  #pragma omp parallel
  {
  #pragma omp for
  for (ya_sizet i=0; i<nrows; i++) {
    #ifdef YA_OMP // Remove race conditions efficiently with full loop
    for (ya_sizet j=0; j<nrows; j++) {
      if (i==j) continue;
    #else
    for (ya_sizet j=i+1; j<nrows; j++) {
    #endif
      double dist = 0.0;
      ya_sizet iC=input.cols();
      for (ya_sizet c=0; c<iC; c++) {
        double v = input(i,c) - input(j,c);
        dist+=v*v;
      }
      if (dist <= eps2) {
        output[i].push_back(j);
        dists[i].push_back(sqrt(dist));
        #ifndef YA_OMP
        output[j].push_back(i);
        dists[j].push_back(sqrt(dist));
        #endif
      }
    }
    #ifdef YA_PROGMETER
    if (verbose) {
      #ifndef YA_OMP
      pm.iterate();
      #else
      pm.iterate(input.rows()-i);
      #endif
    }
    #endif
  }
  }
  #ifdef YA_PROGMETER
  if (verbose) { 
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";

    if (verbose>1) {
      cerr << "Neighbor Counts Stats:\n";
      ya_sizet dmin, dmax;
      eltype dmean, dstd;
      ya_sizet n=output.size();
      YA_ColI counts(n);
      for (ya_sizet i=0; i<n; i++)
        counts(i)=output[i].numel();
  
      vstat(counts,dmin,dmax,dmean,dstd);
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
    }
  }
  #endif
}

template<class eltype, class mat_type,class ya_type2, class ya_type3,
         class ya_type4>
void kneighbors(const YA_BaseT &input, const ya_type2 &nbors, const ya_sizet k, 
                ya_type3 &output, ya_type4 &dists, const int verbose) {
  YA_DEBUG_ERROR(k>0,"Cannot compute 0 nearest neighbors.");
  YA_DEBUG_ERROR(k<=nbors.rows(),"More neighbors than data for kneighbors");

  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Calculating k-neighbors:",70,input.rows(),false);
  }
  #endif
  
  std::vector<std::multimap<double, ya_sizet> > distances(input.rows());
  output.setup(input.rows(),k);
  dists.setup(input.rows(),k);
  ya_sizet lrows=input.rows();
  ya_sizet nR=nbors.rows();
  #pragma omp parallel
  {
  #pragma omp for
  for (ya_sizet i=0; i<lrows; i++) {
    for (ya_sizet j=0; j<nR; j++) {
      double dist=norm(input(i,":")-nbors(j,":"));

      // Keep track of the smallest k distances and indices
      if (static_cast<ya_sizet>(distances[i].size())<k)
        distances[i].insert(std::pair<double, ya_sizet>(dist,j));
      else if (dist<distances[i].rbegin()->first) {
        distances[i].erase(--distances[i].end());
        distances[i].insert(std::pair<double, ya_sizet>(dist,j));
      }
    }
    #ifdef YA_PROGMETER
    if (verbose) pm.iterate();
    #endif
  }

  // Move the indices into the matrix
  #pragma omp for
  for (ya_sizet i=0; i<lrows; i++) {
    ya_sizet j=0;
    for (std::multimap<double,ya_sizet>::iterator m=distances[i].begin();
         m!=distances[i].end(); m++) {
      output(i,j)=m->second;
      dists(i,j)=m->first;
      j++;
    }
  }
  }
  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";

    if (verbose>1) {
      cerr << "Neighbor Distance Stats:\n";
      eltype dmin, dmax, dmean, dstd;
      vstat(dists,dmin,dmax,dmean,dstd);
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
    }
  }
  #endif
}

template<class eltype, class mat_type, class ya_type2>
void eneighbors(const YA_BaseT &input, const ya_type2 &nbors, 
                const eltype epsilon, std::vector<YA_DynI> &output,
                std::vector<YA_DynT> &dists, const int verbose) {
  YA_DEBUG_ERROR(epsilon>0,"Cannot compute 0 nearest neighbors.");

  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Calculating E-neighbors:",70,input.rows(),false);
  }
  #endif
  
  ya_sizet lrows=input.rows();
  output.resize(lrows);
  dists.resize(lrows);
  ya_sizet nR=nbors.rows();
  #pragma omp parallel
  {
  #pragma omp for
  for (ya_sizet i=0; i<lrows; i++) {
    for (ya_sizet j=0; j<nR; j++) {
      double dist=norm(input(i,":")-nbors(j,":"));
        if (dist <= epsilon) {
          output[i].push_back(j);
          dists[i].push_back(dist);
        }
      }
    }
    #ifdef YA_PROGMETER
    if (verbose) pm.iterate();
    #endif
  }

  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";

    if (verbose>1) {
      cerr << "Neighbor Counts Stats:\n";
      ya_sizet dmin, dmax;
      eltype dmean, dstd;
      ya_sizet n=output.size();
      YA_ColI counts(n);
      for (ya_sizet i=0; i<n; i++)
        counts(i)=output[i].numel();
  
      vstat(counts,dmin,dmax,dmean,dstd);
      cerr << "  Min: " << dmin << " Max: " << dmax << endl
           << "  Mean: " << dmean << " Std: " << dstd << endl << endl;
    }
  }
  #endif
}

// ---- 1D Plot (using GNU plot)
template<class eltype, class mat_type>
int plot(const YA_BaseT &in, const std::string titles, const char cstyle, const std::string xlabel, const std::string ylabel) {
  std::ofstream out;
  int err_flag = ya_fileopen(out,"482vm897.vmt");
  if (err_flag!=0)
    return err_flag;

  // Write the matrix to a temporary file
  ya_sizet count;
  ya_sizet points;
  if (in.rows()==1) {
    out << in.T() << std::endl;
    count=1;
    points=in.numel();
  } else {
    out << in << std::endl;
    count=in.cols();
    points=in.rows();
  }
  err_flag = ya_fileclose(out);
  if (err_flag!=0)
    return err_flag;
  
  // Style of plot
  std::string style;
  if (cstyle=='+')
    style="with linespoints";
  else if (cstyle=='o')
    style="with points";
  else if (cstyle=='.')
    style="with dots";
  else
    style="smooth csplines";
  std::string spoints=ya_to_ascii(points);
    
  // Titles for plot
  std::vector<std::string> title_mat;
  ya_get_tokens(',',titles,title_mat);
  while (static_cast<ya_sizet>(title_mat.size())<count)
    title_mat.push_back("no_title");
  
  std::string command("echo ");
  #ifndef _MSC_VER
  command += "\"";
  #endif
  command += "set xlabel '";
  command += xlabel;
  command += "' ; ";
  command += "set ylabel '";
  command += ylabel;
  command += "' ; plot ";
  for (ya_sizet i=0; i<count; i++) {
    command += "'482vm897.vmt' using "+ya_to_ascii(i+1)+" "+style+" title '"+ title_mat[i];
    if (i!=count-1)
       command += "', ";
  }
  command += "'";

  #ifndef _MSC_VER
  command += "\"";
  #endif

  command += " | gnuplot -persist";

  if (system(command.c_str())!=0) {
    ya_addwarn(308,9,"vm_fun_plot",
               "Could not plot matrix file due to system error.");
    return 308;
  }
  if (remove("482vm897.vmt")!=0) {
    ya_addwarn(309,9,"vm_fun_plot",
                "Could not delete temporary matrix file due to system error.");
    return 309;
  } 
  return err_flag;
}

// ---- 2D color Plot (using GNU plot)
int plot(int* colors, const YA_MatD &output){
  std::ofstream outfile;
  int err_flag = ya_fileopen(outfile,"482vm897.vmt");
  if (err_flag != 0)
    return err_flag;
  int nrows = output.rows();
  for (int i=0; i<nrows; i++)
    outfile<<output(i, 0)<< " "<< output(i, 1) << " " << colors[i] << std::endl;
  err_flag = ya_fileclose(outfile);
  if (err_flag != 0)
    return err_flag;
  std::ofstream outfilec;
  err_flag = ya_fileopen(outfilec,"command.gnu");
  if (err_flag != 0)
    return err_flag;
  outfilec<<"set palette rgb 33,13,10\n";
  outfilec<<"set view map\n";
  outfilec<<"set cbrange [1:10]\n";
  outfilec<<"splot \"482vm897.vmt\" with points palette pt 9\n";
  err_flag=ya_fileclose(outfilec);
  if (err_flag != 0)
    return err_flag;
  std::string spec("at");
  spec[0] = static_cast<char>(92);
  spec[1]= static_cast<char>(34);
  // when running the Visual Studio executables from cygwin, gnuplot croaks
  // with the trick above involving the "spec" - the escape character isn't
  // recognized. it isn't clear whether this is related to cygwin or Windows.
  // in any case, this work-around fixes the problem.
#ifndef _MSC_VER
  std::string command="echo load " + spec + "command.gnu" + spec+  " | gnuplot -persist";
#else
  string command="echo load " + std::string("\"") + "command.gnu" + std::string("\"") +  " | gnuplot -persist";
#endif

  if (system(command.c_str())!=0) {
    ya_addwarn(308,9,"vm_fun_plot",
               "Could not plot matrix file due to system error.");
    return 308;
  }
  if (remove("482vm897.vmt")!=0) {
    ya_addwarn(309,9,"vm_fun_plot",
                "Could not delete temporary matrix file due to system error.");
    return 309;
  } 
  if (remove("command.gnu")!=0) {
    ya_addwarn(309,9,"vm_fun_plot",
                "Could not delete temporary matrix file due to system error.");
    return 309;
  } 
  return err_flag;
}


// ---- 2D Plot (using GNU plot)
template<class eltype, class mat_type, class eltype2, class mat_type2>
int plot(const YA_BaseT &xin, const YA_BaseT2T &yin, const std::string titles,
         const char cstyle, const std::string xlabel, const std::string ylabel) {
  std::ofstream out;
  int err_flag = ya_fileopen(out,"482vm897.vmt");
  if (err_flag != 0)
    return err_flag;

  // Write the matrix to a temporary file
  ya_sizet xcount,ycount;
  if (xin.rows()==1) {
    xcount=1;
    if (yin.rows()==1) {
      out << concat(YA_MatT(xin.T()),YA_MatT(yin.T())) << std::endl;
      ycount=1;
    } else {
      out << concat(YA_MatT(xin.T()),YA_MatT(yin)) << std::endl;
      ycount=yin.cols();
    }
  } else {
    xcount=xin.cols();
    if (yin.rows()==1) {
      ycount=1;
      out << concat(YA_MatT(xin),YA_MatT(yin.T())) << std::endl;
    } else {
      ycount=yin.cols();
      out << concat(YA_MatT(xin),YA_MatT(yin)) << std::endl;
    }
  }
  err_flag=ya_fileclose(out);
  if (err_flag != 0)
    return err_flag;

  YA_DEBUG_ERROR(xcount>0 && xcount<=ycount && (xcount==1 || xcount==ycount),
                 "Improper matrix dimensions for plot");
  
  // Style of plot
  std::string style;
  if (cstyle=='+')
    style="with linespoints";
  else if (cstyle=='o')
    style="with points";
  else if (cstyle=='.')
    style="with dots";
  else
    style="smooth csplines";
    
  // Titles for plot
  std::vector<std::string> title_mat;
  ya_get_tokens(',',titles,title_mat);
  while (static_cast<ya_sizet>(title_mat.size())<xcount)
    title_mat.push_back("no_title");
  
  std::string command("echo ");
  #ifndef _MSC_VER
  command += "\"";
  #endif
  command += "set xlabel '";
  command += xlabel;
  command += "' ; ";
  command += "set ylabel '";
  command += ylabel;
  command += "' ; plot ";
  ya_sizet xi=1, yi=xcount+1;
  for (ya_sizet i=0; i<ycount; i++) {
    command+="'482vm897.vmt' using "+ya_to_ascii(xi)+":"+ya_to_ascii(yi)+" "+
             style+" title '"+title_mat[i];
    if (i!=ycount-1)
       command+="', ";
    else
      #ifdef _MSC_VER
      command+="' | gnuplot -persist";
      #else
      command+="'\" | gnuplot -persist";
      #endif
    yi++;
    if (xcount!=1)
      xi++;
  }
  if (system(command.c_str())!=0) {
    ya_addwarn(308,9,"vm_fun_plot",
               "Could not plot matrix file due to system error.");
    return 308;
  }
  if (remove("482vm897.vmt")!=0) {
    ya_addwarn(309,9,"vm_fun_plot",
                "Could not delete temporary matrix file due to system error.");
    return 309;
  } 
  return err_flag;
}

}

// ---- Max for matrix
template<class eltype, class mat_type, class eltype2, class mat_type2,
         class eltype3, class mat_type3>
void maxi(const YA_BaseT &m, YA_BaseT2T &ans, YA_BaseT3T &ind) {
  ans=m(0,":");
  ind=vmzeros(1,m.cols());
  ya_sizet mC=m.cols(), mR=m.rows();
  for (ya_sizet i=0; i<mC; i++)
    for (ya_sizet j=1; j<mR; j++)
      if (m(j,i)>ans(i)) {
        ans(i)=m(j,i);
        ind(i)=j;
      }
}
// --- Min for matrix
template <class eltype, class mat_type>
YA_RowT min(const YA_BaseT &m) {
  YA_DEBUG_ERROR(m.rows()>0,"Cannot perform min on empty matrix");
  YA_RowT ans=m(0,":");
  ya_sizet mC=m.cols(), mR=m.rows();
  for (ya_sizet i=0; i<mC; i++)
    for (ya_sizet j=0; j<mR; j++)
      if (ans(i)>m(j,i))
        ans(i)=m(j,i);
  return ans;
}
// --- Min for matrix
template <class eltype, class mat_type,class ya_type2>
YA_RowT min(const YA_BaseT &m, ya_type2 &indices) {
  YA_DEBUG_ERROR(m.rows()>0,"Cannot perform min on empty matrix");
  YA_RowT ans=m(0,":");
  indices=vm_cast<ya_sizet>::sc(vmzeros(1,m.cols()));
  ya_sizet mC=m.cols(), mR=m.rows();
  for (ya_sizet i=0; i<mC; i++)
    for (ya_sizet j=0; j<mR; j++)
      if (ans(i)>m(j,i)) {
        ans(i)=m(j,i);
        indices(i)=j;
      }
  return ans;
}
    
// ---- Max for vector
template<class eltype, class mat_type>
void max(const YA_BaseT &m, eltype &ans, ya_sizet &ind) {
  ans=m(0);
  ind=0;
  ya_sizet mN=m.numel();
  for (ya_sizet j=1; j<mN; j++)
     if (m(j)>ans) {
        ans=m(j);
        ind=j;
      }
}

// ---- Sort
template<class eltype, class mat_type>
YA_MatT sort(const YA_BaseT &m) {
  YA_MatT ans(m);
  ipsort(ans);
  return ans;
}

// ---- Sort
template<class eltype, class mat_type, class eltype2, class mat_type2>
void sort(const YA_BaseT &m, YA_BaseT2T &ans) {
  ans=m;
  ipsort(ans);
}

template<class eltype, class mat_type,class eltype2, class mat_type2,
         class ya_type2>
void sorti(const YA_BaseT &m, ya_type2 &ans, YA_BaseT2T &indices) {
  sort_indices(m,indices);
  ans=m(indices);
  ya_sizet counter=indices.rows();
  ya_sizet iC=indices.cols();
  for (ya_sizet i=1; i<iC; i++) {
    indices(":",i)-=counter;
    counter+=indices.rows();
  }
}

#endif
