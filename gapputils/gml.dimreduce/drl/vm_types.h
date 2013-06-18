/***************************************************************************
                                 vm_types.h
                             -------------------
                               W. Michael Brown

  Definitions and types for VecMat classes

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Oct 27 2006
    authors    : W. Michael Brown

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Fri Oct 27 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov

  New types
  
  D = double
  S = float
  Z = Complex
  C = Complex
  I = Int
  T = Template


 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef VM_TYPES_H
#define VM_TYPES_H

#include <map>
#include <vector>

YALANAMESPACE

// --------------------------- Constants
///\def Infinity
#define VM_INF std::numeric_limits<eltype>::max()
///\def -Infinity
#define VM_MINUS_INF -std::numeric_limits<eltype>::max()
///\def Not-a-Number
#define VM_NAN std::numeric_limits<eltype>::nan()

// Deciding if a type is supported by blas
template <class eltype> struct _VM_BLAS_TYPE;
template <> struct _VM_BLAS_TYPE<double> { enum { supported=1 }; };
template <> struct _VM_BLAS_TYPE<float> { enum { supported=1 }; };
template <class eltype> struct _VM_BLAS_TYPE { enum { supported=0 }; };

// Helper to decide the order for the transpose of a matrix
template <int order> struct _ya_trans_order;
template <> struct _ya_trans_order<0> { enum { ans=1 }; };
template <> struct _ya_trans_order<1> { enum { ans=0 }; };
template <int order> struct _ya_trans_order { enum { ans=order }; };

template <int> struct _ya_const_zero;
template <> struct _ya_const_zero<0> { enum { ans=6 }; };
template <int> struct _ya_const_zero { enum { ans=0 }; };

template <int o1, int o2> struct _YA_TBIN_ORDER;
template <> struct _YA_TBIN_ORDER<1,1> { enum { ans=1 }; };
template <> struct _YA_TBIN_ORDER<1,2> { enum { ans=1 }; };
template <> struct _YA_TBIN_ORDER<2,1> { enum { ans=1 }; };
template <> struct _YA_TBIN_ORDER<4,4> { enum { ans=4 }; };
template <int o1> struct _YA_TBIN_ORDER<o1,4> { enum { ans=4 }; };
template <int o2> struct _YA_TBIN_ORDER<4,o2> { enum { ans=4 }; };
template <int o1, int o2> struct _YA_TBIN_ORDER { enum { ans=0}; };

// --------------------------- Matrices

template <class eltype> class VecMatSym;
///\def VM_SymMat
#define VM_SymMat VecMatSym<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_SymMat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,VM_SymMat > iter; 
  typedef YA_ConstIter<eltype,const VM_SymMat > citer;
  typedef YA_RowIter<eltype,VM_SymMat > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_SymMat > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 2,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 1,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 4,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_SymMat > :
public YA_Traits<VM_SymMat > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_SymMat > iter; 
  typedef YA_ConstRowIter<eltype,const VM_SymMat > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype& ref; 
};

template <class eltype, ya_sizet nrows> class VecMatSymF;
///\def VM_SymMatF
#define VM_SymMatF VecMatSymF<eltype,nrows>

// Matrix Specific Traits
template <class eltype, ya_sizet nrows>
class YA_Traits<VM_SymMatF > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,VM_SymMatF > iter; 
  typedef YA_ConstIter<eltype,const VM_SymMat > citer;
  typedef YA_RowIter<eltype,VM_SymMatF > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_SymMat > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype& ref; 
  typedef const eltype& cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 2,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 1,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 4,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, ya_sizet nrows>
class YA_Traits<const VM_SymMatF > :
public YA_Traits<VM_SymMatF > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_SymMatF > iter; 
  typedef YA_ConstRowIter<eltype,const VM_SymMatF > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype& ref; 
};

template <class eltype> class _vm_sparse_zero;
template <class eltype> class _vm_sparse_inf;
template <class eltype, class mat_type> class VMSparseReturn;
template <class eltype, class zerotype=_vm_sparse_zero<eltype> > 
class VecMatSparse;
///\def VM_Sparse
#define VM_Sparse VecMatSparse<eltype,zerotype>

// Matrix Specific Traits
template <class eltype, class zerotype>
class YA_Traits<VM_Sparse > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,VM_Sparse > iter;
  typedef YA_ConstIter<eltype,const VM_Sparse > citer;
  typedef YA_RowIter<eltype,VM_Sparse > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_Sparse > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const VMSparseReturn<eltype,VM_Sparse > ref;
  typedef const VMSparseReturn<eltype,const VM_Sparse > cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 4,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class zerotype>
class YA_Traits<const VM_Sparse > :
public YA_Traits<VM_Sparse > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Sparse > iter;
  typedef YA_ConstRowIter<eltype,const VM_Sparse > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const VMSparseReturn<eltype,const VM_Sparse > ref;
};

template <class eltype, class mat_type> class VecMatC;
///\def VM_Column
#define VM_Column VecMatC<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_Column > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::iter iter;
  typedef typename YA_Traits<mat_type>::iter citer;
  typedef typename YA_Traits<mat_type>::iter rowiter;
  typedef typename YA_Traits<mat_type>::iter crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = _VM_TEMPLATE_VALUE<YA_TRAIT::VME_MEMORY>::isone,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 1,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = 1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = YA_TRAIT::VME_BLAS,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = YA_TRAIT::VME_USE_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_Column > :
public YA_Traits<VM_Column > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::iter iter;
  typedef typename YA_Traits<mat_type>::iter rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatR;
///\def VM_Row
#define VM_Row VecMatR<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_Row > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::rowiter iter;
  typedef typename YA_Traits<mat_type>::crowiter citer;
  typedef typename YA_Traits<mat_type>::rowiter rowiter;
  typedef typename YA_Traits<mat_type>::crowiter crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = 1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_Row > : 
public YA_Traits<VM_Row > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::rowiter iter;
  typedef typename YA_Traits<mat_type>::rowiter rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type, class row_type, class col_type>
class VecMatS;
///\def VM_Slice
#define VM_Slice VecMatS<eltype,mat_type,rows_type,cols_type>

// Matrix Specific Traits
template <class eltype, class mat_type, class rows_type, class cols_type>
class YA_Traits<VM_Slice > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Slice > iter;
  typedef YA_Iter<eltype,const VM_Slice > citer;
  typedef YA_RowIter<eltype,const VM_Slice > rowiter;
  typedef YA_RowIter<eltype,const VM_Slice > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type, class rows_type, class cols_type>
class YA_Traits<const VM_Slice > :
public YA_Traits<VM_Slice > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Slice > iter;
  typedef YA_RowIter<eltype,const VM_Slice > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatDiag;
///\def VM_Diag
#define VM_Diag VecMatDiag<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_Diag > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Diag > iter;
  typedef YA_Iter<eltype,const VM_Diag > citer;
  typedef YA_RowIter<eltype,const VM_Diag > rowiter;
  typedef YA_RowIter<eltype,const VM_Diag > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref;
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 1,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_Diag > :
public YA_Traits<VM_Diag > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Diag > iter;
  typedef YA_RowIter<eltype,const VM_Diag > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref;
};

template <class eltype, class mat_type, class mat_type2> class VecMatReI;
///\def VM_ReI
#define VM_ReI VecMatReI<eltype,mat_type,mat_type2>

// Matrix Specific Traits
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<VM_ReI > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ReI > iter;
  typedef YA_Iter<eltype,const VM_ReI > citer;
  typedef YA_RowIter<eltype,const VM_ReI > rowiter;
  typedef YA_RowIter<eltype,const VM_ReI > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = YA_TRAIT2::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT2::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT2::VME_ROW_VECTOR,
    VME_SYMMETRIC = YA_TRAIT2::VME_SYMMETRIC,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<const VM_ReI > : 
public YA_Traits<VM_ReI > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ReI > iter;
  typedef YA_RowIter<eltype,const VM_ReI > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatReshape;
///\def VM_Reshape
#define VM_Reshape VecMatReshape<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_Reshape > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::iter iter;
  typedef typename YA_Traits<mat_type>::citer citer;
  typedef YA_RowIter<eltype,const VM_Reshape > rowiter;
  typedef YA_RowIter<eltype,const VM_Reshape > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = YA_TRAIT::VME_MEMORY,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = YA_TRAIT::VME_BLAS,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = YA_TRAIT::VME_USE_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_Reshape > : 
public YA_Traits<VM_Reshape > { 
 public:
  // Iterators
  typedef typename YA_Traits<mat_type>::iter iter;
  typedef YA_RowIter<eltype,const VM_Reshape > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatCast;
///\def VM_Cast
#define VM_Cast VecMatCast<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VecMatCast<eltype, mat_type> > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Cast> iter;
  typedef YA_Iter<eltype,const VM_Cast> citer;
  typedef YA_RowIter<eltype,const VM_Cast> rowiter;
  typedef YA_RowIter<eltype,const VM_Cast> crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = YA_TRAIT::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_SYMMETRIC = YA_TRAIT::VME_SYMMETRIC,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VecMatCast<eltype, mat_type> > : 
public YA_Traits<VecMatCast<eltype, mat_type> > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Cast> iter;
  typedef YA_RowIter<eltype,const VM_Cast> rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef eltype ref; 
};

template <class eltype, class mat_type> class VecMatRepMat;
///\def VM_RepMat
#define VM_RepMat VecMatRepMat<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_RepMat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_RepMat > iter;
  typedef YA_Iter<eltype,const VM_RepMat > citer;
  typedef YA_RowIter<eltype,const VM_RepMat > rowiter;
  typedef YA_RowIter<eltype,const VM_RepMat > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_RepMat >  :
public YA_Traits<VM_RepMat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_RepMat > iter;
  typedef YA_RowIter<eltype,const VM_RepMat > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatRowRep;
///\def VM_RowRep
#define VM_RowRep VecMatRowRep<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_RowRep > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_RowRep > iter;
  typedef YA_Iter<eltype,const VM_RowRep > citer;
  typedef YA_RowIter<eltype,const VM_RowRep > rowiter;
  typedef YA_RowIter<eltype,const VM_RowRep > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_RowRep > : 
public YA_Traits<VM_RowRep > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_RowRep > iter;
  typedef YA_RowIter<eltype,const VM_RowRep > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type> class VecMatColRep;
///\def VM_ColRep
#define VM_ColRep VecMatColRep<eltype,mat_type>

// Matrix Specific Traits
template <class eltype, class mat_type>
class YA_Traits<VM_ColRep > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ColRep > iter;
  typedef YA_Iter<eltype,const VM_ColRep > citer;
  typedef YA_RowIter<eltype,const VM_ColRep > rowiter;
  typedef YA_RowIter<eltype,const VM_ColRep > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type>
class YA_Traits<const VM_ColRep > : 
public YA_Traits<VM_ColRep > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ColRep > iter;
  typedef YA_RowIter<eltype,const VM_ColRep > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type, class mat_type2> class VecMatConcat;
///\def VM_Concat
#define VM_Concat VecMatConcat<eltype,mat_type,mat_type2>

// Matrix Specific Traits
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<VM_Concat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Concat > iter;
  typedef YA_Iter<eltype,const VM_Concat > citer;
  typedef YA_RowIter<eltype,const VM_Concat > rowiter;
  typedef YA_RowIter<eltype,const VM_Concat > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_ROW_VECTOR,YA_TRAIT2::VME_ROW_VECTOR>::isone,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_ROW_VECTOR,YA_TRAIT2::VME_ROW_VECTOR>::isone,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<const VM_Concat > : 
public YA_Traits<VM_Concat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Concat > iter;
  typedef YA_RowIter<eltype,const VM_Concat > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype, class mat_type, class mat_type2> class VecMatConcatR;
///\def VM_ConcatR
#define VM_ConcatR VecMatConcatR<eltype,mat_type,mat_type2>

// Matrix Specific Traits
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<VM_ConcatR > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ConcatR > iter;
  typedef YA_Iter<eltype,const VM_ConcatR > citer;
  typedef YA_RowIter<eltype,const VM_ConcatR > rowiter;
  typedef YA_RowIter<eltype,const VM_ConcatR > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
  typedef typename YA_Traits<mat_type>::cref cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 5,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_COL_VECTOR,YA_TRAIT2::VME_COL_VECTOR>::isone,
    VME_COL_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_COL_VECTOR,YA_TRAIT2::VME_COL_VECTOR>::isone,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, class mat_type, class mat_type2>
class YA_Traits<const VM_ConcatR > : 
public YA_Traits<VM_ConcatR > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_ConcatR > iter;
  typedef YA_RowIter<eltype,const VM_ConcatR > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef typename YA_Traits<mat_type>::ref ref;
};

template <class eltype> class VecMatScalar;
///\def VM_Scalar
#define VM_Scalar VecMatScalar<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_Scalar > { 
 public:
  // Iterators
  typedef eltype* iter; 
  typedef const eltype * citer;
  typedef eltype* rowiter;
  typedef const eltype* crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 1,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = _VM_BLAS_TYPE<eltype>::supported,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = 0,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_Scalar > :
public YA_Traits<VM_Scalar > { 
 public:
  // Iterators
  typedef const eltype * iter; 
  typedef const eltype* rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype> class VecMatCount;
///\def VM_CountM
#define VM_CountM VecMatCount<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_CountM > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_CountM > iter;
  typedef YA_ConstIter<eltype,const VM_CountM > citer;
  typedef YA_ConstRowIter<eltype,const VM_CountM > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_CountM > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_CountM > : 
public YA_Traits<VM_CountM > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_CountM > iter;
  typedef YA_ConstRowIter<eltype,const VM_CountM > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype> class VecMatCountV;
///\def VM_Count
#define VM_Count VecMatCountV<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_Count > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Count > iter;
  typedef YA_ConstIter<eltype,const VM_Count > citer;
  typedef YA_ConstRowIter<eltype,const VM_Count > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_Count > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_Count > : 
public YA_Traits<VM_Count > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Count > iter;
  typedef YA_ConstRowIter<eltype,const VM_Count > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype> class VecMatSeriesV;
///\def VM_Series
#define VM_Series VecMatSeriesV<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_Series > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Series > iter;
  typedef YA_ConstIter<eltype,const VM_Series > citer;
  typedef YA_ConstRowIter<eltype,const VM_Series > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_Series > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_Series > : 
public YA_Traits<VM_Series > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Series > iter;
  typedef YA_ConstRowIter<eltype,const VM_Series > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype, int count_start> class VMCountFromI;
///\def VM_FromI
#define VM_FromI VMCountFromI<eltype,count_start>

// Matrix Specific Traits
template <class eltype, int count_start>
class YA_Traits<VM_FromI > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_FromI > iter;
  typedef YA_ConstIter<eltype,const VM_FromI > citer;
  typedef YA_ConstRowIter<eltype,const VM_FromI > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_FromI > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, int count_start>
class YA_Traits<const VM_FromI > : 
public YA_Traits<VM_FromI > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_FromI > iter;
  typedef YA_ConstRowIter<eltype,const VM_FromI > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype=double> class VMLinSpace;
///\def VM_LinSpace
#define VM_LinSpace VMLinSpace<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_LinSpace > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_LinSpace > iter;
  typedef YA_ConstIter<eltype,const VM_LinSpace > citer;
  typedef YA_ConstRowIter<eltype,const VM_LinSpace > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_LinSpace > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_LinSpace > : 
public YA_Traits<VM_LinSpace > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_LinSpace > iter;
  typedef YA_ConstRowIter<eltype,const VM_LinSpace > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype, int cnum> class VecMatConstV;
///\def VM_ConstV
#define VM_ConstV VecMatConstV<eltype,cnum>

// Matrix Specific Traits
template <class eltype, int cnum>
class YA_Traits<VM_ConstV > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_ConstV > iter;
  typedef YA_ConstIter<eltype,const VM_ConstV > citer;
  typedef YA_ConstRowIter<eltype,const VM_ConstV > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_ConstV > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = _ya_const_zero<cnum>::ans,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 1,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, int cnum>
class YA_Traits<const VM_ConstV > : 
public YA_Traits<VM_ConstV > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_ConstV > iter;
  typedef YA_ConstRowIter<eltype,const VM_ConstV > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype, int cnum> class VecMatConst;
///\def VM_Const
#define VM_Const VecMatConst<eltype,cnum>

// Matrix Specific Traits
template <class eltype, int cnum>
class YA_Traits<VM_Const > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Const > iter;
  typedef YA_ConstIter<eltype,const VM_Const > citer;
  typedef YA_ConstRowIter<eltype,const VM_Const > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_Const > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = _ya_const_zero<cnum>::ans,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype, int cnum>
class YA_Traits<const VM_Const > :
public YA_Traits<VM_Const > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Const > iter;
  typedef YA_ConstRowIter<eltype,const VM_Const > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

template <class eltype> class VecMatEye;
///\def VM_Eye
#define VM_Eye VecMatEye<eltype>

// Matrix Specific Traits
template <class eltype>
class YA_Traits<VM_Eye > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Eye > iter;
  typedef YA_ConstIter<eltype,const VM_Eye > citer;
  typedef YA_ConstRowIter<eltype,const VM_Eye > rowiter;
  typedef YA_ConstRowIter<eltype,const VM_Eye > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref; 

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 1,
    VME_DIAGONAL = 1,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template <class eltype>
class YA_Traits<const VM_Eye > : 
public YA_Traits<VM_Eye > { 
 public:
  // Iterators
  typedef YA_ConstIter<eltype,const VM_Eye > iter;
  typedef YA_ConstRowIter<eltype,const VM_Eye > rowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 

  // Reference Types
  typedef const eltype ref; 
};

// --------------------- Expressions
  
template<class eltype, class contype, class mat_type, class op,
         class stor1=const mat_type&> class VM_Binary_ExprS;
#define VM_BinaryS VM_Binary_ExprS<eltype,contype,mat_type,op,stor1>

// Matrix Specific Traits
template<class eltype, class contype, class mat_type, class op,
         class stor1>
class YA_Traits<VM_BinaryS > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_BinaryS > iter;
  typedef YA_Iter<eltype,const VM_BinaryS > citer;
  typedef YA_RowIter<eltype,const VM_BinaryS > rowiter;
  typedef YA_RowIter<eltype,const VM_BinaryS > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = YA_TRAIT::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_SYMMETRIC = YA_TRAIT::VME_SYMMETRIC,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class contype, class mat_type, class op,
         class stor1>
class YA_Traits<const VM_BinaryS > : 
public YA_Traits<VM_BinaryS > {};

template<class eltype, class mat_type, class mat_type2, class op,
         class stor1=const mat_type&, class stor2=const mat_type2&>
class VM_Binary_Expr;
#define VM_Binary VM_Binary_Expr<eltype,mat_type,mat_type2,op,stor1,stor2>

// Matrix Specific Traits
template<class eltype, class mat_type, class mat_type2, class op,
         class stor1, class stor2>
class YA_Traits<VM_Binary > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Binary > iter;
  typedef YA_Iter<eltype,const VM_Binary > citer;
  typedef YA_RowIter<eltype,const VM_Binary > rowiter;
  typedef YA_RowIter<eltype,const VM_Binary > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref;
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_VECTOR,YA_TRAIT2::VME_VECTOR>::isone,
    VME_COL_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_COL_VECTOR,YA_TRAIT2::VME_COL_VECTOR>::isone,
    VME_ROW_VECTOR = _VM_TEMPLATE_OR_VALUE<YA_TRAIT::VME_ROW_VECTOR,YA_TRAIT2::VME_ROW_VECTOR>::isone,
    VME_SYMMETRIC = _VM_TEMPLATE_AND_VALUE<YA_TRAIT::VME_SYMMETRIC,YA_TRAIT2::VME_SYMMETRIC>::isone,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = _YA_TBIN_ORDER<YA_TRAIT::VME_ORDER,YA_TRAIT2::VME_ORDER>::ans,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

template<class eltype, class mat_type, class mat_type2, class op,
         class stor1, class stor2>
class YA_Traits<const VM_Binary > : 
public YA_Traits<VM_Binary > {};

template<class eltype, class mat_type> class VM_Size_Mat;
template<class eltype, class mat_type>
class YA_Traits<VM_Size_Mat<eltype,mat_type> > { 
 public:
  typedef YA_Iter<eltype,const VM_Size_Mat<eltype, mat_type> > iter;
  typedef YA_Iter<eltype,const VM_Size_Mat<eltype, mat_type> > citer;
  typedef YA_RowIter<eltype,const VM_Size_Mat<eltype, mat_type> > rowiter;
  typedef YA_RowIter<eltype,const VM_Size_Mat<eltype, mat_type> > crowiter;
  typedef iter cptr; 
  typedef citer const_cptr;
  typedef const ya_sizet ref; 
  typedef const ya_sizet cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

template<class eltype, class mat_type, class op, class stor1=const mat_type&>
class VM_Unary_Row;
#define VM_UnaryRow VM_Unary_Row<eltype,mat_type,op,stor1>

// Matrix Specific Traits
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<VM_UnaryRow > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_UnaryRow > iter;
  typedef YA_Iter<eltype,const VM_UnaryRow > citer;
  typedef YA_RowIter<eltype,const VM_UnaryRow > rowiter;
  typedef YA_RowIter<eltype,const VM_UnaryRow > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 1,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<const VM_UnaryRow > :
public YA_Traits<VM_UnaryRow > {};

template<class eltype, class mat_type, class op, class stor1=const mat_type&>
class VM_Unary_Col;
#define VM_UnaryCol VM_Unary_Col<eltype,mat_type,op,stor1>

// Matrix Specific Traits
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<VM_UnaryCol > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_UnaryCol > iter;
  typedef YA_Iter<eltype,const VM_UnaryCol > citer;
  typedef YA_RowIter<eltype,const VM_UnaryCol > rowiter;
  typedef YA_RowIter<eltype,const VM_UnaryCol > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 1,
    VME_COL_VECTOR = 1,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 2,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<const VM_UnaryCol > :
public YA_Traits<VM_UnaryCol > {};

template<class eltype, class mat_type, class op, class stor1=const mat_type&>
class VM_Unary_Mat;
#define VM_UnaryMat VM_Unary_Mat<eltype,mat_type,op,stor1>

// Matrix Specific Traits
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<VM_UnaryMat > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_UnaryMat > iter;
  typedef YA_Iter<eltype,const VM_UnaryMat > citer;
  typedef YA_RowIter<eltype,const VM_UnaryMat > rowiter;
  typedef YA_RowIter<eltype,const VM_UnaryMat > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = YA_TRAIT::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = YA_TRAIT::VME_ORDER,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<const VM_UnaryMat > : 
public YA_Traits<VM_UnaryMat> {};

template<class eltype, class mat_type, class mat_type2, class op, class size,
         class stor1=const mat_type&, class stor2=const mat_type2&>
class VM_Matrix_Expr;
#define VM_MatrixBin VM_Matrix_Expr<eltype,mat_type,mat_type2,op,size,stor1,stor2>

// Matrix Specific Traits
template<class eltype, class mat_type, class mat_type2, class op, class size,
         class stor1, class stor2>
class YA_Traits<VM_MatrixBin > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_MatrixBin > iter;
  typedef YA_Iter<eltype,const VM_MatrixBin > citer;
  typedef YA_RowIter<eltype,const VM_MatrixBin > rowiter;
  typedef YA_RowIter<eltype,const VM_MatrixBin > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = 0,
    VME_COL_VECTOR = 0,
    VME_ROW_VECTOR = 0,
    VME_SYMMETRIC = 0,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = 0,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class mat_type, class mat_type2, class op, class size,
         class stor1, class stor2>
class YA_Traits<const VM_MatrixBin > :
public YA_Traits<VM_MatrixBin> {};

template<class eltype, class mat_type, class op, class stor1=const mat_type&>
class VM_Unary_Expr;
#define VM_Unary VM_Unary_Expr<eltype,mat_type,op,stor1>

// Matrix Specific Traits
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<VM_Unary > { 
 public:
  // Iterators
  typedef YA_Iter<eltype,const VM_Unary > iter;
  typedef YA_Iter<eltype,const VM_Unary > citer;
  typedef YA_RowIter<eltype,const VM_Unary > rowiter;
  typedef YA_RowIter<eltype,const VM_Unary > crowiter;

  // Pointers for BLAS, etc.
  typedef iter cptr; 
  typedef citer const_cptr;

  // Reference Types
  typedef const eltype ref; 
  typedef const eltype cref;

  // See matrix_example.h for more info on types  
  enum MatrixInfo {
    VME_MEMORY = 0,
    VME_VECTOR = YA_TRAIT::VME_VECTOR,
    VME_COL_VECTOR = YA_TRAIT::VME_COL_VECTOR,
    VME_ROW_VECTOR = YA_TRAIT::VME_ROW_VECTOR,
    VME_SYMMETRIC = YA_TRAIT::VME_SYMMETRIC,
    VME_DIAGONAL = 0,
    VME_NCOLS = -1,
    VME_NROWS = -1,
    VME_ORDER = YA_TRAIT::VME_ORDER,
    VME_BLAS = 0,
    VME_BLAS_STRIDE = VME_BLAS,
    #ifdef YA_BLAS
    VME_USE_BLAS = VME_BLAS,
    #else
    VME_USE_BLAS = 0,
    #endif
    VME_TRANSPOSE = 0
  };
};

// Traits for const Matrices
template<class eltype, class mat_type, class op, class stor1>
class YA_Traits<const VM_Unary > :
public YA_Traits<VM_Unary > {};



class EigenOptions;

///\def VM_RowS
#define VM_RowS VM_Row

///\var typedef VecMatC<double> VM_ColumnSliceD;
typedef VecMatC<double,YA_MatD> VM_ColumnSliceD;
///\var typedef VecMatR<double> VM_RowSliceD;
typedef VecMatR<double,YA_MatD> VM_RowSliceD;
///\var typedef YA_Trans<double> YA_TransTD;
typedef YA_Trans<double,YA_MatD> YA_TransTD;
///\var typedef VecMatS<double> VM_SliceD;
typedef VecMatS<double,YA_MatD,YA_MatI,YA_MatI> VM_SliceD;
///\var typedef VecMatSym<double> VM_SymMatD;
typedef VecMatSym<double> VM_SymMatD;
///\var typedef VecMatSparse<double> VM_SparseD;
typedef VecMatSparse<double> VM_SparseD;
///\var typedef VecMatSparse<double,_vm_sparse_inf<double> > VM_SparseINF;
typedef VecMatSparse<double, _vm_sparse_inf<double> > VM_SparseINF;
///\var typedef VecMatScalar<double> VM_ScalarD;
typedef VecMatScalar<double> VM_ScalarD;
///\var typedef VecMatCountV<double> VM_CountD;
typedef VecMatCountV<double> VM_CountD;
///\var typedef VecMatSeriesV<double> VM_SeriesD;
typedef VecMatSeriesV<double> VM_SeriesD;
///\var typedef VecMatCount<double> VM_CountMatD;
typedef VecMatCount<double> VM_CountMatD;
///\var typedef VMLinSpace<double> VM_LinSpaceD;
typedef VMLinSpace<double> VM_LinSpaceD;
///\var typedef VecMatEye<double> VM_EyeD;
typedef VecMatEye<double> VM_EyeD;

///\var typedef VecMatC<ya_sizet> VM_ColumnSliceU;
typedef VecMatC<ya_sizet,YA_MatI> VM_ColumnSliceU;
///\var typedef VecMatR<ya_sizet> VM_RowSliceU;
typedef VecMatR<ya_sizet,YA_MatI> VM_RowSliceU;
///\var typedef YA_Trans<ya_sizet> YA_TransTU;
typedef YA_Trans<ya_sizet,YA_MatI> YA_TransTU;
///\var typedef VecMatS<ya_sizet> VM_SliceU;
typedef VecMatS<ya_sizet,YA_MatI,YA_VecI,YA_VecI> VM_SliceU;
///\var typedef VecMatSym<ya_sizet> VM_SymMatU;
typedef VecMatSym<ya_sizet> VM_SymMatU;
///\var typedef VecMatSparse<ya_sizet> VM_SparseU;
typedef VecMatSparse<ya_sizet> VM_SparseU;
///\var typedef VecMatScalar<ya_sizet> VM_ScalarU;
typedef VecMatScalar<ya_sizet> VM_ScalarU;
///\var typedef VecMatCountV<ya_sizet> VM_CountU;
typedef VecMatCountV<ya_sizet> VM_CountU;
///\var typedef VecMatSeriesV<ya_sizet> VM_SeriesU;
typedef VecMatSeriesV<ya_sizet> VM_SeriesU;
///\var typedef VecMatCount<ya_sizet> VM_CountMatU;
typedef VecMatCount<ya_sizet> VM_CountMatU;
///\var typedef VMCountFromI<ya_sizet,0> VM_ZeroToN;
typedef VMCountFromI<ya_sizet,0> VM_ZeroToN;
///\var typedef VecMatConstV<ya_sizet,0> VM_ZerosV;
typedef VecMatConstV<double,0> VM_ZerosV;
///\var typedef VecMatConstV<ya_sizet,1> VM_OnesV;
typedef VecMatConstV<double,1> VM_OnesV;
///\var typedef VecMatConst<ya_sizet,0> VM_Zeros;
typedef VecMatConst<double,0> VM_Zeros;
///\var typedef VecMatConst<ya_sizet,1> VM_Ones;
typedef VecMatConst<double,1> VM_Ones;

}
#endif
