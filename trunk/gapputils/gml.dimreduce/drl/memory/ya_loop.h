/***************************************************************************
                                  ya_loop.h
                             -------------------
                               W. Michael Brown

  Routines for looping over matrices
                               
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

#ifndef YA_LOOP_H
#define YA_LOOP_H

YALANAMESPACE

// *************************************************************************
// Loop over elements in one matrix in no necessary order
// *************************************************************************

template<class op,int order> struct ya_no_order_loop2;

template <int mem, int order> struct _ya_nool_order;
template <int order> struct _ya_nool_order<2,order> { enum { ans=0 }; };
template <int mem, int order> struct _ya_nool_order { enum { ans=order }; };

template<class op>
struct ya_no_order_loop {
  // Loop over elements in any order and apply op
  template<class mat_type>
  static inline void lp(mat_type &mat) {
    ya_no_order_loop2<op,_ya_nool_order<YA_TRAIT::VME_MEMORY,
                                        YA_TRAIT::VME_ORDER>::ans>::lp(mat);
  }
  // Loop over elements in any order and apply op with aux struct
  template<class mat_type, class itype>
  static inline void lp(mat_type &mat, itype &aux) {
    ya_no_order_loop2<op,_ya_nool_order<YA_TRAIT::VME_MEMORY,
                                        YA_TRAIT::VME_ORDER>::ans>::lp(mat,aux);
  }
  // Loop over elements and add the column major index if op is true
  template<class mat_type, class itype>
  static inline void find(mat_type &m, itype &ii) {
    ya_no_order_loop2<op,_ya_nool_order<YA_TRAIT::VME_MEMORY,
                                        YA_TRAIT::VME_ORDER>::ans>::find(m,ii);
  }
};

// Stride vector
template<class op>
struct ya_no_order_loop2<op,3> {
  template<class mat_type>
  static inline void lp(mat_type &mat) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor 1: stride) " << endl << flush;
    #endif
    const ya_sizet se=mat.stride_last();
    const ya_sizet si=mat.stride();
    for (ya_sizet i=mat.stride_start(); i<=se; i+=si)
      op::op(mat.stride_at(i));
  }
  template<class mat_type, class itype>
  static inline void lp(mat_type &mat, itype &aux) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 1: stride) " << endl << flush;
    #endif
    const ya_sizet se=mat.stride_last();
    const ya_sizet si=mat.stride();
    for (ya_sizet i=mat.stride_start(); i<=se; i+=si)
      op::op(mat.stride_at(i),aux);
  }
  template<class mat_type, class itype>
  static inline void find(mat_type &m, itype &ii) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 1: stride) " << endl << flush;
    #endif
    const ya_sizet se=m.stride_last();
    const ya_sizet si=m.stride();
    ya_sizet j=0;
    for (ya_sizet i=m.stride_start(); i<=se; i+=si) {
      if (op::op(m.stride_at(i)))
        ii.push_back(j);
      j++;
    }
  }
};

// Row major
template<class op>
struct ya_no_order_loop2<op,1> {
  template<class mat_type>
  static inline void lp(mat_type &mat) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor 2: row) " << endl << flush;
    #endif
    const ya_sizet ne=mat.numel();
    for (ya_sizet i=0; i<ne; i++)
      op::op(mat.row_at(i));
  }
  template<class mat_type, class itype>
  static inline void lp(mat_type &mat, itype &aux) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 2: row) " << endl << flush;
    #endif
    const ya_sizet ne=mat.numel();
    for (ya_sizet i=0; i<ne; i++)
      op::op(mat.row_at(i),aux);
  }
  template<class mat_type, class itype>
  static inline void find(mat_type &m, itype &ii) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 2: row) " << endl << flush;
    #endif
    const ya_sizet nE=m.numel();
    const ya_sizet nR=m.rows();
    const ya_sizet stride=m.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=j; i<nE; i=i+stride) {
        if (op::op(m.row_at(c)))
          ii.push_back(i);
        c++;
      }
  }
};

// 2D
template<class op>
struct ya_no_order_loop2<op,4> {
  template<class mat_type>
  static inline void lp(mat_type &mat) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor 2: row) " << endl << flush;
    #endif
    const ya_sizet nR=mat.rows();
    const ya_sizet nC=mat.cols();
    for (ya_sizet i=0; i<nR; i++)
      for (ya_sizet j=0; j<nC; j++)
        op::op(mat(i,j));
  }
  template<class mat_type, class itype>
  static inline void lp(mat_type &mat, itype &aux) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 2: row) " << endl << flush;
    #endif
    const ya_sizet nR=mat.rows();
    const ya_sizet nC=mat.cols();
    for (ya_sizet i=0; i<nR; i++)
      for (ya_sizet j=0; j<nC; j++)
        op::op(mat(i,j),aux);
  }
  template<class mat_type, class itype>
  static inline void find(mat_type &m, itype &ii) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 2: row) " << endl << flush;
    #endif
    const ya_sizet nR=m.rows();
    const ya_sizet nC=m.cols();
    ya_sizet c=0;
    for (ya_sizet i=0; i<nC; i++)
      for (ya_sizet j=0; j<nR; j++) {
        if (op::op(m(j,i)))
          ii.push_back(c);
        c++;
      }
  }
};

// Column major
template<class op,int order>
struct ya_no_order_loop2 {
  template<class mat_type>
  static inline void lp(mat_type &mat) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor 4: col) " << endl << flush;
    #endif
    const ya_sizet ne=mat.numel();
    for (ya_sizet i=0; i<ne; i++)
      op::op(mat(i));
  }
  template<class mat_type, class itype>
  static inline void lp(mat_type &mat, itype &aux) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 4: col) " << endl << flush;
    #endif
    const ya_sizet ne=mat.numel();
    for (ya_sizet i=0; i<ne; i++)
      op::op(mat(i),aux);
  }
  template<class mat_type, class itype>
  static inline void find(mat_type &m, itype &ii) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB ufor_a 4: col) " << endl << flush;
    #endif
    const ya_sizet ne=m.numel();
    for (ya_sizet i=0; i<ne; i++)
      if (op::op(m(i)))
        ii.push_back(i);
  }
};

// *************************************************************************
// Loop over elements in first matrix and same elements in second matrix
// *************************************************************************

template<int order1, int order2> struct _YA_TORDER;
// Set vector to be row major if other is row major, otherwise column major
template<int a> struct _YA_TORDER<2,a> { enum { ans=_YA_TAND<a,1>::ans }; };
// Set vector to be column major if other is a vector
template<> struct _YA_TORDER<1,3> { enum { ans=0 }; };
template<int a,int b> struct _YA_TORDER { enum { ans=a }; };

template<class op,int,int,int,int,int> struct ya_binary_loop2;

template<class op>
struct ya_binary_loop {
  // Loop over any elements known to be non-zero at compile time in mat2
  // and same in mat1
  template<class ya_type1, class ya_type2>
  static inline void nz(ya_type1 &mat1, ya_type2 &mat2) {
    lm(mat1,mat2);
  }
  // Loop over modifiable elements in mat1 and same in mat2 and apply op
  template<class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, ya_type2 &mat2) {
    ya_binary_loop2<op,YATRAIT1::VME_DIAGONAL,YATRAIT1::VME_SYMMETRIC,
      YATRAIT2::VME_SYMMETRIC,
      _YA_TORDER<YATRAIT1::VME_ORDER,YATRAIT2::VME_ORDER>::ans,
      _YA_TORDER<YATRAIT2::VME_ORDER,YATRAIT1::VME_ORDER>::ans >::lm(mat1,mat2);
  }
  // Loop over all elements in mat1 and mat2 testing op for each pair
  // - Returns true if op is satisfied for all pairs, otherwise false
  template<class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, ya_type2 &mat2) {
    return ya_binary_loop2<op,YATRAIT1::VME_DIAGONAL,YATRAIT1::VME_SYMMETRIC,
                           YATRAIT2::VME_SYMMETRIC,
     _YA_TORDER<YATRAIT1::VME_ORDER,YATRAIT2::VME_ORDER>::ans,
     _YA_TORDER<YATRAIT2::VME_ORDER,YATRAIT1::VME_ORDER>::ans >::lpr(mat1,mat2);
  }
};

// Target is diagonal
template<class op, int a, int b, int c, int d>
struct ya_binary_loop2<op,1,a,b,c,d> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 1: diagonal) " << endl << flush;
    #endif
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++)
      op::op(mat1(i,i),mat2(i,i));
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, ya_type2 &mat2) {
    return ya_binary_loop2<op,0,a,b,c,d>::lpr(mat1,mat2);
  }
};

// Symmetric/Symmetric
template<class op, int b, int c> 
struct ya_binary_loop2<op,0,1,1,b,c> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 2: symmetric/any) " << endl << flush;
    #endif
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++)
      for (ya_sizet j=0; j<=i; j++)
        op::op(mat1(i,j),mat2(i,j));
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++)
      for (ya_sizet j=0; j<=i; j++)
        if (!op::op(mat1(i,j),mat2(i,j)))
          return false;
    return true;
  }
};

// Symmetric/Not symmetric
template<class op, int b, int c> 
struct ya_binary_loop2<op,0,1,0,b,c> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    ya_binary_loop2<op,0,1,1,b,c>::lm(mat1,mat2);
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++) {
      for (ya_sizet j=0; j<i; j++) {
        if (!op::op(mat1(i,j),mat2(i,j)) || !op::op(mat1(i,j),mat2(j,i)))
          return false;
      }
      if (!op::op(mat1(i,i),mat2(i,i)))
        return false;
    }
    return true;
  }
};

// Target is not symmetric, source is
template<class op, int a, int b> 
struct ya_binary_loop2<op,0,0,1,a,b> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 3: any/symmetric) " << endl << flush;
    #endif
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++) {
      for (ya_sizet j=0; j<i; j++) {
        op::op(mat1(i,j),mat2(i,j));
        op::op(mat1(j,i),mat2(i,j));
      }
      op::op(mat1(i,i),mat2(i,i));
    }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat1.rows();
    for (ya_sizet i=0; i<nR; i++) {
      for (ya_sizet j=0; j<i; j++) {
        if (!op::op(mat1(i,j),mat2(i,j)))
          return false;
        if (!op::op(mat1(j,i),mat2(i,j)))
          return false;
      }
      if (!op::op(mat1(i,i),mat2(i,i)))
        return false;
    }
    return true;
  }
};

// Row major loop
template<class op> 
struct ya_binary_loop2<op,0,0,0,1,1> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 4: row/row) " << endl << flush;
    #endif
    const ya_sizet nE=mat1.numel();
    for (ya_sizet i=0; i<nE; i++)
      op::op(mat1.row_at(i),mat2.row_at(i));
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nE=mat1.numel();
    for (ya_sizet i=0; i<nE; i++)
      if (!op::op(mat1.row_at(i),mat2.row_at(i)))
        return false;
    return true;
  }
};

// Vector stride loop
template<class op>
struct ya_binary_loop2<op,0,0,0,3,3> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 5: stride/stride) " << endl << flush;
    #endif
    const ya_sizet se=mat1.stride_last();
    const ya_sizet si=mat1.stride();
    const ya_sizet si2=mat2.stride();
    ya_sizet i2=mat2.stride_start();
    
    for (ya_sizet i=mat1.stride_start(); i<=se; i+=si) {
      op::op(mat1.stride_at(i),mat2.stride_at(i2));
      i2+=si2;
    }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet se=mat1.stride_last();
    const ya_sizet si=mat1.stride();
    const ya_sizet si2=mat2.stride();
    ya_sizet i2=mat2.stride_start();
    
    for (ya_sizet i=mat1.stride_start(); i<=se; i+=si) {
      if (!op::op(mat1.stride_at(i),mat2.stride_at(i2)))
        return false;
      i2+=si2;
    }
    return true;
  }
};

// 2D loop
template<class op>
struct ya_binary_loop2<op,0,0,0,4,4> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 6: 2D/2D) " << endl << flush;
    #endif
    const ya_sizet nC=mat1.cols();
    const ya_sizet nR=mat1.rows();
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++)
        op::op(mat1(j,i),mat2(j,i));
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nC=mat1.cols();
    const ya_sizet nR=mat1.rows();
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++)
        if (!op::op(mat1(j,i),mat2(j,i)))
          return false;
    return true;
  }
};

// 2D loop / vector
template<class op>
struct ya_binary_loop2<op,0,0,0,4,2> : ya_binary_loop2<op,0,0,0,4,4> { };

// vector / 2D loop
template<class op>
struct ya_binary_loop2<op,0,0,0,2,4> : ya_binary_loop2<op,0,0,0,4,4> { };

// Row major/Column major
template<class op> 
struct ya_binary_loop2<op,0,0,0,1,0> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 7: row/column) " << endl << flush;
    #endif
    const ya_sizet nE=mat1.numel();
    const ya_sizet nR=mat2.rows();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=j; i<nE; i+=nR) {
        op::op(mat1.row_at(c),mat2(i));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nE=mat1.numel();
    const ya_sizet nR=mat2.rows();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=j; i<nE; i+=nR) {
        if (!op::op(mat1.row_at(c),mat2(i)))
          return false;
        c++;
      }
    return true;
  }
};

// Row major/2D
template<class op> 
struct ya_binary_loop2<op,0,0,0,1,4> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 8: row/2D) " << endl << flush;
    #endif
    const ya_sizet nR=mat2.rows();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++) {
        op::op(mat1.row_at(c),mat2(j,i));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat2.rows();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++) {
        if (!op::op(mat1.row_at(c),mat2(j,i)))
          return false;
        c++;
      }
    return true;
  }
};

// 2D/Row major
template<class op> 
struct ya_binary_loop2<op,0,0,0,4,1> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 9: 2D/row) " << endl << flush;
    #endif
    const ya_sizet nR=mat1.rows();
    const ya_sizet nC=mat1.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++) {
        op::op(mat1(j,i),mat2.row_at(c));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat1.rows();
    const ya_sizet nC=mat1.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nR; j++)
      for (ya_sizet i=0; i<nC; i++) {
        if (!op::op(mat1(j,i),mat2.row_at(c)))
          return false;
        c++;
      }
    return true;
  }
};

// Column major/Row major
template<class op> 
struct ya_binary_loop2<op,0,0,0,0,1> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 10: col/row) " << endl << flush;
    #endif
    const ya_sizet nE=mat1.numel();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nC; j++)
      for (ya_sizet i=j; i<nE; i+=nC) {
        op::op(mat1(c),mat2.row_at(i));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nE=mat1.numel();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet j=0; j<nC; j++)
      for (ya_sizet i=j; i<nE; i+=nC) {
        if (!op::op(mat1(c),mat2.row_at(i)))
          return false;
        c++;
      }
    return true;
  }
};

// Column major/2D
template<class op> 
struct ya_binary_loop2<op,0,0,0,0,4> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 11: col/2D) " << endl << flush;
    #endif
    const ya_sizet nR=mat2.rows();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet i=0; i<nC; i++)
      for (ya_sizet j=0; j<nR; j++) {
        op::op(mat1(c),mat2(j,i));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat2.rows();
    const ya_sizet nC=mat2.cols();
    ya_sizet c=0;
    
    for (ya_sizet i=0; i<nC; i++)
      for (ya_sizet j=0; j<nR; j++) {
        if (!op::op(mat1(c),mat2(j,i)))
          return false;
        c++;
      }
    return true;
  }
};

// 2D/Column major
template<class op> 
struct ya_binary_loop2<op,0,0,0,4,0> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 12: 2D/col) " << endl << flush;
    #endif
    const ya_sizet nR=mat1.rows();
    const ya_sizet nC=mat1.cols();
    ya_sizet c=0;
    
    for (ya_sizet i=0; i<nC; i++)
      for (ya_sizet j=0; j<nR; j++) {
        op::op(mat1(j,i),mat2.row(c));
        c++;
      }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nR=mat1.rows();
    const ya_sizet nC=mat1.cols();
    ya_sizet c=0;
    
    for (ya_sizet i=0; i<nC; i++)
      for (ya_sizet j=0; j<nR; j++) {
        if (!op::op(mat1(j,i),mat2.row(c)))
          return false;
        c++;
      }
    return true;
  }
};

// Column major/vector stride loop
template<class op>
struct ya_binary_loop2<op,0,0,0,0,3> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 13: col/stride) " << endl << flush;
    #endif
    const ya_sizet si=mat2.stride();
    ya_sizet i2=mat2.stride_start();
    ya_sizet nE=mat1.numel();    
    for (ya_sizet i=0; i<nE; i++) {
      op::op(mat1(i),mat2.stride_at(i2));
      i2+=si;
    }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet si=mat2.stride();
    ya_sizet i2=mat2.stride_start();
    ya_sizet nE=mat1.numel();
    for (ya_sizet i=0; i<nE; i++) {
      if (!op::op(mat1(i),mat2.stride_at(i2)))
        return false;
      i2+=si;
    }
    return true;
  }
};

// Vector stride/Column major loop
template<class op>
struct ya_binary_loop2<op,0,0,0,3,0> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 14: stride/col) " << endl << flush;
    #endif
    const ya_sizet si=mat1.stride();
    ya_sizet i2=mat1.stride_start();
    ya_sizet nE=mat2.numel();    
    for (ya_sizet i=0; i<nE; i++) {
      op::op(mat1.stride_at(i2),mat2(i));
      i2+=si;
    }
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet si=mat1.stride();
    ya_sizet i2=mat1.stride_start();
    ya_sizet nE=mat2.numel();
    for (ya_sizet i=0; i<nE; i++) {
      if (!op::op(mat1.stride_at(i2),mat2(i)))
        return false;
      i2+=si;
    }
    return true;
  }
};

// Use column major for both
template<class op, int a, int b> 
struct ya_binary_loop2<op,0,0,0,a,b> {
  template <class ya_type1, class ya_type2>
  static inline void lm(ya_type1 &mat1, const ya_type2 &mat2) {
    #ifdef YA_MEMDBG
    cout << endl << " (MDB bfor_m 15: col/col) " << endl << flush;
    #endif
    const ya_sizet nE=mat1.numel();
    for (ya_sizet i=0; i<nE; i++)
      op::op(mat1(i),mat2(i));
  }
  template <class ya_type1, class ya_type2>
  static inline bool lpr(ya_type1 &mat1, const ya_type2 &mat2) {
    const ya_sizet nE=mat1.numel();
    for (ya_sizet i=0; i<nE; i++)
      if (!op::op(mat1(i),mat2(i)))
        return false;
    return true;
  }
};

}
#endif
