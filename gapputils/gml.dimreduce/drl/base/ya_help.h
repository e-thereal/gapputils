/***************************************************************************
                                  ya_help.h
                             -------------------
                               W. Michael Brown

  Miscellaneous helper functions

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Wed Sep 19 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_HELP_H
#define YA_HELP_H

YALANAMESPACE

/// Templated function for converting strings to a given type
/** Requires that operator >> be overloaded for the type.
  * Generates error 12 L 9 if there is an error converting the string **/
template <class t_type>
inline int ya_ascii_to(const std::string &value, t_type &newtype) {
  std::istringstream in(value);
   in >> newtype;
   if (!in) {
     ya_addwarn(12,9,"YA_Help","Error converting from string: "+value);
     return 12;
  }
  return 0;
}

// Return the 2D index corresponding to a 1D index
template<class eltype,class mat_type>
inline void oneDtwoD(const YA_BaseT &mat, const ya_sizet oned, 
                     ya_sizet &row, ya_sizet &col)
  { div_t index=div(oned,mat.rows()); row=index.rem; col=index.quot; }  

// Return the 2D index corresponding to a rowmajor 1D index
template<class eltype,class mat_type>
inline void rowoneDtwoD(const YA_BaseT &mat, const ya_sizet oned, 
                        ya_sizet &row, ya_sizet &col)
  { div_t index=div(oned,mat.cols()); col=index.rem; row=index.quot; }  

// Return the 1D index corresponding to a rowmajor 1D index
template<class eltype,class mat_type>
inline ya_sizet rowoneDoneD(const YA_BaseT &mat, const ya_sizet oned)
  { ya_sizet iR,iC; rowoneDtwoD(mat,oned,iR,iC); return iC*mat.rows()+iR; }

// Return the 1D rowmajor index corresponding to a 1D index
template<class eltype,class mat_type>
inline ya_sizet oneDoneD(const YA_BaseT &mat, const ya_sizet oned)
  { ya_sizet iR,iC; oneDtwoD(mat,oned,iR,iC); return iR*mat.cols()+iC; }

template<class ref, class mat_type, int order, int vec> struct _ya_order_help;

// Vector helper functions
// - These functions allow, in a very cryptic manner, default functions
//   for vectors to efficiently use at/row_at, front/back, begin/rowbegin
//   because they are both column-major/row-major
template<class ref, class mat_type> 
struct _ya_order_help<ref,mat_type,2,1> {
  static inline ref at2(mat_type &mat, const ya_sizet row, const ya_sizet col)
    { return mat.row_at(row+col); }
  static inline ref rowmajor_at(mat_type &mat, const ya_sizet i)
    { return mat.at(i); }
  static inline ref at(mat_type &mat, const ya_sizet i)
    { return mat.rowmajor_at(i); }
  static inline ref front(mat_type &mat)
    { return mat.at(0); }
  static inline ref back(mat_type &mat)
    { return mat.at(mat.numel()-1); }
};

// Two-D helper functions
template<class ref, class mat_type> 
struct _ya_order_help<ref,mat_type,4,0> {
  static inline ref at(mat_type &mat, const ya_sizet i)
    { ya_sizet row,col; oneDtwoD(mat,i,row,col); return mat.at2(row,col); }
  static inline ref rowmajor_at(mat_type &mat, const ya_sizet i)
    { ya_sizet row,col; rowoneDtwoD(mat,i,row,col); return mat.at2(row,col); }
  static inline ref front(mat_type &mat)
    { return mat.at2(0,0); }
  static inline ref back(mat_type &mat)
    { return mat.at2(mat.rows()-1,mat.cols()-1); }
};

// Row-major helper functions
template<class ref, class mat_type> 
struct _ya_order_help<ref,mat_type,1,0> {
  static inline ref at2(mat_type &mat, const ya_sizet row, const ya_sizet col)
    { return mat.rowmajor_at(row*mat.cols()+col); }
  static inline ref at(mat_type &mat, const ya_sizet i)
    { return mat.rowmajor_at(oneDoneD(mat,i)); }
  static inline ref front(mat_type &mat)
    { return mat.rowmajor_at(0); }
  static inline ref back(mat_type &mat)
    { return mat.rowmajor_at(mat.numel()-1); }
};

// Column-major helper functions
template<class ref, class mat_type, int order, int vec> 
struct _ya_order_help {
  static inline ref at2(mat_type &mat, const ya_sizet row, const ya_sizet col)
    { return mat.at(col*mat.rows()+row); }
  static inline ref rowmajor_at(mat_type &mat, const ya_sizet i)
    { return mat.at(rowoneDoneD(mat,i)); }
  static inline ref front(mat_type &mat)
    { return mat.at(0); }
  static inline ref back(mat_type &mat)
    { return mat.at(mat.numel()-1); }
};

template<class mat_type, int vec> struct _ya_vec_help;

// Vector helper functions
// - These functions allow, in a very cryptic manner, default functions
//   for vectors to efficiently use begin/rowbegin
//   because they are both column-major/row-major
template<class mat_type> 
struct _ya_vec_help<mat_type,1> {
  static inline typename YA_IterT begin(mat_type &mat)
    { return typename YA_IterT(&mat); }
  static inline typename YA_ConstIterT cbegin(mat_type &mat)
    { return typename YA_ConstIterT(&mat); }
  static inline typename YA_IterT rowbegin(mat_type &mat)
    { return mat.begin(); }
  static inline typename YA_ConstIterT crowbegin(mat_type &mat)
    { return mat.begin(); }
};

// Matrix helper functions
template<class mat_type, int vec> 
struct _ya_vec_help {
  static inline typename YA_IterT begin(mat_type &mat)
    { return typename YA_IterT(&mat); }
  static inline typename YA_ConstIterT cbegin(mat_type &mat)
    { return typename YA_ConstIterT(&mat); }
  static inline typename YA_RowIterT rowbegin(mat_type &mat)
    { return typename YA_RowIterT(&mat); }
  static inline typename YA_ConstRowIterT crowbegin(mat_type &mat)
    { return typename YA_ConstRowIterT(&mat); }
};

}
#endif
