/***************************************************************************
                               ya_fun_wrap.cpp
                             -------------------
                               W. Michael Brown

  Functions that return wrappers
 
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

#ifndef YA_FUN_WRAP_S_H
#define YA_FUN_WRAP_S_H

YALANAMESPACE

/// Generate a column vector wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,eltype *,2,-1,1> 
ya_wrap(eltype *array,const ya_sizet rows) {
  return YA_Wrap<eltype,eltype *,2,-1,1>(array,rows);
}

/// Generate a column vector wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,const eltype *,2,-1,1> 
ya_wrap(const eltype *array, const ya_sizet rows) {
  return YA_Wrap<eltype,const eltype *>(array,rows);
}

/// Generate a column-major wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,eltype *> 
ya_wrap(eltype *array,const ya_sizet rows,const ya_sizet cols) {
  return YA_Wrap<eltype,eltype *>(array,rows,cols);
}

/// Generate a column-major wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,const eltype *> 
ya_wrap(const eltype *array, const ya_sizet rows, const ya_sizet cols) {
  return YA_Wrap<eltype,const eltype *>(array,rows,cols);
}

/// Generate a row-major wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,eltype *,1> 
ya_wrap_rm(eltype *array,const ya_sizet rows,const ya_sizet cols) {
  return YA_Wrap<eltype,eltype *,1>(array,rows,cols);
}

/// Generate a row-major wrapper around a C pointer to an array
template<class eltype>
inline const YA_Wrap<eltype,const eltype *,1> 
ya_wrap_rm(const eltype *array, const ya_sizet rows, const ya_sizet cols) {
  return YA_Wrap<eltype,const eltype *,1>(array,rows,cols);
}

/// Generate a vector wrapper around a C pointer array with non-unit stride
template<class eltype>
inline const YA_Wrap<eltype,eltype *,3> 
ya_wrap(eltype *array, const ya_sizet rows, const ya_sizet cols,
        const ya_sizet stride) {
  return YA_Wrap<eltype,eltype *,3>(array,rows,cols,stride);
}

/// Generate a vector wrapper around a C pointer array with non-unit stride
template<class eltype>
inline const YA_Wrap<eltype,const eltype *,3> 
ya_wrap(const eltype *array, const ya_sizet rows, const ya_sizet cols,
        const ya_sizet stride) {
  return YA_Wrap<eltype,const eltype *>(array,rows,cols,stride);
}

/// Generate a fixed-size column-major wrapper around a C pointer to an array
template<class eltype,int nrows,int ncols>
inline const YA_Wrap<eltype,eltype *,0,nrows,ncols,eltype&,const eltype&,1> 
ya_wrap_(eltype *array) {
  return YA_Wrap<eltype,eltype *,0,nrows,
                 ncols,eltype&,const eltype&,1>(array);
}

/// Generate a fixed-size column-major wrapper around a C pointer to an array
template<class eltype,int nrows,int ncols>
inline const YA_Wrap<eltype,const eltype *,0,nrows,ncols,
                     const eltype&,const eltype&,1> 
ya_wrap_(const eltype *array) {
  return YA_Wrap<eltype,const eltype *,0,nrows,
                 ncols,const eltype&,const eltype&,1>(array);
}

/// Generate a column vector wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,std::vector<eltype> &,2,-1,1>
ya_wrap(std::vector<eltype> &array,const ya_sizet rows) {
  return YA_Wrap<eltype,std::vector<eltype> &,2,-1,1>(array,rows);
}

/// Generate a column vector wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,const std::vector<eltype> &,2,-1,1>
ya_wrap(const std::vector<eltype> &array, const ya_sizet rows) {
  return YA_Wrap<eltype,const std::vector<eltype> &>(array,rows);
}

/// Generate a column-major wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,std::vector<eltype> &>
ya_wrap(std::vector<eltype> &array,const ya_sizet rows,const ya_sizet cols) {
  return YA_Wrap<eltype,std::vector<eltype> &>(array,rows,cols);
}

/// Generate a column-major wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,const std::vector<eltype> &>
ya_wrap(const std::vector<eltype> &array, const ya_sizet rows,
        const ya_sizet cols) {
  return YA_Wrap<eltype,const std::vector<eltype> &>(array,rows,cols);
}

/// Generate a row-major wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,std::vector<eltype> &,1>
ya_wrap_rm(std::vector<eltype> &array,const ya_sizet rows,const ya_sizet cols) {
  return YA_Wrap<eltype,std::vector<eltype> &,1>(array,rows,cols);
}

/// Generate a row-major wrapper around STL vector
template<class eltype>
inline const YA_Wrap<eltype,const std::vector<eltype> &,1>
ya_wrap_rm(const std::vector<eltype> &array, const ya_sizet rows,
           const ya_sizet cols) {
  return YA_Wrap<eltype,const std::vector<eltype> &,1>(array,rows,cols);
}

/// Generate a vector wrapper around a STL vector with non-unit stride
template<class eltype>
inline const YA_Wrap<eltype,std::vector<eltype> &,3>
ya_wrap(std::vector<eltype> &array, const ya_sizet rows, const ya_sizet cols,
        const ya_sizet stride) {
  return YA_Wrap<eltype,std::vector<eltype> &,3>(array,rows,cols,stride);
}

/// Generate a vector wrapper around a STL vector with non-unit stride
template<class eltype>
inline const YA_Wrap<eltype,const std::vector<eltype> &,3>
ya_wrap(const std::vector<eltype> &array, const ya_sizet rows,
        const ya_sizet cols, const ya_sizet stride) {
  return YA_Wrap<eltype,const std::vector<eltype> &>(array,rows,cols,stride);
}

/// Generate a fixed-size column-major wrapper around a STL vector
template<class eltype,int nrows,int ncols>
inline const YA_Wrap<eltype,std::vector<eltype> &,0,nrows,
                     ncols,eltype&,const eltype&,1> 
ya_wrap_(std::vector<eltype> &array) {
  return YA_Wrap<eltype,std::vector<eltype> &,0,nrows,
                 ncols,eltype&,const eltype&,1>(array);
}

/// Generate a fixed-size column-major wrapper around a STL vector
template<class eltype,int nrows,int ncols>
inline const YA_Wrap<eltype,const std::vector<eltype> &,0,nrows,ncols,
                     const eltype&,const eltype&,1> 
ya_wrap_(const std::vector<eltype> &array) {
  return YA_Wrap<eltype,const std::vector<eltype> &,0,nrows,
                 ncols,const eltype&,const eltype&,1>(array);
}

/// Generate a fixed-size column-major wrapper around some vector type
template<class eltype,class mat_type>
inline const YA_Wrap<eltype,mat_type> 
ya_wrap_u(mat_type array, const ya_sizet nrows, const ya_sizet ncols) {
  return YA_Wrap<eltype,mat_type>(array,nrows,ncols);
}

/// Generate a fixed-size row-major wrapper around some vector type
template<class eltype,class mat_type>
inline const YA_Wrap<eltype,mat_type,1> 
ya_wrap_rmu(mat_type array, const ya_sizet nrows, const ya_sizet ncols) {
  return YA_Wrap<eltype,mat_type,1>(array,nrows,ncols);
}

}
#endif
