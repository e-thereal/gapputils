/***************************************************************************
                                vecmatsym.cpp
                             -------------------
                               W. Michael Brown

  Symmetric Matrix Class
  Insert/Access in constant time

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

#ifndef VECMATSYM_CPP
#define VECMATSYM_CPP
 
using namespace yala;

/// Initialize vector
template<class eltype>
VM_SymMat::VecMatSym() : array(NULL), nrows(0), nlength(0) {
}

/// Copy constructor
template<class eltype>
VM_SymMat::VecMatSym(const VecMatSym &v) : array(NULL), nrows(0) {
  (*this)=v;
}

// Initialize matrix
template<class eltype>
VM_SymMat::VecMatSym(const ya_sizet rowi, const ya_sizet columni) : 
            nrows(rowi), nlength((nrows*nrows+nrows)/2) {
  YA_DEBUG_ERROR(rowi==columni,
             "Attempt to setup symmetric matrix with non-square dimensions");

  array=new eltype[nlength];
}

template<class eltype>
template<class ya_type2>
VM_SymMat::VecMatSym(const ya_type2 &v) : array(NULL), nrows(0) {
  (*this)=v;
}

// Clear any old matrix and set up a new one
template<class eltype>
void VM_SymMat::setup(const ya_sizet rowi,const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi==columni,
             "Attempt to setup symmetric matrix with non-square dimensions");

  if (rowi!=rows()) {
    if (array!=NULL)
      delete[] array;
    nrows=rowi;
    nlength=(nrows*nrows+nrows)/2;
    array=new eltype[nlength];
  }
}

// Clear any old matrix and set up a new one
template<class eltype>
int VM_SymMat::safe_setup(const ya_sizet rowi,const ya_sizet columni) {
  YA_DEBUG_ERROR(rowi==columni,
             "Attempt to setup symmetric matrix with non-square dimensions");

  if (rowi!=rows()) {
    if (array!=NULL)
      delete[] array;
    nrows=rowi;
    nlength=(nrows*nrows+nrows)/2;
    return reserve(numel());
  }
  return 0;
}

// Clear any old matrix and set up a new one
template<class eltype>
void VM_SymMat::setup(const ya_sizet length) {
  std::cerr << "DEVELOP ERROR: Cannot use 1D setup on VM_SymMat.\n";
  exit(1);
}

// Clear the matrix
template<class eltype>
void VM_SymMat::clear() {
  if (array!=NULL)
    delete[] array;
  array=NULL;
  nrows=0;
  nlength=0;
}

// Access matrix element
template<class eltype>
typename VM_SymMat::cref VM_SymMat::at2(const ya_sizet row,
                                                const ya_sizet col) const {
  if (row>col)
    return (this->array)[row*(row+1)/2+col];
  else
    return (this->array)[col*(col+1)/2+row];
}

// Access matrix element
template<class eltype>
typename VM_SymMat::ref VM_SymMat::at2(const ya_sizet row, const ya_sizet col) {
  if (row>col)
    return (this->array)[row*(row+1)/2+col];
  else
    return (this->array)[col*(col+1)/2+row];
}

template<class eltype>
void VM_SymMat::destruct() {
  if (array!=NULL)
    delete[] array;
}

// Reserve memory for vecmat
template<class eltype>
int VM_SymMat::reserve(const ya_sizet length) {
  try {
    array=new eltype[length];
  }
  catch(std::bad_alloc errora) {
    std::ostringstream err_str;
    err_str << "Memory allocation error. The vector/matrix requested "
            << "requires " << length*sizeof(eltype)/(1024*1024)
            << "MB\nSYSTEM: " << errora.what();
    ya_addwarn(300,19,"Grid",err_str.str());
    return 300;
  }
  return 0;
}

// Access vector element
template<class eltype>
typename VM_SymMat::cref VM_SymMat::at(const ya_sizet i) const {
  return array[i];
}

// Access vector element
template<class eltype>
typename VM_SymMat::ref VM_SymMat::at(const ya_sizet i) {
  return array[i];
}

#endif
