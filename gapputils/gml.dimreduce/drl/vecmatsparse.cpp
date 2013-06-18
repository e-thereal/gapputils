/***************************************************************************
                               vecmatsparse.cpp
                             -------------------
                               W. Michael Brown

  Sparse Vector/Matrix
  Insert/Access in log(N)

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon May 24 2003
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATSPARSE_CPP
#define VECMATSPARSE_CPP
 
using namespace yala;

template <class eltype, class mat_type>
VMSparseReturn<eltype,mat_type>::VMSparseReturn(mat_type *par,
                                                  const ya_sizet i) :
    parent(par), index(i) {}

template <class eltype, class mat_type>
VMSparseReturn<eltype,mat_type>::operator eltype() const {
  typename std::map<ya_sizet,eltype>::const_iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end())
    return parent->zerovalue();
  else
    return m->second;
}

template <class eltype, class mat_type>
void VMSparseReturn<eltype,mat_type>::operator=(const eltype two) const {
  if (two==parent->zerovalue())
     parent->zero(index);
  else
     parent->sparse_array[index]=two;
}

template <class eltype, class mat_type>
void VMSparseReturn<eltype,mat_type>::operator+=(const eltype two) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second+=two;
  if (m->second==parent->zerovalue())
     parent->zero(index);
}

template <class eltype, class mat_type>
void VMSparseReturn<eltype,mat_type>::operator-=(const eltype two) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second-=two;
  if (m->second==parent->zerovalue())
     parent->zero(index);
}

template <class eltype, class mat_type>
void VMSparseReturn<eltype,mat_type>::operator*=(const eltype two) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second*=two;
  if (m->second==parent->zerovalue())
     parent->zero(index);
}

template <class eltype, class mat_type>
void VMSparseReturn<eltype,mat_type>::operator/=(const eltype two) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second/=two;
  if (m->second==parent->zerovalue())
     parent->zero(index);
}

template <class eltype, class mat_type>
eltype VMSparseReturn<eltype,mat_type>::operator++(int) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  eltype t=m->second;
  m->second++;
  if (m->second==parent->zerovalue())
     parent->zero(index);
  return t;
}

template <class eltype, class mat_type>
eltype VMSparseReturn<eltype,mat_type>::operator--(int) const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  eltype t=m->second;
  m->second--;
  if (m->second==parent->zerovalue())
     parent->zero(index);
  return t;
}

template <class eltype, class mat_type>
VMSparseReturn<eltype,mat_type>&
VMSparseReturn<eltype,mat_type>::operator++() const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second++;
  if (m->second==parent->zerovalue())
     parent->zero(index);
  return *this;
}

template <class eltype, class mat_type>
VMSparseReturn<eltype,mat_type>&
VMSparseReturn<eltype,mat_type>::operator--() const {
  typename std::map<ya_sizet,eltype>::iterator m;
  m=parent->sparse_array.find(index);
  if (m==parent->sparse_array.end()) {
    std::pair<typename std::map<ya_sizet,eltype>::iterator,bool> ret;
    ret=parent->sparse_array.insert(std::pair<ya_sizet,eltype>(index,
                                   parent->zerovalue()));
    m=ret.first;
  }
  m->second--;
  if (m->second==parent->zerovalue())
     parent->zero(index);
  return *this;
}

template<class eltype, class mat_type>
std::ostream & yala::operator<<(std::ostream &out,
                           const VMSparseReturn<eltype,mat_type> &t) {
   typename std::map<ya_sizet,eltype>::const_iterator m;
   m=t.parent->sparse_array.find(t.index);
   if (m==t.parent->sparse_array.end())
     out << t.parent->zerovalue();
  else
    out << m->second;
  return out;
}

// Initialize matrix and set the zero value for 'empty cells'
template<class eltype, class zerotype>
VM_Sparse::VecMatSparse() : ncolumns(1), nrows(0), nlength(0) {
}

/// Copy constructor
template<class eltype, class zerotype>
VM_Sparse::VecMatSparse(const VecMatSparse &v) {
  (*this)=v;
}

// Initialize vector and set the zero value for 'empty cells'
template<class eltype, class zerotype>
VM_Sparse::VecMatSparse(const ya_sizet len) : ncolumns(1), nrows(len),
        nlength(len) {
}

// Initialize matrix and set zero value for 'empty cells'
template<class eltype, class zerotype>
VM_Sparse::VecMatSparse(const ya_sizet rows, const ya_sizet columns) : 
              nrows(rows), ncolumns(columns), nlength(rows*columns) {
}

template<class eltype, class zerotype>
template<class ya_type2>
VM_Sparse::VecMatSparse(const ya_type2 &v) {
  (*this)=v;
}

// Clear any old vector and set up a new one with 'length' elements
template<class eltype, class zerotype>
void VM_Sparse::setup(ya_sizet lengthi) {
  sparse_array.clear();
  ncolumns=1;
  nrows=lengthi;
  nlength=lengthi;
}

// Clear any old matrix and set up a new one
template<class eltype, class zerotype>
void VM_Sparse::setup(ya_sizet rows, ya_sizet columns) {
  sparse_array.clear();
  ncolumns=columns;
  nrows=rows;
  nlength=rows*columns;
}

// Clear the matrix
template<class eltype, class zerotype>
void VM_Sparse::clear() {
  sparse_array.clear();
  ncolumns=1;
  nrows=0;
  nlength=0;
}

// Test if the matrix is empty
template<class eltype, class zerotype>
bool VM_Sparse::empty() {
  return sparse_array.empty();
}

// Set each element in the matrix to zero
template<class eltype, class zerotype>
void VM_Sparse::zeromatrix() {
  sparse_array.clear();
}

// Set the value of a cell to 'zero' using 1D access
template<class eltype, class zerotype>
void VM_Sparse::zero(const ya_sizet i) {
  Miter m;
  m=sparse_array.find(i);
  if (m!=sparse_array.end())
    sparse_array.erase(m);
}

// Set the value of a cell to 'zero' using 2D access
template<class eltype, class zerotype>
void VM_Sparse::zero(const ya_sizet row, const ya_sizet col) {
  zero(col*rows()+row);
}

// Write out a matrix
template<class eltype, class zerotype>
void VM_Sparse::write_nonzero(std::ostream &out) const {
  for (MiterC m=sparse_array.begin(); m!=sparse_array.end(); m++)
    out << m->first << " " << m->second << std::endl;
  out << "END\n";
}

// Read in a matrix
template<class eltype, class zerotype>
int VM_Sparse::read_nonzero(std::istream &in) {
  std::string token;
  eltype value;
  while (1) {
    in >> token >> value;
    
    if (in.eof() || !in) {
      if (token=="END")
        break;
      ya_addwarn(301,19,"YA_Sparse","Improperly formatted YALA++ file.");
      clear();
      return 301;
    }
    (*this)(atoi(token.c_str()))=value;
  }
  return 0;
}

template<class eltype, class zerotype>
void VM_Sparse::copymatrix(const VM_Sparse &v) {
  this->sparse_array=v.sparse_array;
}

template<class eltype, class zerotype>
bool VM_Sparse::comparematrix(const VM_Sparse &v) const {
  return (this->sparse_array==v.sparse_array);
}

#endif
