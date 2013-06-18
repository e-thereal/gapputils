/***************************************************************************
                                ya_row_iter.h
                             -------------------
                               W. Michael Brown

  General Row Major Iterator

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Mon Feb 12 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_ROW_ITER_H
#define YA_ROW_ITER_H

YALANAMESPACE

/// General Row Major Iterator
template<class eltype, class mat_type>
class YA_RowIter {
 public:
  typedef eltype value_type;
  typedef std::random_access_iterator_tag iterator_category;
  typedef typename mat_type::ref reference;
  typedef eltype* pointer;
  typedef ya_sizet difference_type;
 
  /// Create the iterator (Equal to mat.begin())
  YA_RowIter(mat_type *m) : mat(m), index(0) {}
  /// Create the iterator pointing to the ith element
  YA_RowIter(mat_type *m, const ya_sizet i) : mat(m), index(i) {}

  /// Set the iterator to the beginning of the matrix
  inline void begin() { index=0; }
  /// True if iterator is not at the end of the matrix
  inline bool not_end() const { return index<mat->numel(); }

  inline reference operator()() const { return mat->row_at(index); }
  inline reference operator*() const { return mat->row_at(index); }

  inline reference operator[](const ya_sizet i) const
    { return mat->row_at(index+i); }
  
  inline YA_RowIter& operator++() { index++; return *this; }
  inline YA_RowIter& operator--() { index--; return *this; }
  inline YA_RowIter operator++(ya_sizet) 
    { YA_RowIter t(*this); index++; return t; }
  inline YA_RowIter operator--(ya_sizet) 
    { YA_RowIter t(*this); index--; return t; }

  inline YA_RowIter operator+(const ya_sizet adv) const
   { return YA_RowIter(mat,index+adv); }
  inline YA_RowIter operator-(const ya_sizet adv) const
   { return YA_RowIter(mat,index-adv); }
  /// Set the iterator to point to a given matrix index
  inline const YA_RowIter& operator=(const ya_sizet i) 
    { index=i; return *this; }

  inline ya_sizet operator-(const YA_RowIter &r) const
    { return index-r.index; }
  inline bool operator<(const YA_RowIter &r) const
    { return index<r.index; }

  inline bool identical(const YA_RowIter &t) const
    { YA_DEBUG_ERROR(mat==t.mat,
                     "Can't compare iterators for different objects.");
      return (index==t.index); }
  inline bool operator==(const YA_RowIter &t) const
    { return identical(t); }

  inline bool operator!=(const YA_RowIter &t) const
    { return !identical(t); }
   
 protected:
  mat_type * mat;
  ya_sizet index;
};

/// General Const Row Major Iterator
template<class eltype, class mat_type>
class YA_ConstRowIter : public YA_RowIter<eltype,mat_type> {
 public:
  typedef eltype value_type;
  typedef std::random_access_iterator_tag iterator_category;
  typedef typename mat_type::cref reference;
  typedef const eltype* pointer;
  typedef ya_sizet difference_type;
  
  /// Create the iterator (Equals mat.begin())
  YA_ConstRowIter(mat_type *m) : YA_RowIterT(m) { }
  /// Create the iterator pointing to the ith element
  YA_ConstRowIter(mat_type *m, const ya_sizet i) : YA_RowIterT(m,i) {}
  YA_ConstRowIter(const YA_RowIter<eltype,mat_type> &two) : YA_RowIterT(two){}

  inline reference operator()() const { return this->mat->row_at(this->index); }
  inline reference operator*() const { return this->mat->row_at(this->index); }

  inline reference operator[](const ya_sizet i) const
    { return this->mat->row_at(this->index+i); }

  inline void operator=(const typename YA_RowIterT &two)
    { this->mat=two.mat; this->index=two.index; }
};

}
#endif
