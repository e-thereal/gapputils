/***************************************************************************
                                vecmatconst.h
                             -------------------
                               W. Michael Brown

  Constant Matrices

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Sat Feb 10 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATCONST_H
#define VECMATCONST_H

YALANAMESPACE

///  Constant column vector containing a single number
template <class eltype, int cnum>
class VecMatConstV : public YA_Common<eltype, VM_ConstV> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatConstV>::cref cref;
  typedef typename YA_Traits<VecMatConstV>::citer citer;

  /// Initialize a column vector with a given length
  VecMatConstV(const ya_sizet len) : nrows(len) {}
    
  /// Get the length of a vector
  inline ya_sizet numel() const { return nrows; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return 1; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return static_cast<eltype>(cnum);
  }
  using YA_Common<eltype,VM_ConstV>::operator();

 protected:
  VecMatConstV();
  const ya_sizet nrows;  // size of matrix
};

///  Constant matrix containing a single number
template <class eltype, int cnum>
class VecMatConst : public YA_Common<eltype, VM_Const> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatConst>::cref cref;
  typedef typename YA_Traits<VecMatConst>::citer citer;

  /// Initialize a column vector with a given length
  VecMatConst(const ya_sizet len) : nrows(len), ncolumns(1) {}
  /// Initialize a matrix with a given length
  VecMatConst(const ya_sizet rows, const ya_sizet columns) : 
    nrows(rows), ncolumns(columns) {}
    
  /// Get the length of a vector
  inline ya_sizet numel() const { return nrows*ncolumns; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return static_cast<eltype>(cnum);
  }
  using YA_Common<eltype,VM_Const>::operator();

 protected:
  VecMatConst();
  const ya_sizet nrows;  
  const ya_sizet ncolumns;
};

///  Constant identity matrix
template <class eltype>
class VecMatEye : public YA_Common<eltype, VM_Eye> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatEye>::cref cref;
  typedef typename YA_Traits<VecMatEye>::citer citer;

  /// Initialize with given number of rows(columns)
  VecMatEye(const ya_sizet rowsi) : nrows(rowsi) {}
  /// Initialize with given number of rows(columns)
  VecMatEye(const ya_sizet rowsi, const ya_sizet colsi) : nrows(rowsi) {
    YA_DEBUG_ERROR(rowsi==colsi,
                   "Attempted to create non-square identity matrix");
  }

  /// Get the length of a vector
  inline ya_sizet numel() const { return nrows*nrows; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return nrows; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    div_t index=div(i,rows());
    return at2(index.rem,index.quot);
  }
  /// Access element in vector
  inline cref at2 (const ya_sizet i, const ya_sizet j) const {
    if (i==j)
      return 1;
    return 0;
  }
  using YA_Common<eltype,VM_Eye>::operator();

 protected:
  VecMatEye();
  const ya_sizet nrows;  // size of matrix
};

///  Constant Vector/Matrix of a Continuous Range of Integers
template<class eltype>
class VecMatCount : public YA_Common<eltype, VM_CountM> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatCount>::cref cref;

  /// Initialize a column vector with a given length with zero as the first int
  VecMatCount(const ya_sizet len) : count_start(0), ncolumns(1), nrows(len),
                                    nlength(len) {}
  /// Initialize a column vector with a given length with start as the first int
  VecMatCount(const ya_sizet len, const eltype start) : count_start(start),
                                    ncolumns(1), nrows(len), nlength(len) {}
  /// Initialize a matrix with start as the first int
  VecMatCount(const ya_sizet rowi, const ya_sizet coli, const eltype start) :
        count_start(start), ncolumns(coli), nrows(rowi), nlength(coli*rowi) {}
  /// Initialize a matrix with numbers from start to finish
  VecMatCount(const eltype start, const char * colon,
              const eltype finish) :  count_start(start), ncolumns(1),
              nrows((finish-start)+1), nlength(nrows) {}
    
  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return nrows; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return count_start+i;
  }
  using YA_Common<eltype,VM_CountM>::operator();

 protected:
  VecMatCount();
  const eltype count_start;
  const ya_sizet ncolumns;
  const ya_sizet nrows;
  const ya_sizet nlength;
};

///  Constant Row Vector of a Continuous Range of Integers
template<class eltype>
class VecMatCountV : public YA_Common<eltype, VM_Count> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatCountV>::cref cref;

  /// Initialize a column vector with a given length with zero as the first int
  VecMatCountV(const ya_sizet len) : count_start(0), ncolumns(len) {}
  /// Initialize a column vector with a given length with start as the first int
  VecMatCountV(const ya_sizet len, const eltype start) : count_start(start),
                                 ncolumns(len) {}
  /// Initialize a matrix with numbers from start to finish
  VecMatCountV(const eltype start, const char * colon,
               const eltype finish) : count_start(start),
                             ncolumns((finish-start)+1) {}
    
  /// Get the length of a vector
  inline ya_sizet numel() const { return ncolumns; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return count_start+i;
  }

  using YA_Common<eltype,VM_Count>::operator();

 protected:
  VecMatCountV();
  const eltype count_start;
  const ya_sizet ncolumns;
};

///  Constant Row Vector of a Continuous Range of Integers
template<class eltype>
class VecMatSeriesV : public YA_Common<eltype, VM_Series> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VecMatSeriesV>::cref cref;

  /// Initialize a column vector with a given length with zero as the first int
  VecMatSeriesV(const ya_sizet len) : count_start(0),count_incr(1),
                                      ncolumns(len) {}
  /// Initialize a column vector with a given length with start as the first int
  VecMatSeriesV(const ya_sizet len, const eltype start) : count_start(start),
            count_incr(1),ncolumns(len) {}
  /// Initialize a matrix with numbers from start to finish
  VecMatSeriesV(const eltype start, const char * colon,
                const eltype finish) : count_start(start), count_incr(1),
                              ncolumns((finish-start)+1) {}
  /// Initialize a matrix with numbers from start to finish
  VecMatSeriesV(const eltype start, const char * colon, const eltype incr, 
                const char * c2, const eltype finish) : 
    count_start(start), count_incr(incr), ncolumns((finish-start)/incr+1) {}
    
  /// Get the length of a vector
  inline ya_sizet numel() const { return ncolumns; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return count_start+(i*count_incr);
  }

  using YA_Common<eltype,VM_Series>::operator();

 protected:
  VecMatSeriesV();
  const eltype count_start;
  const eltype count_incr;
  const ya_sizet ncolumns;
};

///  Constant Row Vector of a Continuous Range of Integers
template<typename eltype, int count_start>
class VMCountFromI : public YA_Common<eltype,VM_FromI> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VMCountFromI>::cref cref;

  /// Initialize a column vector with a given length with zero as the first int
  VMCountFromI(const ya_sizet len) : ncolumns(len) {}

  /// Get the length of a vector
  inline ya_sizet numel() const { return ncolumns; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return ncolumns; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return static_cast<eltype>(count_start)+i;
  }

  using YA_Common<eltype,VM_FromI>::operator();
  
 protected:
  VMCountFromI();
  const ya_sizet ncolumns;  // size of matrix
};

///  Linearly spaced row vector
template<typename eltype>
class VMLinSpace : public YA_Common<eltype,VM_LinSpace> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_LinSpace>::cref cref;

  /// Initialize a linearly spaced vector from start to end of length 100
  VMLinSpace(const eltype start, const eltype end) : 
    nstart(start), niterate((end-start)/99.0), nlength(100) {}
  /// Initialize a linearly spaced vector from start to end of length len
  VMLinSpace(const eltype start, const eltype end, const ya_sizet len) : 
    nstart(start), niterate((end-start)/(len-1.0)), nlength(len) {}

  /// Get the length of a vector
  inline ya_sizet numel() const { return nlength; }
  /// Get the number of columns in the matrix
  inline ya_sizet cols() const { return nlength; }
  /// Get the number of rows in the matrix
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const {
    return nstart+(niterate*i);
  }

  using YA_Common<eltype,VM_LinSpace>::operator();
  
 protected:
  VMLinSpace();
  const eltype nstart;
  const eltype niterate;
  const ya_sizet nlength;
};

}
#endif
