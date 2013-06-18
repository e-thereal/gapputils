/***************************************************************************
                                vecmatscalar.h
                             -------------------
                               W. Michael Brown

  1 x 1 Matrix
 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Jan 9 2007
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef VECMATSCALAR_H
#define VECMATSCALAR_H

YALANAMESPACE

/// 1x1 Matrix
template<class eltype>
class VecMatScalar : public YA_Common<eltype, VM_Scalar> {
 public:
  /// Return types for indexing into this matrix
  typedef typename YA_Traits<VM_Scalar>::ref ref;
  typedef typename YA_Traits<VM_Scalar>::cref cref;
  typedef typename YA_Traits<VM_Scalar>::iter iter;
  typedef typename YA_Traits<VM_Scalar>::citer citer;

  /// Empty 0x0 matrix
  VecMatScalar();
  /// Set the value of the scalar
  VecMatScalar(const eltype value);
  /// Copy constructor
  VecMatScalar(const VecMatScalar &v);
  
  /// Allow assignt to a scalar
  inline operator eltype() { return array; }
  
  /// Not supported for scalar, generates error
  inline void clear();
  /// Generates error for length!=1
  inline void setup(const ya_sizet length);
  /// Generates error for rows!=1 and columns!=1
  inline void setup(const ya_sizet rows, const ya_sizet columns);

  /// Returns 1
  inline ya_sizet numel() const { return 1; }
  /// Returns 1
  inline ya_sizet cols() const { return 1; }
  /// Returns 1
  inline ya_sizet rows() const { return 1; }

  /// Access element in vector
  inline cref at (const ya_sizet i) const;
  inline ref at (const ya_sizet i);
  using YA_Common<eltype,VM_Scalar>::operator();
  using YA_Common<eltype,VM_Scalar>::operator=;

  inline const VM_Scalar& operator=(const VM_Scalar &v)
    { setup(v.rows(),v.cols()); ya_copy_matrix(*this,v); return *this; }

  /// Return iterator to begining of matrix
  inline iter begin() { return &array; }
  /// Return iterator to end of matrix
  inline iter end() { return &array+1; }
  /// Return iterator to begining of matrix
  inline citer begin() const { return &array; }
  /// Return iterator to end of matrix
  inline citer end() const { return &array+1; }
 protected:
  eltype array;
};

}
#endif
